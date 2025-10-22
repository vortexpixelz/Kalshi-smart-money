"""
Active learning query strategies for minimal labeled data

Implements optimal selection of which anomalies to present for manual review:
- Query-by-Committee (QBC) - maximize ensemble disagreement
- BALD - Bayesian Active Learning by Disagreement
- Uncertainty Sampling - select near decision boundary
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from scipy.stats import entropy


class QueryStrategy(ABC):
    """Base class for active learning query strategies"""

    def __init__(self, batch_size: int = 10):
        """
        Initialize query strategy

        Args:
            batch_size: Number of samples to query at once
        """
        self.batch_size = batch_size

    @abstractmethod
    def select_queries(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        n_queries: Optional[int] = None,
    ) -> np.ndarray:
        """
        Select samples for manual review

        Args:
            X: Unlabeled data
            scores: Model scores/predictions
            n_queries: Number of queries to select (default: batch_size)

        Returns:
            Indices of selected samples
        """
        pass


class RandomSampling(QueryStrategy):
    """Random sampling baseline"""

    def select_queries(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        n_queries: Optional[int] = None,
    ) -> np.ndarray:
        """Randomly select samples"""
        n_queries = n_queries or self.batch_size
        n_samples = len(X)

        indices = np.random.choice(n_samples, size=min(n_queries, n_samples), replace=False)
        return indices


class UncertaintySampling(QueryStrategy):
    """
    Uncertainty sampling - select samples near decision boundary

    Useful for F1 optimization with limited feedback.
    Selects samples where model is most uncertain (scores near 0.5).
    """

    def __init__(self, batch_size: int = 10, margin_based: bool = False):
        """
        Initialize uncertainty sampling

        Args:
            batch_size: Number of samples to query
            margin_based: If True, use margin (difference between top 2 classes)
        """
        super().__init__(batch_size)
        self.margin_based = margin_based

    def select_queries(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        n_queries: Optional[int] = None,
    ) -> np.ndarray:
        """
        Select samples with highest uncertainty

        Args:
            X: Unlabeled data
            scores: Anomaly scores in [0, 1]
            n_queries: Number of queries

        Returns:
            Indices of most uncertain samples
        """
        n_queries = n_queries or self.batch_size

        # Uncertainty = distance from decision boundary (0.5)
        # Or: entropy of binary distribution
        if self.margin_based:
            # Margin: min(p, 1-p) - closer to 0.5 = more uncertain
            uncertainty = -np.abs(scores - 0.5)
        else:
            # Entropy: -p*log(p) - (1-p)*log(1-p)
            # Clamp to avoid log(0)
            p = np.clip(scores, 1e-10, 1 - 1e-10)
            uncertainty = -(p * np.log(p) + (1 - p) * np.log(1 - p))

        # Select top uncertain samples
        indices = np.argsort(uncertainty)[-n_queries:][::-1]

        return indices


class QueryByCommittee(QueryStrategy):
    """
    Query-by-Committee (QBC) - maximize ensemble disagreement

    The vote entropy criterion: H(y|x) = -Σ (V(y_i)/C) log(V(y_i)/C)
    where V(y_i) counts committee votes for class y_i.

    For ensemble of detection methods: select trades where methods
    disagree most strongly. Reduces required manual reviews by 50-80%
    compared to random sampling.
    """

    def __init__(self, batch_size: int = 10, disagreement_measure: str = 'vote_entropy'):
        """
        Initialize QBC

        Args:
            batch_size: Number of samples to query
            disagreement_measure: 'vote_entropy', 'kl_divergence', or 'variance'
        """
        super().__init__(batch_size)
        self.disagreement_measure = disagreement_measure

    def compute_vote_entropy(self, committee_predictions: np.ndarray) -> np.ndarray:
        """
        Compute vote entropy for each sample

        Args:
            committee_predictions: Shape (n_samples, n_committee_members)

        Returns:
            Vote entropy for each sample
        """
        n_samples, n_members = committee_predictions.shape

        # Count votes for each class
        vote_counts_1 = committee_predictions.sum(axis=1)
        vote_counts_0 = n_members - vote_counts_1

        # Compute probabilities
        p_1 = vote_counts_1 / n_members
        p_0 = vote_counts_0 / n_members

        # Compute entropy (clamp to avoid log(0))
        p_1_safe = np.clip(p_1, 1e-10, 1 - 1e-10)
        p_0_safe = np.clip(p_0, 1e-10, 1 - 1e-10)

        vote_entropy = -(p_1_safe * np.log(p_1_safe) + p_0_safe * np.log(p_0_safe))

        return vote_entropy

    def compute_variance(self, committee_scores: np.ndarray) -> np.ndarray:
        """
        Compute variance of committee scores

        Args:
            committee_scores: Shape (n_samples, n_committee_members)

        Returns:
            Variance for each sample
        """
        return np.var(committee_scores, axis=1)

    def select_queries(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        n_queries: Optional[int] = None,
        committee_predictions: Optional[np.ndarray] = None,
        committee_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select samples with maximum committee disagreement

        Args:
            X: Unlabeled data
            scores: Combined ensemble scores (unused if committee provided)
            n_queries: Number of queries
            committee_predictions: Binary predictions from each committee member
            committee_scores: Continuous scores from each committee member

        Returns:
            Indices of samples with maximum disagreement
        """
        n_queries = n_queries or self.batch_size

        if self.disagreement_measure == 'vote_entropy':
            if committee_predictions is None:
                raise ValueError("committee_predictions required for vote_entropy")
            disagreement = self.compute_vote_entropy(committee_predictions)

        elif self.disagreement_measure == 'variance':
            if committee_scores is None:
                raise ValueError("committee_scores required for variance")
            disagreement = self.compute_variance(committee_scores)

        else:
            raise ValueError(f"Unknown disagreement measure: {self.disagreement_measure}")

        # Select samples with highest disagreement
        indices = np.argsort(disagreement)[-n_queries:][::-1]

        return indices


class BALD(QueryStrategy):
    """
    Bayesian Active Learning by Disagreement

    Selects instances maximizing mutual information:
    x* = argmax I(y; θ | x, D) = H(y|x, D) - E_θ[H(y|x, θ)]

    The difference between marginal entropy and expected conditional entropy
    captures epistemic uncertainty - what the model doesn't know.
    """

    def __init__(self, batch_size: int = 10, n_mc_samples: int = 100):
        """
        Initialize BALD

        Args:
            batch_size: Number of samples to query
            n_mc_samples: Number of Monte Carlo samples from posterior
        """
        super().__init__(batch_size)
        self.n_mc_samples = n_mc_samples

    def compute_mutual_information(
        self, posterior_samples: np.ndarray
    ) -> np.ndarray:
        """
        Compute mutual information for BALD

        Args:
            posterior_samples: Shape (n_samples, n_mc_samples)
                             Predictions from posterior samples

        Returns:
            Mutual information for each sample
        """
        n_samples = posterior_samples.shape[0]

        # Marginal entropy: H(y|x, D)
        marginal_probs = posterior_samples.mean(axis=1)
        marginal_probs = np.clip(marginal_probs, 1e-10, 1 - 1e-10)
        marginal_entropy = -(
            marginal_probs * np.log(marginal_probs)
            + (1 - marginal_probs) * np.log(1 - marginal_probs)
        )

        # Expected conditional entropy: E_θ[H(y|x, θ)]
        posterior_samples_safe = np.clip(posterior_samples, 1e-10, 1 - 1e-10)
        conditional_entropies = -(
            posterior_samples_safe * np.log(posterior_samples_safe)
            + (1 - posterior_samples_safe) * np.log(1 - posterior_samples_safe)
        )
        expected_conditional_entropy = conditional_entropies.mean(axis=1)

        # Mutual information
        mutual_info = marginal_entropy - expected_conditional_entropy

        return mutual_info

    def select_queries(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        n_queries: Optional[int] = None,
        posterior_samples: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select samples maximizing mutual information

        Args:
            X: Unlabeled data
            scores: Model scores (unused if posterior_samples provided)
            n_queries: Number of queries
            posterior_samples: Predictions from posterior distribution

        Returns:
            Indices of samples with highest mutual information
        """
        n_queries = n_queries or self.batch_size

        if posterior_samples is None:
            # Fallback to uncertainty sampling if no posterior
            uncertainty = -np.abs(scores - 0.5)
            indices = np.argsort(uncertainty)[-n_queries:][::-1]
        else:
            # Compute BALD scores
            mutual_info = self.compute_mutual_information(posterior_samples)

            # Select samples with highest mutual information
            indices = np.argsort(mutual_info)[-n_queries:][::-1]

        return indices


class ExpectedModelChange(QueryStrategy):
    """
    Expected Model Change - select samples that will change model the most

    Estimates gradient of model parameters with respect to including a sample.
    """

    def __init__(self, batch_size: int = 10):
        super().__init__(batch_size)

    def select_queries(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        n_queries: Optional[int] = None,
        gradients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select samples with largest expected gradient magnitude

        Args:
            X: Unlabeled data
            scores: Model scores
            n_queries: Number of queries
            gradients: Gradient estimates for each sample

        Returns:
            Indices of samples with largest expected model change
        """
        n_queries = n_queries or self.batch_size

        if gradients is None:
            # Fallback: use score magnitude as proxy
            expected_change = np.abs(scores)
        else:
            # Use gradient magnitude
            expected_change = np.linalg.norm(gradients, axis=1)

        # Select samples with largest expected change
        indices = np.argsort(expected_change)[-n_queries:][::-1]

        return indices
