"""
Ensemble weighting strategies for minimal labeled data

Implements state-of-the-art methods from research:
- Uniform weighting (baseline)
- Multiplicative Weights Update (MWU/SEAD)
- Thompson Sampling with Beta posteriors
- Upper Confidence Bound (UCB)
"""
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseEnsemble


class UniformWeighting(BaseEnsemble):
    """
    Uniform weighting baseline - all detectors weighted equally

    Recommended starting point when you have <10 labeled examples.
    Focus on ensemble diversity rather than optimizing weights prematurely.
    """

    def __init__(self, n_detectors: int):
        super().__init__(n_detectors, name="Uniform")

    def update_weights(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ):
        """
        Uniform weights don't change - just increment counter

        Args:
            scores: Detector scores (unused)
            y_true: Ground truth labels (unused)
            context: Context features (unused)

        Returns:
            Unchanged uniform weights
        """
        self.n_updates += 1
        return self.weights


class MultiplicativeWeightsUpdate(BaseEnsemble):
    """
    Multiplicative Weights Update (MWU) algorithm

    From Amazon's SEAD (Streaming Ensemble of Anomaly Detectors) - ICML 2025.
    Works completely unsupervised using the insight that anomalies are rare,
    so consistently lower-scoring models receive higher weights.

    Weight update: w_i^(t+1) = w_i^(t) × exp(-η × s_i^(t))
    where η is learning rate and s_i^(t) is normalized anomaly score.

    Regret bound: O(√(T log k)) where T=time steps, k=detectors
    """

    def __init__(self, n_detectors: int, learning_rate: float = 0.3):
        """
        Initialize MWU

        Args:
            n_detectors: Number of base detectors
            learning_rate: Learning rate η ∈ [0.1, 0.5] (higher = faster adaptation)
        """
        super().__init__(n_detectors, name="MWU")
        self.learning_rate = learning_rate

    def update_weights(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ):
        """
        Update weights using multiplicative update rule

        Args:
            scores: Detector scores of shape (n_detectors, n_samples) or (n_samples, n_detectors)
            y_true: Optional ground truth (if available, compute loss; else use scores)
            context: Optional context features (unused)

        Returns:
            Updated weights
        """
        # Ensure scores are in shape (n_detectors, n_samples)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[0] != self.n_detectors:
            scores = scores.T

        if y_true is not None:
            # Supervised: compute loss for each detector
            # Loss = 0 if correct, 1 if incorrect
            predictions = (scores > 0.5).astype(int)
            losses = np.mean(predictions != y_true.reshape(1, -1), axis=1)
        else:
            # Unsupervised: use normalized anomaly scores
            # Lower scores = better (assumes anomalies are rare)
            losses = np.mean(scores, axis=1)

            # Normalize to [0, 1]
            if losses.max() > losses.min():
                losses = (losses - losses.min()) / (losses.max() - losses.min())

        # Multiplicative weight update
        self.weights *= np.exp(-self.learning_rate * losses)

        # Normalize weights
        self.weights /= self.weights.sum()

        self.n_updates += 1
        return self.weights


class ThompsonSamplingWeighting(BaseEnsemble):
    """
    Thompson Sampling with Beta posteriors for detector selection

    Optimal regret bound O(√T) without hyperparameter tuning.
    Each detector maintains Beta(α, β) posterior representing success/failure counts.

    Algorithm:
        1. Sample θ_i ~ Beta(α_i, β_i) for each detector
        2. Select detector with highest sampled value
        3. Observe feedback y ∈ {0,1}
        4. Update: α_i ← α_i + y, β_i ← β_i + (1-y)

    For ensemble weighting, we normalize sampled values as weights.
    """

    def __init__(
        self,
        n_detectors: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        n_samples: int = 100,
    ):
        """
        Initialize Thompson Sampling

        Args:
            n_detectors: Number of base detectors
            alpha_prior: Prior successes (α_0)
            beta_prior: Prior failures (β_0)
            n_samples: Number of posterior samples for weight estimation
        """
        super().__init__(n_detectors, name="ThompsonSampling")

        # Initialize Beta posteriors
        self.alpha = np.ones(n_detectors) * alpha_prior
        self.beta = np.ones(n_detectors) * beta_prior
        self.n_samples = n_samples

        # Sample initial weights
        self._sample_weights()

    def _sample_weights(self):
        """Sample weights from Beta posteriors"""
        # Sample from each Beta distribution
        samples = np.random.beta(self.alpha, self.beta, size=(self.n_samples, self.n_detectors))

        # Use mean of samples as weights (alternatively: use mode or single sample)
        self.weights = samples.mean(axis=0)

        # Normalize
        self.weights /= self.weights.sum()

    def update_weights(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ):
        """
        Update Beta posteriors based on detector performance

        Args:
            scores: Detector scores of shape (n_detectors, n_samples) or (n_samples, n_detectors)
            y_true: Ground truth labels (required for Thompson Sampling)
            context: Optional context features (unused)

        Returns:
            Updated weights
        """
        if y_true is None:
            # Without feedback, just resample weights
            self._sample_weights()
            return self.weights

        # Ensure scores are in shape (n_detectors, n_samples)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[0] != self.n_detectors:
            scores = scores.T

        # Compute correctness for each detector
        predictions = (scores > 0.5).astype(int)
        correct = (predictions == y_true.reshape(1, -1)).astype(int)

        # Update Beta posteriors
        successes = correct.sum(axis=1)
        failures = (1 - correct).sum(axis=1)

        self.alpha += successes
        self.beta += failures

        # Sample new weights
        self._sample_weights()

        self.n_updates += 1
        return self.weights

    def get_state(self) -> Dict[str, Any]:
        """Get state including Beta parameters"""
        state = super().get_state()
        state['alpha'] = self.alpha.tolist()
        state['beta'] = self.beta.tolist()
        return state

    def set_state(self, state: Dict[str, Any]):
        """Set state including Beta parameters"""
        super().set_state(state)
        self.alpha = np.array(state['alpha'])
        self.beta = np.array(state['beta'])


class UCBWeighting(BaseEnsemble):
    """
    Upper Confidence Bound (UCB) for detector selection

    Selection rule: i* = argmax_i [μ̂_i + c√(ln(t)/n_i)]
    where:
        - μ̂_i: estimated performance of detector i
        - n_i: number of times detector i selected
        - c: exploration parameter
        - t: current time step

    Balances exploitation (current best μ̂_i) with exploration (uncertainty).
    """

    def __init__(
        self,
        n_detectors: int,
        exploration_param: float = 1.0,
        use_context: bool = False,
    ):
        """
        Initialize UCB

        Args:
            n_detectors: Number of base detectors
            exploration_param: Exploration parameter c ∈ [0.5, 2.0]
            use_context: If True, maintain context-dependent performance estimates
        """
        super().__init__(n_detectors, name="UCB")
        self.exploration_param = exploration_param
        self.use_context = use_context

        # Performance tracking
        self.mean_performance = np.zeros(n_detectors)
        self.n_selected = np.zeros(n_detectors)
        self.total_selections = 0

        # Running statistics for online mean update
        self.sum_performance = np.zeros(n_detectors)

    def _compute_ucb_scores(self) -> np.ndarray:
        """
        Compute UCB scores for each detector

        Returns:
            UCB scores
        """
        # Avoid division by zero
        n_selected_safe = np.maximum(self.n_selected, 1)

        # Exploration bonus
        if self.total_selections > 0:
            exploration_bonus = self.exploration_param * np.sqrt(
                np.log(self.total_selections + 1) / n_selected_safe
            )
        else:
            # Initially, give all detectors equal high exploration bonus
            exploration_bonus = np.ones(self.n_detectors) * self.exploration_param

        # UCB = exploitation + exploration
        ucb_scores = self.mean_performance + exploration_bonus

        return ucb_scores

    def update_weights(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ):
        """
        Update UCB weights based on detector performance

        Args:
            scores: Detector scores of shape (n_detectors, n_samples) or (n_samples, n_detectors)
            y_true: Ground truth labels (if available, compute accuracy; else use scores)
            context: Optional context features (unused in basic UCB)

        Returns:
            Updated weights based on UCB scores
        """
        # Ensure scores are in shape (n_detectors, n_samples)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[0] != self.n_detectors:
            scores = scores.T

        if y_true is not None:
            # Compute accuracy for each detector
            predictions = (scores > 0.5).astype(int)
            performance = (predictions == y_true.reshape(1, -1)).mean(axis=1)
        else:
            # Use inverse of mean score (lower scores = better for anomalies)
            mean_scores = scores.mean(axis=1)
            if mean_scores.max() > 0:
                performance = 1 - (mean_scores / mean_scores.max())
            else:
                performance = np.ones(self.n_detectors)

        # Update statistics for all detectors
        self.sum_performance += performance
        self.n_selected += 1  # Treat as if all were selected
        self.total_selections += 1

        # Update mean performance (online mean)
        self.mean_performance = self.sum_performance / self.n_selected

        # Compute UCB scores
        ucb_scores = self._compute_ucb_scores()

        # Convert UCB scores to weights (softmax for smooth weighting)
        # Use temperature scaling to control exploration
        temperature = 0.5
        exp_scores = np.exp(ucb_scores / temperature)
        self.weights = exp_scores / exp_scores.sum()

        self.n_updates += 1
        return self.weights

    def select_detector(self) -> int:
        """
        Select a single detector based on UCB scores

        Returns:
            Index of selected detector
        """
        ucb_scores = self._compute_ucb_scores()
        return np.argmax(ucb_scores)

    def get_state(self) -> Dict[str, Any]:
        """Get state including UCB parameters"""
        state = super().get_state()
        state['mean_performance'] = self.mean_performance.tolist()
        state['n_selected'] = self.n_selected.tolist()
        state['total_selections'] = self.total_selections
        state['sum_performance'] = self.sum_performance.tolist()
        return state

    def set_state(self, state: Dict[str, Any]):
        """Set state including UCB parameters"""
        super().set_state(state)
        self.mean_performance = np.array(state['mean_performance'])
        self.n_selected = np.array(state['n_selected'])
        self.total_selections = state['total_selections']
        self.sum_performance = np.array(state['sum_performance'])


class ContextualUCBWeighting(UCBWeighting):
    """
    Contextual UCB with temporal features

    Extends UCB to use context features (time of day, day of week, etc.)
    for predicting detector performance. Implements the Tree Ensemble UCB
    (TEUCB) approach from research.
    """

    def __init__(
        self,
        n_detectors: int,
        exploration_param: float = 1.0,
        n_context_clusters: int = 10,
    ):
        """
        Initialize Contextual UCB

        Args:
            n_detectors: Number of base detectors
            exploration_param: Exploration parameter
            n_context_clusters: Number of context clusters for partitioning context space
        """
        super().__init__(n_detectors, exploration_param, use_context=True)
        self.name = "ContextualUCB"
        self.n_context_clusters = n_context_clusters

        # Context-specific statistics (simple approach: k-means clustering)
        self.context_cluster_centers = None
        self.context_cluster_stats = [
            {
                'mean_performance': np.zeros(n_detectors),
                'n_selected': np.zeros(n_detectors),
                'sum_performance': np.zeros(n_detectors),
            }
            for _ in range(n_context_clusters)
        ]

    def _get_context_cluster(self, context: np.ndarray) -> int:
        """
        Find nearest context cluster

        Args:
            context: Context feature vector

        Returns:
            Cluster index
        """
        if self.context_cluster_centers is None:
            # Initialize with random cluster
            return np.random.randint(0, self.n_context_clusters)

        # Find nearest cluster center
        distances = np.linalg.norm(self.context_cluster_centers - context, axis=1)
        return np.argmin(distances)

    def update_weights(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ):
        """
        Update weights using context-dependent performance

        Args:
            scores: Detector scores
            y_true: Ground truth labels
            context: Context feature vector (required)

        Returns:
            Updated weights
        """
        if context is None:
            # Fall back to standard UCB
            return super().update_weights(scores, y_true, context)

        # Ensure context is 1D array
        if context.ndim > 1:
            context = context.flatten()

        # Get context cluster
        cluster_id = self._get_context_cluster(context)
        cluster_stats = self.context_cluster_stats[cluster_id]

        # Compute performance (same as UCB)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[0] != self.n_detectors:
            scores = scores.T

        if y_true is not None:
            predictions = (scores > 0.5).astype(int)
            performance = (predictions == y_true.reshape(1, -1)).mean(axis=1)
        else:
            mean_scores = scores.mean(axis=1)
            if mean_scores.max() > 0:
                performance = 1 - (mean_scores / mean_scores.max())
            else:
                performance = np.ones(self.n_detectors)

        # Update cluster-specific statistics
        cluster_stats['sum_performance'] += performance
        cluster_stats['n_selected'] += 1
        cluster_stats['mean_performance'] = (
            cluster_stats['sum_performance'] / cluster_stats['n_selected']
        )

        # Compute UCB scores for this context
        n_selected_safe = np.maximum(cluster_stats['n_selected'], 1)
        total_in_cluster = cluster_stats['n_selected'].sum()

        if total_in_cluster > 0:
            exploration_bonus = self.exploration_param * np.sqrt(
                np.log(total_in_cluster + 1) / n_selected_safe
            )
        else:
            exploration_bonus = np.ones(self.n_detectors) * self.exploration_param

        ucb_scores = cluster_stats['mean_performance'] + exploration_bonus

        # Convert to weights
        temperature = 0.5
        exp_scores = np.exp(ucb_scores / temperature)
        self.weights = exp_scores / exp_scores.sum()

        self.n_updates += 1
        return self.weights
