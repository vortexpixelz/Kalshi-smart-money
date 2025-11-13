"""
Main ensemble class coordinating multiple detectors and weighting strategies
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple
from ..detectors.base import BaseDetector
from .weighting import (
    UniformWeighting,
    MultiplicativeWeightsUpdate,
    ThompsonSamplingWeighting,
    UCBWeighting,
    ContextualUCBWeighting,
)


class AnomalyEnsemble:
    """
    Ensemble of anomaly detectors with adaptive weighting

    Coordinates multiple base detectors and combines their predictions using
    various weighting strategies optimized for minimal labeled data.

    Features:
        - Multiple base detectors (z-score, IQR, percentile, volume)
        - Adaptive weighting (uniform, MWU, Thompson, UCB)
        - Online learning with human feedback
        - Context-aware weighting with temporal features
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        weighting_method: str = 'thompson',
        weighting_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize anomaly ensemble

        Args:
            detectors: List of base anomaly detectors
            weighting_method: Weighting strategy ('uniform', 'mwu', 'thompson', 'ucb', 'contextual_ucb')
            weighting_params: Optional parameters for weighting method
        """
        self.detectors = detectors
        self.n_detectors = len(detectors)
        self.weighting_method = weighting_method

        # Initialize weighting strategy
        weighting_params = weighting_params or {}
        self.weighting = self._create_weighting(weighting_method, weighting_params)

        # Tracking
        self.n_samples_seen = 0
        self.n_feedbacks_received = 0

        # Store detector names for interpretability
        self.detector_names = [d.name for d in detectors]

    def _create_weighting(
        self, method: str, params: Dict[str, Any]
    ) -> Union[
        UniformWeighting,
        MultiplicativeWeightsUpdate,
        ThompsonSamplingWeighting,
        UCBWeighting,
        ContextualUCBWeighting,
    ]:
        """Create weighting strategy based on method name"""
        method = method.lower()

        if method == 'uniform':
            return UniformWeighting(self.n_detectors)
        elif method == 'mwu':
            learning_rate = params.get('learning_rate', 0.3)
            return MultiplicativeWeightsUpdate(self.n_detectors, learning_rate)
        elif method == 'thompson':
            alpha_prior = params.get('alpha_prior', 1.0)
            beta_prior = params.get('beta_prior', 1.0)
            return ThompsonSamplingWeighting(self.n_detectors, alpha_prior, beta_prior)
        elif method == 'ucb':
            exploration_param = params.get('exploration_param', 1.0)
            return UCBWeighting(self.n_detectors, exploration_param)
        elif method == 'contextual_ucb':
            exploration_param = params.get('exploration_param', 1.0)
            n_clusters = params.get('n_context_clusters', 10)
            return ContextualUCBWeighting(self.n_detectors, exploration_param, n_clusters)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit all base detectors

        Args:
            X: Training data
            y: Optional labels (for semi-supervised detectors)

        Returns:
            self
        """
        for detector in self.detectors:
            detector.fit(X, y)

        self.n_samples_seen = len(X)
        return self

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels using weighted ensemble

        Args:
            X: Data to predict
            context: Optional context features for contextual weighting

        Returns:
            Binary predictions
        """
        scores = self.score(X, context)
        predictions = (scores > 0.5).astype(int)
        return predictions

    def score(
        self, X: Union[np.ndarray, pd.DataFrame], context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute anomaly scores using weighted ensemble

        Args:
            X: Data to score
            context: Optional context features for contextual weighting

        Returns:
            Anomaly scores
        """
        detector_scores, _ = self._compute_normalized_score_matrix(X)

        # Combine using weights
        ensemble_scores = self.weighting.combine_scores(detector_scores)

        return ensemble_scores

    def update(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: np.ndarray,
        context: Optional[np.ndarray] = None,
    ):
        """
        Update ensemble weights based on feedback

        Args:
            X: Data samples
            y_true: Ground truth labels
            context: Optional context features

        Returns:
            Updated weights
        """
        detector_scores, _ = self._compute_normalized_score_matrix(X)

        # Update weights
        updated_weights = self.weighting.update_weights(
            detector_scores.T, y_true, context
        )

        self.n_feedbacks_received += len(y_true)

        return updated_weights

    def get_weights(self) -> np.ndarray:
        """Get current ensemble weights"""
        return self.weighting.get_weights()

    def get_detector_contributions(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Get individual detector scores and contributions

        Useful for interpretability and debugging.

        Args:
            X: Data to score

        Returns:
            Dictionary mapping detector names to scores
        """
        _, normalized_scores = self._compute_normalized_score_matrix(X)
        weights = self.get_weights()

        contributions = {}

        for i, (detector, scores) in enumerate(zip(self.detectors, normalized_scores)):
            contributions[detector.name] = {
                'scores': scores,
                'weight': weights[i],
                'weighted_scores': scores * weights[i],
            }

        return contributions

    def _compute_normalized_score_matrix(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Compute and normalize detector scores, returning stacked matrix and list."""

        normalized_scores: List[np.ndarray] = []

        for detector in self.detectors:
            scores = detector.score(X)
            min_score = np.min(scores)
            max_score = np.max(scores)

            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)

            normalized_scores.append(scores)

        detector_scores = np.column_stack(normalized_scores)

        return detector_scores, normalized_scores

    def get_ensemble_agreement(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute ensemble agreement (consistency) for each sample

        Agreement = fraction of detectors predicting anomaly.
        High agreement (>0.8) indicates high confidence.
        Low agreement (<0.5) indicates disagreement - good candidates for active learning.

        Args:
            X: Data to score

        Returns:
            Agreement scores in [0, 1]
        """
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(X)
            predictions.append(pred)

        # Stack predictions: shape (n_samples, n_detectors)
        predictions = np.column_stack(predictions)

        # Compute agreement (fraction voting for anomaly)
        agreement = predictions.mean(axis=1)

        return agreement

    def get_ensemble_diversity(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Compute ensemble diversity using pairwise disagreement

        Diversity = 1 - avg_pairwise(correlation)
        Higher diversity (>0.6) indicates complementary detection methods.

        Args:
            X: Data to score

        Returns:
            Diversity score in [0, 1]
        """
        scores = []
        for detector in self.detectors:
            score = detector.score(X)
            scores.append(score)

        # Stack scores: shape (n_samples, n_detectors)
        scores = np.column_stack(scores)

        # Compute pairwise correlations
        correlations = []
        for i in range(self.n_detectors):
            for j in range(i + 1, self.n_detectors):
                corr = np.corrcoef(scores[:, i], scores[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if len(correlations) == 0:
            return 0.0

        # Diversity = 1 - average correlation
        avg_correlation = np.mean(correlations)
        diversity = 1 - avg_correlation

        return diversity

    def reset_weights(self):
        """Reset ensemble weights to initial state"""
        self.weighting.reset()

    def get_state(self) -> Dict[str, Any]:
        """Get ensemble state for serialization"""
        return {
            'weighting_method': self.weighting_method,
            'weighting_state': self.weighting.get_state(),
            'n_samples_seen': self.n_samples_seen,
            'n_feedbacks_received': self.n_feedbacks_received,
            'detector_names': self.detector_names,
        }

    def set_state(self, state: Dict[str, Any]):
        """Set ensemble state from serialization"""
        self.weighting.set_state(state['weighting_state'])
        self.n_samples_seen = state['n_samples_seen']
        self.n_feedbacks_received = state['n_feedbacks_received']
