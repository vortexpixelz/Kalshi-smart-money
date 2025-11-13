"""Main ensemble class coordinating multiple detectors and weighting strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd

from ..detectors.base import BaseDetector
from .weighting import (
    ContextualUCBWeighting,
    MultiplicativeWeightsUpdate,
    ThompsonSamplingWeighting,
    UCBWeighting,
    UniformWeighting,
)


class ProbabilityCalibrator(Protocol):
    """Minimal protocol implemented by probability calibration helpers."""

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        """Fit the calibrator using ensemble scores and ground-truth labels."""

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw ensemble scores into calibrated probabilities."""


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

        # Optional calibration hook
        self._calibrator: Optional[ProbabilityCalibrator] = None

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
        array = self._to_numpy(X)
        for detector in self.detectors:
            detector.fit(array, y)

        self.n_samples_seen = array.shape[0]
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
        return (scores > 0.5).astype(int)

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
        array = self._to_numpy(X)
        detector_scores = self._collect_scores(array)
        ensemble_scores = self.weighting.combine_scores(detector_scores)

        if self._calibrator is not None:
            ensemble_scores = self._calibrator.transform(ensemble_scores)

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
        array = self._to_numpy(X)
        detector_scores = self._collect_scores(array)
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
        contributions = {}
        weights = self.get_weights()

        array = self._to_numpy(X)
        detector_scores = self._collect_scores(array)

        for i, detector in enumerate(self.detectors):
            scores = detector_scores[:, i]
            contributions[detector.name] = {
                'scores': scores,
                'weight': weights[i],
                'weighted_scores': scores * weights[i],
            }

        return contributions

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
        array = self._to_numpy(X)
        predictions = np.column_stack([detector.predict(array) for detector in self.detectors])
        return predictions.mean(axis=1)

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
        array = self._to_numpy(X)
        scores = self._collect_scores(array)

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

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    def set_calibrator(self, calibrator: ProbabilityCalibrator) -> None:
        """Attach a calibrator that will post-process ensemble scores."""

        self._calibrator = calibrator

    def fit_calibrator(
        self, scores: np.ndarray, y_true: np.ndarray
    ) -> None:
        """Fit the attached calibrator using labelled scores."""

        if self._calibrator is None:
            raise ValueError("No calibrator has been set on the ensemble")
        self._calibrator.fit(scores, y_true)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convert *X* to a floating point NumPy array once per call path."""

        if isinstance(X, pd.DataFrame):
            array = X.to_numpy(copy=False)
        else:
            array = np.asarray(X)

        if array.ndim == 1:
            array = array.reshape(-1, 1)

        return array.astype(float, copy=False)

    def _collect_scores(self, X: np.ndarray) -> np.ndarray:
        """Collect and normalize detector scores for the provided samples."""

        scores = np.column_stack([detector.score(X) for detector in self.detectors])
        min_vals = scores.min(axis=0)
        max_vals = scores.max(axis=0)
        ranges = max_vals - min_vals
        non_zero = ranges > 0
        if np.any(non_zero):
            scores[:, non_zero] = (scores[:, non_zero] - min_vals[non_zero]) / ranges[non_zero]

        return scores
