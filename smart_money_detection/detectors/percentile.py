"""Percentile-based anomaly detector."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import BaseDetector


class PercentileDetector(BaseDetector):
    """
    Detects anomalies based on percentile threshold

    A data point is considered anomalous if it exceeds a specified percentile
    of the training data distribution.

    Parameters:
        percentile: Percentile threshold (default: 95.0 for 95th percentile)
        rolling_window: Size of rolling window for online detection (default: None)
        two_sided: If True, detect both high and low extremes (default: True)
    """

    def __init__(
        self,
        percentile: float = 95.0,
        rolling_window: Optional[int] = None,
        two_sided: bool = True,
    ):
        super().__init__(name="Percentile")
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100")

        self.percentile = percentile
        self.rolling_window = rolling_window
        self.two_sided = two_sided

        self.upper_threshold_ = None
        self.lower_threshold_ = None

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Compute percentile thresholds."""

        self.upper_threshold_ = np.percentile(X, self.percentile, axis=0)
        if self.two_sided:
            self.lower_threshold_ = np.percentile(X, 100 - self.percentile, axis=0)
        else:
            self.lower_threshold_ = None

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return normalized distances from the learned percentile bounds."""

        if self.two_sided:
            upper_distance = np.maximum(X - self.upper_threshold_, 0)
            lower_distance = np.maximum(self.lower_threshold_ - X, 0)
            threshold_range = np.where(
                (self.upper_threshold_ - self.lower_threshold_) > 0,
                self.upper_threshold_ - self.lower_threshold_,
                1.0,
            )
            scores = np.maximum(upper_distance, lower_distance) / threshold_range
        else:
            distance = np.maximum(X - self.upper_threshold_, 0)
            threshold_safe = np.where(self.upper_threshold_ > 0, self.upper_threshold_, 1.0)
            scores = distance / threshold_safe

        if scores.ndim > 1:
            return np.max(scores, axis=1)
        return scores.reshape(-1)

    def _scores_to_predictions(
        self,
        scores: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        predictions = (np.asarray(scores) > 0).astype(int)
        return predictions

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        predictions, _ = self.predict_with_scores(X)
        return predictions

    def score_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Compute rolling percentile-based scores for online detection

        Args:
            X: Time series data

        Returns:
            Rolling anomaly scores
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X.flatten())

        if self.rolling_window is None:
            raise ValueError("rolling_window must be set for rolling computation")

        # Compute rolling percentile
        rolling_upper = X.rolling(window=self.rolling_window, min_periods=1).quantile(
            self.percentile / 100
        )

        if self.two_sided:
            rolling_lower = X.rolling(
                window=self.rolling_window, min_periods=1
            ).quantile((100 - self.percentile) / 100)

            # Compute distances
            upper_distance = np.maximum(X - rolling_upper, 0)
            lower_distance = np.maximum(rolling_lower - X, 0)

            # Normalize by range
            threshold_range = rolling_upper - rolling_lower
            threshold_range = threshold_range.replace(0, 1.0)

            upper_scores = upper_distance / threshold_range
            lower_scores = lower_distance / threshold_range

            scores = np.maximum(upper_scores, lower_scores)
        else:
            distance = np.maximum(X - rolling_upper, 0)
            rolling_upper_safe = rolling_upper.replace(0, 1.0)
            scores = distance / rolling_upper_safe

        return scores.values

    def predict_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict anomalies using rolling percentiles

        Args:
            X: Time series data

        Returns:
            Binary predictions
        """
        scores = self.score_rolling(X)
        predictions = (scores > 0).astype(int)
        return predictions

    def get_thresholds(self) -> tuple:
        """
        Get the computed threshold(s)

        Returns:
            Tuple of (lower_threshold, upper_threshold) or (None, upper_threshold)
        """
        self._check_is_fitted()
        return (self.lower_threshold_, self.upper_threshold_)
