"""Z-score based anomaly detector."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import BaseDetector
from smart_money_detection.utils.performance import track_performance


class ZScoreDetector(BaseDetector):
    """
    Detects anomalies using Z-score (standard deviations from mean)

    A data point is considered anomalous if its Z-score exceeds the threshold:
    |Z| = |(x - μ) / σ| > threshold

    Parameters:
        threshold: Number of standard deviations for anomaly threshold (default: 3.0)
        rolling_window: Size of rolling window for online detection (default: None)
        use_median: If True, use median and MAD instead of mean and std (robust)
    """

    def __init__(
        self,
        threshold: float = 3.0,
        rolling_window: Optional[int] = None,
        use_median: bool = False,
    ):
        super().__init__(name="ZScore")
        self.threshold = threshold
        self.rolling_window = rolling_window
        self.use_median = use_median

        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.mad_ = None

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Compute reference statistics for the detector."""

        if self.use_median:
            self.median_ = np.median(X, axis=0)
            self.mad_ = np.median(np.abs(X - self.median_), axis=0)
            self.mad_ = np.where(self.mad_ == 0, 1.0, self.mad_)
        else:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return absolute Z-scores for each row in *X*."""

        if self.use_median:
            z_scores = np.abs(X - self.median_) / (1.4826 * self.mad_)
        else:
            z_scores = np.abs(X - self.mean_) / self.std_

        if z_scores.ndim > 1:
            return np.max(z_scores, axis=1)
        return z_scores.reshape(-1)

    def _scores_to_predictions(
        self,
        scores: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> np.ndarray:
        predictions = (np.asarray(scores) > self.threshold).astype(int)
        return predictions

    @track_performance("detector.zscore.predict", metadata={"detector": "zscore"})
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        predictions, _ = self.predict_with_scores(X)
        return predictions

    @track_performance("detector.zscore.score_rolling", metadata={"detector": "zscore"})
    def score_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Compute rolling Z-scores for online detection

        Args:
            X: Time series data

        Returns:
            Rolling Z-scores
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X.flatten())

        if self.rolling_window is None:
            raise ValueError("rolling_window must be set for rolling computation")

        # Compute rolling statistics
        rolling_mean = X.rolling(window=self.rolling_window, min_periods=1).mean()
        rolling_std = X.rolling(window=self.rolling_window, min_periods=1).std()

        # Prevent division by zero
        rolling_std = rolling_std.replace(0, 1.0)

        # Compute Z-scores
        z_scores = np.abs((X - rolling_mean) / rolling_std)

        return z_scores.values

    @track_performance("detector.zscore.predict_rolling", metadata={"detector": "zscore"})
    def predict_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict anomalies using rolling Z-scores

        Args:
            X: Time series data

        Returns:
            Binary predictions
        """
        scores = self.score_rolling(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions
