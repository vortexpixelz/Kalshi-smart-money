"""Interquartile Range (IQR) based anomaly detector."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import BaseDetector


class IQRDetector(BaseDetector):
    """
    Detects anomalies using Interquartile Range (IQR) method

    A data point is considered anomalous if:
    x < Q1 - multiplier × IQR  or  x > Q3 + multiplier × IQR

    where IQR = Q3 - Q1

    Parameters:
        multiplier: IQR multiplier for outlier bounds (default: 1.5)
        rolling_window: Size of rolling window for online detection (default: None)
        q1: Lower quartile (default: 0.25)
        q3: Upper quartile (default: 0.75)
    """

    def __init__(
        self,
        multiplier: float = 1.5,
        rolling_window: Optional[int] = None,
        q1: float = 0.25,
        q3: float = 0.75,
    ):
        super().__init__(name="IQR")
        self.multiplier = multiplier
        self.rolling_window = rolling_window
        self.q1_percentile = q1
        self.q3_percentile = q3

        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Compute quartiles and anomaly bounds."""

        self.q1_ = np.percentile(X, self.q1_percentile * 100, axis=0)
        self.q3_ = np.percentile(X, self.q3_percentile * 100, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_bound_ = self.q1_ - self.multiplier * self.iqr_
        self.upper_bound_ = self.q3_ + self.multiplier * self.iqr_

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return normalized distance outside of the IQR bounds."""

        lower_distance = np.maximum(self.lower_bound_ - X, 0)
        upper_distance = np.maximum(X - self.upper_bound_, 0)
        iqr_safe = np.where(self.iqr_ > 0, self.iqr_, 1.0)
        distances = np.maximum(lower_distance, upper_distance) / iqr_safe

        if distances.ndim > 1:
            return np.max(distances, axis=1)
        return distances.reshape(-1)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomaly)

        Args:
            X: Data to predict of shape (n_samples, n_features)

        Returns:
            Binary predictions of shape (n_samples,)
        """
        scores = self.score(X)
        # Any positive score means outside bounds (anomaly)
        predictions = (scores > 0).astype(int)
        return predictions

    def score_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Compute rolling IQR-based scores for online detection

        Args:
            X: Time series data

        Returns:
            Rolling anomaly scores
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X.flatten())

        if self.rolling_window is None:
            raise ValueError("rolling_window must be set for rolling computation")

        # Compute rolling quartiles
        rolling_q1 = X.rolling(window=self.rolling_window, min_periods=1).quantile(
            self.q1_percentile
        )
        rolling_q3 = X.rolling(window=self.rolling_window, min_periods=1).quantile(
            self.q3_percentile
        )
        rolling_iqr = rolling_q3 - rolling_q1

        # Compute bounds
        lower_bound = rolling_q1 - self.multiplier * rolling_iqr
        upper_bound = rolling_q3 + self.multiplier * rolling_iqr

        # Compute distances
        lower_distance = np.maximum(lower_bound - X, 0)
        upper_distance = np.maximum(X - upper_bound, 0)

        # Normalize by IQR
        rolling_iqr_safe = rolling_iqr.replace(0, 1.0)
        scores = np.maximum(lower_distance, upper_distance) / rolling_iqr_safe

        return scores.values

    def predict_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict anomalies using rolling IQR

        Args:
            X: Time series data

        Returns:
            Binary predictions
        """
        scores = self.score_rolling(X)
        predictions = (scores > 0).astype(int)
        return predictions

    def get_bounds(self) -> tuple:
        """
        Get the computed lower and upper bounds

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        self.check_is_fitted()
        return (self.lower_bound_, self.upper_bound_)
