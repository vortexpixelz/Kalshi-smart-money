"""
Interquartile Range (IQR) based anomaly detector
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
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

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit the detector by computing quartiles and bounds

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored (unsupervised method)

        Returns:
            self
        """
        X = self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        # Compute quartiles
        self.q1_ = np.percentile(X_values, self.q1_percentile * 100, axis=0)
        self.q3_ = np.percentile(X_values, self.q3_percentile * 100, axis=0)
        self.iqr_ = self.q3_ - self.q1_

        # Compute bounds
        self.lower_bound_ = self.q1_ - self.multiplier * self.iqr_
        self.upper_bound_ = self.q3_ + self.multiplier * self.iqr_

        self.is_fitted_ = True
        self.n_samples_seen_ = X_values.shape[0]

        return self

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores based on distance from IQR bounds

        Score is 0 if within bounds, otherwise the distance from the nearest bound
        normalized by IQR.

        Args:
            X: Data to score of shape (n_samples, n_features)

        Returns:
            Anomaly scores of shape (n_samples,)
        """
        self.check_is_fitted()
        X = self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        # Compute distances from bounds
        lower_distance = self.lower_bound_ - X_values
        upper_distance = X_values - self.upper_bound_

        # Distance is positive if outside bounds, zero if inside
        lower_distance = np.maximum(lower_distance, 0)
        upper_distance = np.maximum(upper_distance, 0)

        # Normalize by IQR (prevent division by zero)
        iqr_safe = np.where(self.iqr_ > 0, self.iqr_, 1.0)
        lower_distance_norm = lower_distance / iqr_safe
        upper_distance_norm = upper_distance / iqr_safe

        # Take maximum distance
        distances = np.maximum(lower_distance_norm, upper_distance_norm)

        # Take maximum across features
        if distances.ndim > 1:
            scores = np.max(distances, axis=1)
        else:
            scores = distances

        return scores

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
