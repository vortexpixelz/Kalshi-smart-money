"""
Percentile-based anomaly detector
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
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

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit the detector by computing percentile threshold(s)

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

        # Compute upper threshold
        self.upper_threshold_ = np.percentile(X_values, self.percentile, axis=0)

        # Compute lower threshold if two-sided
        if self.two_sided:
            self.lower_threshold_ = np.percentile(
                X_values, 100 - self.percentile, axis=0
            )

        self.is_fitted_ = True
        self.n_samples_seen_ = X_values.shape[0]

        return self

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores based on percentile rank

        Score is the percentile rank of the data point. For two-sided detection,
        uses distance from the median (50th percentile).

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

        if self.two_sided:
            # Compute distance from both thresholds
            upper_distance = np.maximum(X_values - self.upper_threshold_, 0)
            lower_distance = np.maximum(self.lower_threshold_ - X_values, 0)

            # Normalize by range
            threshold_range = self.upper_threshold_ - self.lower_threshold_
            threshold_range = np.where(threshold_range > 0, threshold_range, 1.0)

            upper_scores = upper_distance / threshold_range
            lower_scores = lower_distance / threshold_range

            # Take maximum
            scores = np.maximum(upper_scores, lower_scores)
        else:
            # One-sided: only upper tail
            distance = np.maximum(X_values - self.upper_threshold_, 0)

            # Normalize by threshold value
            threshold_safe = np.where(
                self.upper_threshold_ > 0, self.upper_threshold_, 1.0
            )
            scores = distance / threshold_safe

        # Take maximum across features
        if scores.ndim > 1:
            scores = np.max(scores, axis=1)
        else:
            scores = scores.flatten()

        return scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomaly)

        Args:
            X: Data to predict of shape (n_samples, n_features)

        Returns:
            Binary predictions of shape (n_samples,)
        """
        self.check_is_fitted()
        X = self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        if self.two_sided:
            # Anomaly if outside either threshold
            predictions = (
                (X_values > self.upper_threshold_) | (X_values < self.lower_threshold_)
            )
        else:
            # Anomaly if above upper threshold
            predictions = X_values > self.upper_threshold_

        # Any feature exceeds threshold -> anomaly
        if predictions.ndim > 1:
            predictions = np.any(predictions, axis=1)
        else:
            predictions = predictions.flatten()

        return predictions.astype(int)

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
        self.check_is_fitted()
        return (self.lower_threshold_, self.upper_threshold_)
