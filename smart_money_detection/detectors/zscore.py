"""
Z-score based anomaly detector
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
from .base import BaseDetector


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

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit the detector by computing mean and standard deviation

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

        if self.use_median:
            # Robust statistics using median and MAD
            self.median_ = np.median(X_values, axis=0)
            # MAD = median absolute deviation
            self.mad_ = np.median(np.abs(X_values - self.median_), axis=0)
            # Prevent division by zero
            self.mad_[self.mad_ == 0] = 1.0
        else:
            # Standard statistics
            self.mean_ = np.mean(X_values, axis=0)
            self.std_ = np.std(X_values, axis=0)
            # Prevent division by zero
            self.std_[self.std_ == 0] = 1.0

        self.is_fitted_ = True
        self.n_samples_seen_ = X_values.shape[0]

        return self

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores using absolute Z-score

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

        if self.use_median:
            # Robust Z-score
            z_scores = np.abs(X_values - self.median_) / (1.4826 * self.mad_)
        else:
            # Standard Z-score
            z_scores = np.abs(X_values - self.mean_) / self.std_

        # Take maximum Z-score across features
        if z_scores.ndim > 1:
            scores = np.max(z_scores, axis=1)
        else:
            scores = z_scores

        return scores

    def _scores_to_predictions(
        self,
        scores: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> np.ndarray:
        predictions = (np.asarray(scores) > self.threshold).astype(int)
        return predictions

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        predictions, _ = self.predict_with_scores(X)
        return predictions

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
