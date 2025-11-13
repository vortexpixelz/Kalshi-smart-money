"""
Relative volume anomaly detector for detecting large trades
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
from .base import BaseDetector


class RelativeVolumeDetector(BaseDetector):
    """
    Detects anomalies based on relative trade volume

    A trade is considered anomalous if its volume exceeds a multiple of the
    median/mean volume in the rolling window.

    Parameters:
        threshold_multiplier: Volume multiplier threshold (default: 3.0 for 3x median)
        rolling_window: Size of rolling window for baseline (default: 100)
        use_median: If True, use median instead of mean (more robust)
        absolute_threshold: Optional absolute volume threshold
    """

    def __init__(
        self,
        threshold_multiplier: float = 3.0,
        rolling_window: int = 100,
        use_median: bool = True,
        absolute_threshold: Optional[float] = None,
    ):
        super().__init__(name="RelativeVolume")
        self.threshold_multiplier = threshold_multiplier
        self.rolling_window = rolling_window
        self.use_median = use_median
        self.absolute_threshold = absolute_threshold

        self.baseline_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit the detector by computing baseline volume

        Args:
            X: Training volume data of shape (n_samples, 1) or (n_samples,)
            y: Ignored (unsupervised method)

        Returns:
            self
        """
        X = self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            X_values = X.values.flatten()
        else:
            X_values = X.flatten()

        # Compute baseline
        if self.use_median:
            self.baseline_ = np.median(X_values)
        else:
            self.baseline_ = np.mean(X_values)

        # Prevent zero baseline
        if self.baseline_ == 0:
            self.baseline_ = 1.0

        self.is_fitted_ = True
        self.n_samples_seen_ = len(X_values)

        return self

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores based on volume ratio

        Score is the ratio of volume to baseline.

        Args:
            X: Volume data to score of shape (n_samples, 1) or (n_samples,)

        Returns:
            Anomaly scores of shape (n_samples,)
        """
        self.check_is_fitted()
        X = self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            X_values = X.values.flatten()
        else:
            X_values = X.flatten()

        # Compute volume ratio
        scores = X_values / self.baseline_

        return scores

    def _scores_to_predictions(
        self,
        scores: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> np.ndarray:
        scores_arr = np.asarray(scores)
        predictions = scores_arr > self.threshold_multiplier

        if self.absolute_threshold is not None and X is not None:
            if isinstance(X, pd.DataFrame):
                X_values = X.values.flatten()
            else:
                X_values = np.asarray(X).flatten()
            predictions = np.logical_or(predictions, X_values > self.absolute_threshold)

        return predictions.astype(int)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        predictions, _ = self.predict_with_scores(X)
        return predictions

    def score_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Compute rolling relative volume scores for online detection

        Args:
            X: Time series volume data

        Returns:
            Rolling anomaly scores (volume ratios)
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X.flatten())

        # Compute rolling baseline
        if self.use_median:
            rolling_baseline = X.rolling(
                window=self.rolling_window, min_periods=1
            ).median()
        else:
            rolling_baseline = X.rolling(
                window=self.rolling_window, min_periods=1
            ).mean()

        # Prevent division by zero
        rolling_baseline = rolling_baseline.replace(0, 1.0)

        # Compute volume ratios
        scores = X / rolling_baseline

        return scores.values

    def predict_rolling(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict anomalies using rolling volume baseline

        Args:
            X: Time series volume data

        Returns:
            Binary predictions
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X.flatten())

        scores = self.score_rolling(X)

        # Check relative threshold
        relative_anomaly = scores > self.threshold_multiplier

        # Check absolute threshold if provided
        if self.absolute_threshold is not None:
            absolute_anomaly = X > self.absolute_threshold
            predictions = (relative_anomaly | absolute_anomaly).astype(int)
        else:
            predictions = relative_anomaly.astype(int)

        return predictions.astype(int)

    def get_baseline(self) -> float:
        """
        Get the computed baseline volume

        Returns:
            Baseline volume value
        """
        self.check_is_fitted()
        return self.baseline_


class MarketCapAwareVolumeDetector(RelativeVolumeDetector):
    """
    Volume detector that adjusts thresholds based on market characteristics

    For large/liquid markets, uses higher absolute thresholds.
    For small/illiquid markets, uses higher relative multipliers.
    """

    def __init__(
        self,
        major_market_threshold: float = 10000.0,
        niche_market_threshold: float = 1000.0,
        major_market_multiplier: float = 2.0,
        niche_market_multiplier: float = 5.0,
        rolling_window: int = 100,
        use_median: bool = True,
    ):
        super().__init__(
            threshold_multiplier=major_market_multiplier,
            rolling_window=rolling_window,
            use_median=use_median,
        )
        self.name = "MarketCapAwareVolume"
        self.major_market_threshold = major_market_threshold
        self.niche_market_threshold = niche_market_threshold
        self.major_market_multiplier = major_market_multiplier
        self.niche_market_multiplier = niche_market_multiplier

        self.is_major_market_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        market_size: Optional[float] = None,
    ):
        """
        Fit the detector with market size awareness

        Args:
            X: Training volume data
            y: Ignored
            market_size: Total market size (open interest, total volume, etc.)

        Returns:
            self
        """
        # Determine if major market
        if market_size is not None:
            self.is_major_market_ = market_size > self.major_market_threshold * 10
        else:
            # Estimate from data
            X_values = X.values.flatten() if isinstance(X, pd.DataFrame) else X.flatten()
            median_volume = np.median(X_values)
            self.is_major_market_ = median_volume > self.major_market_threshold / 10

        # Adjust parameters based on market type
        if self.is_major_market_:
            self.threshold_multiplier = self.major_market_multiplier
            self.absolute_threshold = self.major_market_threshold
        else:
            self.threshold_multiplier = self.niche_market_multiplier
            self.absolute_threshold = self.niche_market_threshold

        # Fit base detector
        return super().fit(X, y)
