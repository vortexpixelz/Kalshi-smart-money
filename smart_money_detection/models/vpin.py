"""
Volume-Synchronized Probability of Informed Trading (VPIN)

From Easley, López de Prado, and O'Hara (2012, Review of Financial Studies).
VPIN provides real-time toxicity measurement without complex MLE optimization.

Key innovation: operate in volume-time rather than clock-time, matching information arrival speed.

VPIN correlation with next-bucket absolute returns reached 0.40 for E-mini S&P 500 futures.
CDF(VPIN) > 0.90 successfully predicted Flash Crash hours in advance.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading

    Algorithm:
        1. Bucket trades into equal volume chunks V (typically 1/50 of daily volume)
        2. For each bucket τ, classify buy vs sell volume:
           V_τ^B = Σ V_i × Φ((P_i - P_{i-1})/σ_ΔP)
           where Φ = standard normal CDF
        3. Compute VPIN over n buckets (typically n=50):
           VPIN = (1/nV) Σ_{τ=1}^n |V_τ^S - V_τ^B|

    Interpretation:
        - VPIN ∈ [0,1], higher = more toxic/informed flow
        - VPIN > 0.75 indicates elevated informed trading
        - Track CDF(VPIN) for extreme event prediction
    """

    def __init__(
        self,
        n_buckets: int = 50,
        volume_bucket_size: Optional[float] = None,
        bucket_pct_of_daily: float = 0.02,
    ):
        """
        Initialize VPIN calculator

        Args:
            n_buckets: Number of volume buckets for VPIN calculation (default: 50)
            volume_bucket_size: Fixed volume per bucket (if None, use bucket_pct_of_daily)
            bucket_pct_of_daily: Bucket size as fraction of daily volume (default: 0.02 = 2%)
        """
        self.n_buckets = n_buckets
        self.volume_bucket_size = volume_bucket_size
        self.bucket_pct_of_daily = bucket_pct_of_daily

        # Fitted parameters
        self.price_volatility_ = None
        self.daily_volume_ = None

    def fit(
        self,
        prices: Union[np.ndarray, pd.Series],
        volumes: Union[np.ndarray, pd.Series],
        daily_volume: Optional[float] = None,
    ):
        """
        Fit VPIN parameters from historical data

        Args:
            prices: Historical prices
            volumes: Historical volumes
            daily_volume: Total daily volume (if None, estimated from data)

        Returns:
            self
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        if isinstance(volumes, pd.Series):
            volumes = volumes.values

        # Estimate price volatility (for bulk classification)
        price_changes = np.diff(prices)
        self.price_volatility_ = np.std(price_changes)

        # Estimate daily volume
        if daily_volume is not None:
            self.daily_volume_ = daily_volume
        else:
            self.daily_volume_ = volumes.sum()

        # Set bucket size if not provided
        if self.volume_bucket_size is None:
            self.volume_bucket_size = self.daily_volume_ * self.bucket_pct_of_daily

        return self

    def classify_trades_bulk(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify trades as buy or sell using bulk volume classification

        Uses standard normal CDF to classify:
        V_B = V × Φ((P - P_{-1})/σ_ΔP)
        V_S = V - V_B

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            Tuple of (buy_volumes, sell_volumes)
        """
        # Compute price changes
        price_changes = np.diff(prices, prepend=prices[0])

        # Standardize price changes
        if self.price_volatility_ > 0:
            z_scores = price_changes / self.price_volatility_
        else:
            z_scores = np.zeros_like(price_changes)

        # Apply CDF to get buy probability
        buy_prob = stats.norm.cdf(z_scores)

        # Classify volumes
        buy_volumes = volumes * buy_prob
        sell_volumes = volumes * (1 - buy_prob)

        return buy_volumes, sell_volumes

    def create_volume_buckets(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create equal-volume buckets from trade data

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            Tuple of (bucket_buy_volumes, bucket_sell_volumes, bucket_indices)
        """
        # Classify trades
        buy_volumes, sell_volumes = self.classify_trades_bulk(prices, volumes)

        # Create buckets based on cumulative volume
        cumulative_volume = np.cumsum(volumes)
        bucket_boundaries = np.arange(
            self.volume_bucket_size, cumulative_volume[-1], self.volume_bucket_size
        )

        # Assign trades to buckets
        bucket_indices = np.searchsorted(cumulative_volume, bucket_boundaries, side='right')

        # Aggregate buy/sell volumes per bucket
        bucket_buy_volumes = []
        bucket_sell_volumes = []

        prev_idx = 0
        for idx in bucket_indices:
            bucket_buy_volumes.append(buy_volumes[prev_idx:idx].sum())
            bucket_sell_volumes.append(sell_volumes[prev_idx:idx].sum())
            prev_idx = idx

        # Add final bucket
        if prev_idx < len(buy_volumes):
            bucket_buy_volumes.append(buy_volumes[prev_idx:].sum())
            bucket_sell_volumes.append(sell_volumes[prev_idx:].sum())

        return (
            np.array(bucket_buy_volumes),
            np.array(bucket_sell_volumes),
            bucket_indices,
        )

    def compute_vpin(
        self, bucket_buy_volumes: np.ndarray, bucket_sell_volumes: np.ndarray
    ) -> np.ndarray:
        """
        Compute VPIN from bucketed buy/sell volumes

        VPIN = (1/nV) Σ |V_S - V_B|

        Args:
            bucket_buy_volumes: Buy volumes per bucket
            bucket_sell_volumes: Sell volumes per bucket

        Returns:
            VPIN values (rolling over n_buckets)
        """
        # Compute order flow imbalance per bucket
        ofi = np.abs(bucket_sell_volumes - bucket_buy_volumes)

        # Rolling sum over n_buckets
        vpin_values = []
        for i in range(len(ofi)):
            if i < self.n_buckets - 1:
                # Use available buckets
                window = ofi[: i + 1]
            else:
                # Use full n_buckets window
                window = ofi[i - self.n_buckets + 1 : i + 1]

            # VPIN = sum of OFI / (n * bucket_volume)
            vpin = window.sum() / (len(window) * self.volume_bucket_size)

            # Clip to [0, 1]
            vpin = np.clip(vpin, 0, 1)

            vpin_values.append(vpin)

        return np.array(vpin_values)

    def fit_predict(
        self,
        prices: Union[np.ndarray, pd.Series],
        volumes: Union[np.ndarray, pd.Series],
        daily_volume: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fit and compute VPIN in one step

        Args:
            prices: Trade prices
            volumes: Trade volumes
            daily_volume: Optional total daily volume

        Returns:
            VPIN values per volume bucket
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        if isinstance(volumes, pd.Series):
            volumes = volumes.values

        # Fit parameters
        self.fit(prices, volumes, daily_volume)

        # Create buckets
        bucket_buy_volumes, bucket_sell_volumes, _ = self.create_volume_buckets(
            prices, volumes
        )

        # Compute VPIN
        vpin_values = self.compute_vpin(bucket_buy_volumes, bucket_sell_volumes)

        return vpin_values

    def predict_extreme_event(self, vpin: float, threshold: float = 0.90) -> bool:
        """
        Predict extreme volatility event based on VPIN percentile

        Research shows CDF(VPIN) > 0.90 predicted Flash Crash hours in advance.

        Args:
            vpin: Current VPIN value
            threshold: Percentile threshold (default: 0.90)

        Returns:
            True if extreme event predicted
        """
        return vpin > threshold


class VPINClassifier:
    """
    Binary classifier using VPIN for informed trading detection

    Wraps VPIN in a scikit-learn style classifier interface.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        n_buckets: int = 50,
        volume_bucket_size: Optional[float] = None,
        bucket_pct_of_daily: float = 0.02,
    ):
        """
        Initialize VPIN classifier

        Args:
            threshold: VPIN threshold for anomaly detection (default: 0.75)
            n_buckets: Number of volume buckets
            volume_bucket_size: Fixed volume per bucket
            bucket_pct_of_daily: Bucket size as fraction of daily volume
        """
        self.threshold = threshold
        self.vpin = VPIN(n_buckets, volume_bucket_size, bucket_pct_of_daily)

    def fit(
        self,
        prices: Union[np.ndarray, pd.Series],
        volumes: Union[np.ndarray, pd.Series],
        daily_volume: Optional[float] = None,
    ):
        """
        Fit VPIN parameters

        Args:
            prices: Historical prices
            volumes: Historical volumes
            daily_volume: Total daily volume

        Returns:
            self
        """
        self.vpin.fit(prices, volumes, daily_volume)
        return self

    def predict(
        self, prices: Union[np.ndarray, pd.Series], volumes: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Predict informed trading (0 = normal, 1 = informed)

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            Binary predictions per volume bucket
        """
        vpin_values = self.vpin.fit_predict(prices, volumes)
        predictions = (vpin_values > self.threshold).astype(int)
        return predictions

    def score(
        self, prices: Union[np.ndarray, pd.Series], volumes: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Compute VPIN scores (anomaly scores)

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            VPIN scores per volume bucket
        """
        return self.vpin.fit_predict(prices, volumes)

    def fit_predict(
        self, prices: Union[np.ndarray, pd.Series], volumes: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Fit and predict in one step

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            Binary predictions per volume bucket
        """
        self.fit(prices, volumes)
        return self.predict(prices, volumes)
