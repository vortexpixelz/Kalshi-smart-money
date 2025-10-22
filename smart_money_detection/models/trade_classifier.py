"""
Trade classification algorithms for buy/sell identification

Used in VPIN and other order flow analysis methods.
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy import stats


class BulkVolumeClassifier:
    """
    Bulk volume classification (BVC) for trade direction

    Uses price changes and volume to classify buy vs sell pressure.
    Based on standard normal CDF:

    V_B = V × Φ((P - P_{-1})/σ_ΔP)
    V_S = V - V_B

    where Φ is the standard normal CDF.
    """

    def __init__(self, volatility_window: int = 100):
        """
        Initialize bulk volume classifier

        Args:
            volatility_window: Rolling window for price volatility estimation
        """
        self.volatility_window = volatility_window
        self.price_volatility_ = None

    def fit(self, prices: Union[np.ndarray, pd.Series]):
        """
        Estimate price volatility from historical data

        Args:
            prices: Historical prices

        Returns:
            self
        """
        if isinstance(prices, pd.Series):
            prices = prices.values

        price_changes = np.diff(prices)
        self.price_volatility_ = np.std(price_changes)

        # Prevent division by zero
        if self.price_volatility_ == 0:
            self.price_volatility_ = 1.0

        return self

    def classify(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify trades as buy or sell volume

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            Tuple of (buy_volumes, sell_volumes)
        """
        if self.price_volatility_ is None:
            self.fit(prices)

        # Compute price changes
        price_changes = np.diff(prices, prepend=prices[0])

        # Standardize
        z_scores = price_changes / self.price_volatility_

        # Apply CDF to get buy probability
        buy_prob = stats.norm.cdf(z_scores)

        # Classify volumes
        buy_volumes = volumes * buy_prob
        sell_volumes = volumes * (1 - buy_prob)

        return buy_volumes, sell_volumes

    def classify_direction(self, prices: np.ndarray) -> np.ndarray:
        """
        Classify trade direction only (-1 = sell, 0 = neutral, 1 = buy)

        Args:
            prices: Trade prices

        Returns:
            Direction array
        """
        if self.price_volatility_ is None:
            self.fit(prices)

        # Compute price changes
        price_changes = np.diff(prices, prepend=prices[0])

        # Classify based on sign
        directions = np.sign(price_changes)

        return directions


class TickRuleClassifier:
    """
    Tick rule classification for trade direction

    Simple rule-based classifier:
    - If price > previous price: buy
    - If price < previous price: sell
    - If price = previous price: use last non-zero change
    """

    def __init__(self):
        """Initialize tick rule classifier"""
        pass

    def classify_direction(self, prices: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Classify trade direction using tick rule

        Args:
            prices: Trade prices

        Returns:
            Direction array (-1 = sell, 0 = neutral, 1 = buy)
        """
        if isinstance(prices, pd.Series):
            prices = prices.values

        directions = np.zeros(len(prices))

        # First trade is neutral
        directions[0] = 0

        # Classify based on price changes
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                directions[i] = 1  # Buy
            elif prices[i] < prices[i - 1]:
                directions[i] = -1  # Sell
            else:
                # Use last non-zero direction
                directions[i] = directions[i - 1]

        return directions

    def classify(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify trade volumes as buy or sell

        Args:
            prices: Trade prices
            volumes: Trade volumes

        Returns:
            Tuple of (buy_volumes, sell_volumes)
        """
        directions = self.classify_direction(prices)

        # Allocate volumes based on direction
        buy_volumes = np.where(directions >= 0, volumes, 0)
        sell_volumes = np.where(directions < 0, volumes, 0)

        return buy_volumes, sell_volumes


class QuoteRuleClassifier:
    """
    Quote rule classification using bid-ask midpoint

    Classifies based on trade price relative to midpoint:
    - Price > midpoint: buy
    - Price < midpoint: sell
    - Price = midpoint: neutral or use tick rule
    """

    def __init__(self):
        """Initialize quote rule classifier"""
        pass

    def classify_direction(
        self, prices: np.ndarray, bids: np.ndarray, asks: np.ndarray
    ) -> np.ndarray:
        """
        Classify trade direction using quote rule

        Args:
            prices: Trade prices
            bids: Bid prices
            asks: Ask prices

        Returns:
            Direction array (-1 = sell, 0 = neutral, 1 = buy)
        """
        midpoints = (bids + asks) / 2

        directions = np.zeros(len(prices))
        directions[prices > midpoints] = 1  # Buy
        directions[prices < midpoints] = -1  # Sell
        # prices = midpoint remains 0 (neutral)

        return directions

    def classify(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify trade volumes as buy or sell

        Args:
            prices: Trade prices
            volumes: Trade volumes
            bids: Bid prices
            asks: Ask prices

        Returns:
            Tuple of (buy_volumes, sell_volumes)
        """
        directions = self.classify_direction(prices, bids, asks)

        # Allocate volumes
        buy_volumes = np.where(directions >= 0, volumes, 0)
        sell_volumes = np.where(directions < 0, volumes, 0)

        return buy_volumes, sell_volumes
