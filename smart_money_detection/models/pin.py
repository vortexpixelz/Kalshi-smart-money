"""
Probability of Informed Trading (PIN) model

From Easley, Kiefer, O'Hara and Paperman (Journal of Finance 1996).
Estimates the fraction of orders from informed traders.

PIN = αμ / (αμ + 2ε)
where:
    α = probability of information event
    μ = informed trader arrival rate
    ε = uninformed trader arrival rate
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import factorial


class PIN:
    """
    Probability of Informed Trading (PIN) model

    Estimates the fraction of trades from informed traders using
    maximum likelihood estimation on Poisson arrival processes.

    For prediction markets: large wagers = potential informed trades.
    """

    def __init__(self, max_iter: int = 100):
        """
        Initialize PIN model

        Args:
            max_iter: Maximum iterations for MLE optimization
        """
        self.max_iter = max_iter

        # Fitted parameters
        self.alpha_ = None  # Probability of information event
        self.mu_ = None  # Informed trader arrival rate
        self.epsilon_ = None  # Uninformed trader arrival rate
        self.delta_ = None  # Probability of good news (vs bad news)
        self.pin_ = None  # Computed PIN value

    def _log_likelihood(
        self,
        params: np.ndarray,
        buys: np.ndarray,
        sells: np.ndarray,
    ) -> float:
        """
        Compute negative log-likelihood for MLE

        Args:
            params: [alpha, mu, epsilon, delta]
            buys: Buy order counts per period
            sells: Sell order counts per period

        Returns:
            Negative log-likelihood
        """
        alpha, mu, epsilon, delta = params

        # Ensure parameters are valid
        if (
            alpha < 0
            or alpha > 1
            or mu < 0
            or epsilon < 0
            or delta < 0
            or delta > 1
        ):
            return 1e10

        n_periods = len(buys)
        log_likelihood = 0

        for b, s in zip(buys, sells):
            # Probability of observing (b, s) under different states
            # No information event
            p_no_info = (1 - alpha) * self._poisson_prob(b, epsilon) * self._poisson_prob(
                s, epsilon
            )

            # Good news (informed buyers)
            p_good = (
                alpha
                * delta
                * self._poisson_prob(b, epsilon + mu)
                * self._poisson_prob(s, epsilon)
            )

            # Bad news (informed sellers)
            p_bad = (
                alpha
                * (1 - delta)
                * self._poisson_prob(b, epsilon)
                * self._poisson_prob(s, epsilon + mu)
            )

            # Total probability
            p_total = p_no_info + p_good + p_bad

            if p_total > 0:
                log_likelihood += np.log(p_total)
            else:
                log_likelihood -= 1000

        return -log_likelihood

    def _poisson_prob(self, k: int, lam: float) -> float:
        """Compute Poisson probability"""
        if lam <= 0:
            return 0.0
        return np.exp(-lam) * (lam ** k) / factorial(k)

    def fit(
        self,
        buys: Union[np.ndarray, pd.Series],
        sells: Union[np.ndarray, pd.Series],
        initial_params: Optional[np.ndarray] = None,
    ):
        """
        Fit PIN model using maximum likelihood estimation

        Args:
            buys: Buy order counts per period (e.g., per day)
            sells: Sell order counts per period
            initial_params: Optional initial parameter values [alpha, mu, epsilon, delta]

        Returns:
            self
        """
        if isinstance(buys, pd.Series):
            buys = buys.values
        if isinstance(sells, pd.Series):
            sells = sells.values

        # Initial parameter guess
        if initial_params is None:
            avg_trades = (buys.mean() + sells.mean()) / 2
            initial_params = np.array([0.5, avg_trades * 0.3, avg_trades * 0.7, 0.5])

        # Optimize
        bounds = [(0, 1), (0, None), (0, None), (0, 1)]
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(buys, sells),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iter},
        )

        # Extract parameters
        self.alpha_, self.mu_, self.epsilon_, self.delta_ = result.x

        # Compute PIN
        self.pin_ = (self.alpha_ * self.mu_) / (self.alpha_ * self.mu_ + 2 * self.epsilon_)

        return self

    def predict_informed_probability(self) -> float:
        """
        Get the estimated probability of informed trading

        Returns:
            PIN value in [0, 1]
        """
        if self.pin_ is None:
            raise RuntimeError("Model not fitted yet")
        return self.pin_

    def get_parameters(self) -> dict:
        """Get fitted parameters"""
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted yet")

        return {
            'alpha': self.alpha_,
            'mu': self.mu_,
            'epsilon': self.epsilon_,
            'delta': self.delta_,
            'pin': self.pin_,
        }


class SimplifiedPIN:
    """
    Simplified PIN estimation for prediction markets

    Instead of full MLE optimization, uses simple heuristics:
    - Large trades relative to market size indicate informed trading
    - Directional accuracy of large trades (if resolution known)
    - Timing of large trades (closer to resolution = more informed?)
    """

    def __init__(
        self,
        large_trade_threshold: float = 0.95,  # 95th percentile
        volume_weighted: bool = True,
    ):
        """
        Initialize simplified PIN

        Args:
            large_trade_threshold: Percentile threshold for "large" trades
            volume_weighted: If True, weight by trade volume
        """
        self.large_trade_threshold = large_trade_threshold
        self.volume_weighted = volume_weighted

        self.threshold_value_ = None

    def fit(self, volumes: Union[np.ndarray, pd.Series]):
        """
        Fit threshold from historical volume distribution

        Args:
            volumes: Historical trade volumes

        Returns:
            self
        """
        if isinstance(volumes, pd.Series):
            volumes = volumes.values

        self.threshold_value_ = np.percentile(volumes, self.large_trade_threshold * 100)

        return self

    def predict(self, volumes: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict which trades are informed (0 = uninformed, 1 = informed)

        Args:
            volumes: Trade volumes

        Returns:
            Binary predictions
        """
        if self.threshold_value_ is None:
            raise RuntimeError("Model not fitted yet")

        if isinstance(volumes, pd.Series):
            volumes = volumes.values

        predictions = (volumes >= self.threshold_value_).astype(int)
        return predictions

    def score(self, volumes: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Compute informed trading scores

        Args:
            volumes: Trade volumes

        Returns:
            Scores (normalized volume ratios)
        """
        if self.threshold_value_ is None:
            raise RuntimeError("Model not fitted yet")

        if isinstance(volumes, pd.Series):
            volumes = volumes.values

        # Normalize by threshold
        scores = volumes / self.threshold_value_

        return scores
