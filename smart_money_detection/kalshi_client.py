"""Resilient Kalshi API client with retry/backoff and typed helpers."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class KalshiClientError(RuntimeError):
    """Base error raised when Kalshi API communication fails."""


class KalshiAPIError(KalshiClientError):
    """Raised when the Kalshi API returns an error response."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class KalshiRateLimitError(KalshiAPIError):
    """Raised when the API signals that the rate limit has been exceeded."""


class KalshiClient:
    """Client for interacting with the Kalshi API or local demo data."""

    DEFAULT_TIMEOUT = 10.0
    RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        demo_mode: bool = False,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        """Create a new :class:`KalshiClient` instance."""

        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        resolved_api_base = api_base if api_base is not None else os.getenv(
            'KALSHI_API_BASE', 'https://api.elections.kalshi.com'
        )
        self.api_base = resolved_api_base.rstrip('/')
        self.demo_mode = demo_mode or os.getenv('KALSHI_DEMO_MODE', 'false').lower() == 'true'

        self.logger = logging.getLogger(__name__)

        self._timeout = timeout
        self.session = requests.Session()

        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=self.RETRYABLE_STATUS_CODES,
            allowed_methods=frozenset({'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        if self.api_key and not self.demo_mode:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            })

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute an HTTP request and return the parsed JSON payload."""

        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        timeout = kwargs.pop('timeout', self._timeout)

        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:  # pragma: no cover - thin wrapper
            status_code = exc.response.status_code if exc.response else None
            message = f"Kalshi API request failed with status {status_code}: {exc}"
            error_cls = (
                KalshiRateLimitError if status_code == 429 else KalshiAPIError
            )
            if self.demo_mode:
                self.logger.warning("Falling back to mock data for endpoint %s", endpoint)
                return self._get_mock_data(endpoint)
            raise error_cls(message, status_code=status_code) from exc
        except requests.exceptions.RequestException as exc:  # pragma: no cover - defensive
            message = f"Kalshi API request failed: {exc}"
            if self.demo_mode:
                self.logger.warning("Falling back to mock data for endpoint %s", endpoint)
                return self._get_mock_data(endpoint)
            raise KalshiClientError(message) from exc

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - unexpected response
            raise KalshiAPIError("Kalshi API returned non-JSON response") from exc

    def _get_mock_data(self, endpoint: str) -> Dict[str, Any]:
        """Return mock API payloads for demo mode usage."""
        if 'markets' in endpoint and endpoint.endswith('markets'):
            return self._mock_markets_list()
        elif 'market' in endpoint and 'trades' in endpoint:
            return self._mock_trades()
        elif 'market' in endpoint:
            return self._mock_market_details()
        return {}

    def _mock_markets_list(self) -> Dict[str, Any]:
        """Return a mock market list response."""
        return {
            'markets': [
                {
                    'ticker': 'PRES-2024-WINNER',
                    'title': 'Will the Republican win the 2024 presidential election?',
                    'yes_price': 52,
                    'volume': 15000000,
                    'open_interest': 8000000,
                    'close_time': (datetime.now() + timedelta(days=180)).isoformat(),
                    'status': 'active',
                },
                {
                    'ticker': 'FED-2024-RATE',
                    'title': 'Will the Fed cut rates in 2024?',
                    'yes_price': 75,
                    'volume': 5000000,
                    'open_interest': 2500000,
                    'close_time': (datetime.now() + timedelta(days=90)).isoformat(),
                    'status': 'active',
                },
                {
                    'ticker': 'TECH-EARNINGS-Q4',
                    'title': 'Will tech earnings beat expectations?',
                    'yes_price': 48,
                    'volume': 500000,
                    'open_interest': 250000,
                    'close_time': (datetime.now() + timedelta(days=30)).isoformat(),
                    'status': 'active',
                },
            ]
        }

    def _mock_market_details(self) -> Dict[str, Any]:
        """Return a mock market details response."""
        return {
            'market': {
                'ticker': 'PRES-2024-WINNER',
                'title': 'Will the Republican win the 2024 presidential election?',
                'yes_price': 52,
                'volume': 15000000,
                'open_interest': 8000000,
                'close_time': (datetime.now() + timedelta(days=180)).isoformat(),
                'status': 'active',
                'last_updated': datetime.now().isoformat(),
            }
        }

    def _mock_trades(self) -> Dict[str, Any]:
        """Return a mock trades response with a realistic distribution."""
        import numpy as np

        np.random.seed(42)
        n_trades = 500

        # Generate realistic trade data
        timestamps = pd.date_range(
            end=datetime.now(), periods=n_trades, freq='5min'
        ).tolist()

        # Volume: mostly small, some large (smart money)
        volumes = np.random.lognormal(mean=5, sigma=2, size=n_trades)

        # Add smart money trades (5% of trades are large)
        smart_money_indices = np.random.choice(n_trades, size=int(n_trades * 0.05), replace=False)
        volumes[smart_money_indices] *= 8  # Much larger volumes

        # Prices (cents): random walk around current price
        base_price = 52
        price_changes = np.cumsum(np.random.randn(n_trades) * 0.5)
        prices = base_price + price_changes
        prices = np.clip(prices, 1, 99)  # Keep in valid range

        trades = []
        for i in range(n_trades):
            trades.append({
                'trade_id': f'trade_{i}',
                'timestamp': timestamps[i].isoformat(),
                'volume': float(volumes[i]),
                'price': float(prices[i]),
                'side': 'buy' if np.random.rand() > 0.5 else 'sell',
            })

        return {'trades': trades}

    def get_markets(
        self,
        status: str = 'active',
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return a list of markets filtered by *status* and *limit*."""

        if self.demo_mode:
            data = self._mock_markets_list()
            return data.get('markets', [])

        response = self._request(
            'GET',
            '/trade-api/v2/markets',
            params={'status': status, 'limit': limit},
        )
        return response.get('markets', [])

    def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return market details for *ticker* if available."""

        if self.demo_mode:
            data = self._mock_market_details()
            market = data.get('market', {})
            market['ticker'] = ticker
            return market

        response = self._request('GET', f'/trade-api/v2/markets/{ticker}')
        return response.get('market')

    def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[datetime] = None,
        max_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return trade history for *ticker* as a Pandas DataFrame."""

        if self.demo_mode:
            data = self._mock_trades()
            trades = data.get('trades', [])
        else:
            params: Dict[str, Any] = {'limit': limit}
            if min_ts:
                params['min_ts'] = int(min_ts.timestamp())
            if max_ts:
                params['max_ts'] = int(max_ts.timestamp())

            response = self._request(
                'GET', f'/trade-api/v2/markets/{ticker}/trades', params=params
            )
            trades = response.get('trades', [])

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)

        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Encourage efficient downstream processing via categorical types.
        if 'side' in df.columns:
            df['side'] = df['side'].astype('category')

        return df

    def get_market_summary(self, ticker: str) -> Dict[str, Any]:
        """Return a concise market summary enriched with trade statistics."""

        market = self.get_market(ticker)
        if not market:
            return {}

        trades = self.get_trades(ticker, limit=1000)

        summary: Dict[str, Any] = {
            'ticker': ticker,
            'title': market.get('title', ''),
            'current_price': market.get('yes_price', 0),
            'volume': market.get('volume', 0),
            'open_interest': market.get('open_interest', 0),
            'close_time': market.get('close_time', ''),
            'status': market.get('status', ''),
        }

        if not trades.empty:
            last_day = datetime.now() - timedelta(days=1)
            summary.update(
                {
                    'n_trades': int(trades.shape[0]),
                    'avg_trade_size': float(trades['volume'].mean()),
                    'median_trade_size': float(trades['volume'].median()),
                    'max_trade_size': float(trades['volume'].max()),
                    'total_volume_24h': float(
                        trades.loc[trades['timestamp'] > last_day, 'volume'].sum()
                    ),
                }
            )

        return summary

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()
