"""
Kalshi API client for fetching real market and trade data

Official Kalshi API documentation: https://trading-api.readme.io/reference/getting-started
"""
import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests


@dataclass
class RequestMetrics:
    """Metadata about an API request."""

    latency_ms: float
    retries: int


@dataclass
class ApiResponse:
    """Standardized API response model."""

    data: Dict[str, Any]
    status_code: int
    metrics: RequestMetrics


class KalshiApiError(Exception):
    """Raised when an API request fails."""

    def __init__(
        self,
        message: str,
        endpoint: str,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
        retries: int = 0,
    ):
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body
        self.retries = retries

    def __str__(self) -> str:  # pragma: no cover - uses base formatting
        base = super().__str__()
        return (
            f"{base} (endpoint={self.endpoint}, status={self.status_code}, "
            f"retries={self.retries}, response={self.response_body})"
        )


class KalshiClient:
    """
    Client for interacting with Kalshi API

    Supports both production and demo environments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        demo_mode: bool = False,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        session: Optional[requests.Session] = None,
        mock_response_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        """
        Initialize Kalshi API client

        Args:
            api_key: Kalshi API key (or set KALSHI_API_KEY env var)
            api_base: API base URL
            demo_mode: If True, use demo/mock data instead of real API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            session: Optional requests session for dependency injection (testing)
            mock_response_provider: Optional callable returning mock payload for endpoints
        """
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        resolved_api_base = api_base if api_base is not None else os.getenv(
            'KALSHI_API_BASE', 'https://api.elections.kalshi.com'
        )
        self.api_base = resolved_api_base.rstrip('/')
        self.demo_mode = demo_mode or os.getenv('KALSHI_DEMO_MODE', 'false').lower() == 'true'
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.mock_response_provider = mock_response_provider

        self.logger = logging.getLogger(__name__)

        # Session for connection pooling
        self.session = session or requests.Session()
        if self.api_key and not self.demo_mode:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            })

    def _request(self, method: str, endpoint: str, **kwargs) -> ApiResponse:
        """Make API request with retries, metrics, and structured errors."""
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        start_time = time.monotonic()

        if self.demo_mode:
            return self._wrap_mock_response(endpoint, start_time, retries=0)

        for attempt in range(self.max_retries + 1):
            retries_attempted = attempt
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
            except requests.exceptions.Timeout as exc:
                self._log_retry(endpoint, retries_attempted, 'timeout', exc)
                if attempt < self.max_retries:
                    self._backoff_sleep(attempt)
                    continue
                if self.demo_mode:
                    return self._wrap_mock_response(endpoint, start_time, retries_attempted)
                raise KalshiApiError(
                    'Request timed out',
                    endpoint,
                    status_code=None,
                    response_body=str(exc),
                    retries=retries_attempted,
                ) from exc
            except requests.exceptions.RequestException as exc:
                self._log_retry(endpoint, retries_attempted, 'exception', exc)
                if attempt < self.max_retries:
                    self._backoff_sleep(attempt)
                    continue
                if self.demo_mode:
                    return self._wrap_mock_response(endpoint, start_time, retries_attempted)
                raise KalshiApiError(
                    'Request failed',
                    endpoint,
                    status_code=None,
                    response_body=str(exc),
                    retries=retries_attempted,
                ) from exc

            if response.status_code >= 500 and attempt < self.max_retries:
                self._log_retry(endpoint, retries_attempted, 'status', response)
                self._backoff_sleep(attempt)
                continue

            if response.status_code >= 400:
                body = self._extract_body(response)
                self._log_failure(endpoint, response.status_code, start_time, retries_attempted, body)
                raise KalshiApiError(
                    'Kalshi API returned an error',
                    endpoint,
                    status_code=response.status_code,
                    response_body=body,
                    retries=retries_attempted,
                )

            body = self._extract_body(response)
            metrics = RequestMetrics(
                latency_ms=(time.monotonic() - start_time) * 1000,
                retries=retries_attempted,
            )
            self.logger.info(
                'kalshi_api_request',
                extra={
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'latency_ms': metrics.latency_ms,
                    'retries': retries_attempted,
                },
            )
            return ApiResponse(data=body, status_code=response.status_code, metrics=metrics)

        # Should never reach here because loop returns or raises
        raise KalshiApiError('Unknown request failure', endpoint)

    def _extract_body(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except ValueError:
            return {'raw': response.text}

    def _backoff_sleep(self, attempt: int) -> None:
        sleep_for = self.backoff_factor * (2 ** attempt)
        time.sleep(sleep_for)

    def _log_retry(self, endpoint: str, retries: int, reason: str, detail: Any) -> None:
        self.logger.warning(
            'kalshi_api_retry',
            extra={
                'endpoint': endpoint,
                'retries': retries,
                'reason': reason,
                'detail': str(detail),
            },
        )

    def _log_failure(
        self,
        endpoint: str,
        status_code: int,
        start_time: float,
        retries: int,
        body: Any,
    ) -> None:
        self.logger.error(
            'kalshi_api_failure',
            extra={
                'endpoint': endpoint,
                'status_code': status_code,
                'latency_ms': (time.monotonic() - start_time) * 1000,
                'retries': retries,
                'response_body': body,
            },
        )

    def _wrap_mock_response(self, endpoint: str, start_time: float, retries: int) -> ApiResponse:
        data = self._get_mock_data(endpoint)
        metrics = RequestMetrics(latency_ms=(time.monotonic() - start_time) * 1000, retries=retries)
        self.logger.info(
            'kalshi_api_mock',
            extra={'endpoint': endpoint, 'latency_ms': metrics.latency_ms, 'retries': retries},
        )
        return ApiResponse(data=data, status_code=200, metrics=metrics)

    def _get_mock_data(self, endpoint: str) -> Dict[str, Any]:
        """Return mock data for demo mode"""
        if self.mock_response_provider:
            return self.mock_response_provider(endpoint)
        if 'markets' in endpoint and endpoint.endswith('markets'):
            return self._mock_markets_list()
        elif 'market' in endpoint and 'trades' in endpoint:
            return self._mock_trades()
        elif 'market' in endpoint:
            return self._mock_market_details()
        return {}

    def _mock_markets_list(self) -> Dict[str, Any]:
        """Mock market list response"""
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
        """Mock market details response"""
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
        """Mock trades response with realistic distribution"""
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
        """
        Get list of markets

        Args:
            status: Market status ('active', 'closed', 'settled')
            limit: Maximum number of markets to return

        Returns:
            List of market dictionaries
        """
        try:
            response = self._request(
                'GET',
                '/trade-api/v2/markets',
                params={'status': status, 'limit': limit}
            )
            return response.data.get('markets', [])
        except Exception as e:
            self.logger.error(f"Failed to fetch markets: {e}")
            return []

    def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get market details by ticker

        Args:
            ticker: Market ticker symbol (e.g., 'PRES-2024-WINNER')

        Returns:
            Market details dictionary
        """
        try:
            response = self._request('GET', f'/trade-api/v2/markets/{ticker}')
            market = response.data.get('market')
            if market:
                market.setdefault('ticker', ticker)
            return market
        except Exception as e:
            self.logger.error(f"Failed to fetch market {ticker}: {e}")
            return None

    def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[datetime] = None,
        max_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get trade history for a market

        Args:
            ticker: Market ticker symbol
            limit: Maximum number of trades to return
            min_ts: Minimum timestamp for trades
            max_ts: Maximum timestamp for trades

        Returns:
            DataFrame with trade data
        """
        try:
            params = {'limit': limit}
            if min_ts:
                params['min_ts'] = int(min_ts.timestamp())
            if max_ts:
                params['max_ts'] = int(max_ts.timestamp())

            response = self._request(
                'GET',
                f'/trade-api/v2/markets/{ticker}/trades',
                params=params
            )
            trades = response.data.get('trades', [])
        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {ticker}: {e}")
            return pd.DataFrame()

        if not trades:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure numeric types
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def get_market_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive market summary including stats

        Args:
            ticker: Market ticker symbol

        Returns:
            Dictionary with market summary
        """
        market = self.get_market(ticker)
        if not market:
            return {}

        trades = self.get_trades(ticker, limit=1000)

        summary = {
            'ticker': ticker,
            'title': market.get('title', ''),
            'current_price': market.get('yes_price', 0),
            'volume': market.get('volume', 0),
            'open_interest': market.get('open_interest', 0),
            'close_time': market.get('close_time', ''),
            'status': market.get('status', ''),
        }

        if not trades.empty:
            summary.update({
                'n_trades': len(trades),
                'avg_trade_size': trades['volume'].mean(),
                'median_trade_size': trades['volume'].median(),
                'max_trade_size': trades['volume'].max(),
                'total_volume_24h': trades[
                    trades['timestamp'] > (datetime.now() - timedelta(days=1))
                ]['volume'].sum(),
            })

        return summary

    def close(self):
        """Close HTTP session"""
        self.session.close()
