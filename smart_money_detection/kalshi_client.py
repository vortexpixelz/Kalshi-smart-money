"""Kalshi API clients providing sync and async access layers."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx
import pandas as pd
import requests

from .utils import trades_from_records


DEFAULT_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class KalshiClientError(RuntimeError):
    """Base error raised by Kalshi client operations."""


class KalshiRequestError(KalshiClientError):
    """Raised when a request fails before receiving a response."""

    def __init__(self, message: str, *, method: str, endpoint: str) -> None:
        super().__init__(message)
        self.method = method
        self.endpoint = endpoint


class KalshiAPIError(KalshiClientError):
    """Raised when the Kalshi API returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        method: str,
        endpoint: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_text = response_text
        self.payload = payload


class _KalshiBase:
    """Shared helpers for Kalshi clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        retry_statuses: Optional[Sequence[int]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        resolved_api_base = api_base if api_base is not None else os.getenv(
            'KALSHI_API_BASE', 'https://api.elections.kalshi.com'
        )
        self.api_base = resolved_api_base.rstrip('/')
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_statuses = set(retry_statuses or DEFAULT_RETRY_STATUSES)

    def _build_url(self, endpoint: str) -> str:
        return f"{self.api_base}/{endpoint.lstrip('/')}"

    @staticmethod
    def _format_trades(trades: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        return trades_from_records(trades)

    def _get_backoff_seconds(self, attempt: int) -> float:
        return self.backoff_factor * (2 ** attempt)

    def _log_retry(self, attempt: int, method: str, endpoint: str, status: Optional[int]) -> None:
        status_info = f" status={status}" if status is not None else ""
        self.logger.warning(
            "Retrying %s %s (attempt %d/%d)%s",
            method,
            endpoint,
            attempt + 1,
            self.max_retries,
            status_info,
        )


class KalshiClient(_KalshiBase):
    """Synchronous Kalshi REST client using ``requests``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        retry_statuses: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            retry_statuses=retry_statuses,
        )
        self.timeout = timeout or 10.0
        self.session = session or requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            })

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = self._build_url(endpoint)
        kwargs.setdefault('timeout', self.timeout)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                response_text = exc.response.text if exc.response is not None else None
                if status_code in self.retry_statuses and attempt < self.max_retries:
                    self._log_retry(attempt, method, endpoint, status_code)
                    time.sleep(self._get_backoff_seconds(attempt))
                    continue
                self.logger.error("API request failed: %s", exc)
                raise KalshiAPIError(
                    "Kalshi API error",
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    response_text=response_text,
                ) from exc
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    self._log_retry(attempt, method, endpoint, None)
                    time.sleep(self._get_backoff_seconds(attempt))
                    continue
                self.logger.error("API request failed: %s", exc)
                raise KalshiRequestError(
                    "Kalshi request failed",
                    method=method,
                    endpoint=endpoint,
                ) from exc
            except ValueError as exc:
                self.logger.error("Failed to decode JSON response: %s", exc)
                raise KalshiAPIError(
                    "Kalshi API returned invalid JSON",
                    method=method,
                    endpoint=endpoint,
                    response_text=getattr(response, "text", None),
                ) from exc

        raise KalshiRequestError(
            "Kalshi request failed after retries",
            method=method,
            endpoint=endpoint,
        )

    def get_markets(self, status: str = 'active', limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch a list of markets filtered by status."""
        response = self._request(
            'GET',
            '/trade-api/v2/markets',
            params={'status': status, 'limit': limit},
        )
        return response.get('markets', [])

    def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a specific market."""
        response = self._request('GET', f'/trade-api/v2/markets/{ticker}')
        return response.get('market')

    def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[datetime] = None,
        max_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch recent trades for the provided ticker."""
        params = {'limit': limit}
        if min_ts:
            params['min_ts'] = int(min_ts.timestamp())
        if max_ts:
            params['max_ts'] = int(max_ts.timestamp())
        response = self._request(
            'GET',
            f'/trade-api/v2/markets/{ticker}/trades',
            params=params,
        )
        trades = response.get('trades', [])
        return self._format_trades(trades)

    def get_market_summary(self, ticker: str) -> Dict[str, Any]:
        """Return market metadata along with recent volume statistics."""
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
                'total_volume_24h': trades.loc[
                    trades['timestamp'] > (datetime.now() - timedelta(days=1)), 'volume'
                ].sum(),
            })

        return summary

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> 'KalshiClient':
        """Enter the context manager for the client."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager and close resources."""
        self.close()


class AsyncKalshiClient(_KalshiBase):
    """Asynchronous Kalshi REST client built on ``httpx``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        retry_statuses: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            retry_statuses=retry_statuses,
        )
        self.timeout = timeout or 10.0
        self._owns_client = client is None

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        if client is None:
            self.client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=headers,
                timeout=self.timeout,
            )
        else:
            self.client = client
            for key, value in headers.items():
                self.client.headers.setdefault(key, value)

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        kwargs.setdefault('timeout', self.timeout)

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                response_text = exc.response.text if exc.response is not None else None
                if status_code in self.retry_statuses and attempt < self.max_retries:
                    self._log_retry(attempt, method, endpoint, status_code)
                    await asyncio.sleep(self._get_backoff_seconds(attempt))
                    continue
                self.logger.error("Async API request failed: %s", exc)
                raise KalshiAPIError(
                    "Kalshi API error",
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    response_text=response_text,
                ) from exc
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    self._log_retry(attempt, method, endpoint, None)
                    await asyncio.sleep(self._get_backoff_seconds(attempt))
                    continue
                self.logger.error("Async API request failed: %s", exc)
                raise KalshiRequestError(
                    "Kalshi request failed",
                    method=method,
                    endpoint=endpoint,
                ) from exc
            except ValueError as exc:
                self.logger.error("Failed to decode JSON response: %s", exc)
                raise KalshiAPIError(
                    "Kalshi API returned invalid JSON",
                    method=method,
                    endpoint=endpoint,
                    response_text=getattr(response, "text", None),
                ) from exc

        raise KalshiRequestError(
            "Kalshi request failed after retries",
            method=method,
            endpoint=endpoint,
        )

    async def get_markets(self, status: str = 'active', limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch a list of markets filtered by status."""
        response = await self._request(
            'GET',
            '/trade-api/v2/markets',
            params={'status': status, 'limit': limit},
        )
        return response.get('markets', [])

    async def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a specific market."""
        response = await self._request('GET', f'/trade-api/v2/markets/{ticker}')
        return response.get('market')

    async def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[datetime] = None,
        max_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch recent trades for the provided ticker."""
        params = {'limit': limit}
        if min_ts:
            params['min_ts'] = int(min_ts.timestamp())
        if max_ts:
            params['max_ts'] = int(max_ts.timestamp())
        response = await self._request(
            'GET',
            f'/trade-api/v2/markets/{ticker}/trades',
            params=params,
        )
        trades = response.get('trades', [])
        return self._format_trades(trades)

    async def get_market_summary(self, ticker: str) -> Dict[str, Any]:
        """Return market metadata along with recent volume statistics."""
        market = await self.get_market(ticker)
        if not market:
            return {}

        trades = await self.get_trades(ticker, limit=1000)
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

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> 'AsyncKalshiClient':
        """Enter the async context manager for the client."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the async context manager and close resources."""
        await self.aclose()


__all__ = [
    'KalshiClient',
    'AsyncKalshiClient',
    'KalshiAPIError',
    'KalshiRequestError',
    'KalshiClientError',
]
