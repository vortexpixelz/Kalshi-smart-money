"""Kalshi API clients providing sync and async access layers."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

import httpx
import pandas as pd
import requests

ResponsePayload = TypeVar('ResponsePayload')
MockResponder = Callable[[str, str, Dict[str, Any]], 'KalshiMockResponse']
AsyncMockResponder = Callable[[str, str, Dict[str, Any]], Awaitable['KalshiMockResponse']]


@dataclass(frozen=True)
class RequestMetrics:
    latency: float
    retries_attempted: int


@dataclass(frozen=True)
class KalshiApiResponse(Generic[ResponsePayload]):
    endpoint: str
    status_code: int
    data: ResponsePayload
    metrics: RequestMetrics


@dataclass(frozen=True)
class KalshiMockResponse:
    status_code: int
    data: Dict[str, Any]
    body: Optional[str] = None


class KalshiApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status_code: Optional[int],
        response_body: Optional[str],
        retries_attempted: int,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body
        self.retries_attempted = retries_attempted


class _KalshiBase:
    """Shared helpers for Kalshi clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        resolved_api_base = api_base if api_base is not None else os.getenv(
            'KALSHI_API_BASE', 'https://api.elections.kalshi.com'
        )
        self.api_base = resolved_api_base.rstrip('/')
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _parse_json(response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except ValueError:
            return {}

    def _build_url(self, endpoint: str) -> str:
        return f"{self.api_base}/{endpoint.lstrip('/')}"

    @staticmethod
    def _format_trades(trades: List[Dict[str, Any]]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df

class KalshiClient(_KalshiBase):
    """Synchronous Kalshi REST client using ``requests``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        retry_statuses: Optional[Set[int]] = None,
        mock_responder: Optional[MockResponder] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base)
        self.timeout = timeout or 10.0
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
        self.mock_responder = mock_responder
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            })

    def _request(self, method: str, endpoint: str, **kwargs) -> KalshiApiResponse[Dict[str, Any]]:
        url = self._build_url(endpoint)
        kwargs.setdefault('timeout', self.timeout)
        if self.mock_responder:
            start = time.monotonic()
            mock_response = self.mock_responder(method, endpoint, kwargs)
            metrics = RequestMetrics(
                latency=time.monotonic() - start,
                retries_attempted=0,
            )
            self.logger.info(
                "Kalshi mock request completed",
                extra={
                    'endpoint': endpoint,
                    'status_code': mock_response.status_code,
                    'latency': metrics.latency,
                    'retries_attempted': metrics.retries_attempted,
                },
            )
            if not (200 <= mock_response.status_code < 300):
                raise KalshiApiError(
                    "Mock request failed",
                    endpoint=endpoint,
                    status_code=mock_response.status_code,
                    response_body=mock_response.body,
                    retries_attempted=0,
                )
            return KalshiApiResponse(
                endpoint=endpoint,
                status_code=mock_response.status_code,
                data=mock_response.data,
                metrics=metrics,
            )

        retries_attempted = 0
        start = time.monotonic()
        last_exception: Optional[BaseException] = None
        while True:
            try:
                response = self.session.request(method, url, **kwargs)
                status_code = response.status_code
                if status_code in self.retry_statuses and retries_attempted < self.max_retries:
                    retries_attempted += 1
                    backoff = self.backoff_factor * (2 ** (retries_attempted - 1))
                    self.logger.warning(
                        "Kalshi request retrying",
                        extra={
                            'endpoint': endpoint,
                            'status_code': status_code,
                            'retries_attempted': retries_attempted,
                            'backoff': backoff,
                        },
                    )
                    time.sleep(backoff)
                    continue
                if not (200 <= status_code < 300):
                    raise KalshiApiError(
                        "API request returned non-success status",
                        endpoint=endpoint,
                        status_code=status_code,
                        response_body=response.text,
                        retries_attempted=retries_attempted,
                    )
                metrics = RequestMetrics(
                    latency=time.monotonic() - start,
                    retries_attempted=retries_attempted,
                )
                self.logger.info(
                    "Kalshi request completed",
                    extra={
                        'endpoint': endpoint,
                        'status_code': status_code,
                        'latency': metrics.latency,
                        'retries_attempted': retries_attempted,
                    },
                )
                return KalshiApiResponse(
                    endpoint=endpoint,
                    status_code=status_code,
                    data=self._parse_json(response),
                    metrics=metrics,
                )
            except requests.RequestException as exc:
                last_exception = exc
                if retries_attempted < self.max_retries:
                    retries_attempted += 1
                    backoff = self.backoff_factor * (2 ** (retries_attempted - 1))
                    self.logger.warning(
                        "Kalshi request retrying after exception",
                        extra={
                            'endpoint': endpoint,
                            'exception': str(exc),
                            'retries_attempted': retries_attempted,
                            'backoff': backoff,
                        },
                    )
                    time.sleep(backoff)
                    continue
                break

        self.logger.error(
            "API request failed",
            extra={
                'endpoint': endpoint,
                'exception': str(last_exception),
                'retries_attempted': retries_attempted,
            },
        )
        raise KalshiApiError(
            "API request failed",
            endpoint=endpoint,
            status_code=None,
            response_body=str(last_exception),
            retries_attempted=retries_attempted,
        )

    def get_markets(
        self,
        status: str = 'active',
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        try:
            response = self._request(
                'GET',
                '/trade-api/v2/markets',
                params={'status': status, 'limit': limit},
            )
            return response.data.get('markets', [])
        except KalshiApiError as exc:
            self.logger.error(
                "Failed to fetch markets",
                extra={
                    'endpoint': exc.endpoint,
                    'status_code': exc.status_code,
                    'response_body': exc.response_body,
                },
            )
            return []

    def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._request('GET', f'/trade-api/v2/markets/{ticker}')
            return response.data.get('market')
        except KalshiApiError as exc:
            self.logger.error(
                "Failed to fetch market",
                extra={
                    'endpoint': exc.endpoint,
                    'status_code': exc.status_code,
                    'response_body': exc.response_body,
                    'ticker': ticker,
                },
            )
            return None

    def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[datetime] = None,
        max_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        try:
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
            trades = response.data.get('trades', [])
        except KalshiApiError as exc:
            self.logger.error(
                "Failed to fetch trades",
                extra={
                    'endpoint': exc.endpoint,
                    'status_code': exc.status_code,
                    'response_body': exc.response_body,
                    'ticker': ticker,
                },
            )
            return pd.DataFrame()

        return self._format_trades(trades)

    def get_market_summary(self, ticker: str) -> Dict[str, Any]:
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

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> 'KalshiClient':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AsyncKalshiClient(_KalshiBase):
    """Asynchronous Kalshi REST client built on ``httpx``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        retry_statuses: Optional[Set[int]] = None,
        mock_responder: Optional[AsyncMockResponder] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base)
        self.timeout = timeout or 10.0
        self._owns_client = client is None
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
        self.mock_responder = mock_responder

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

    async def _request(self, method: str, endpoint: str, **kwargs) -> KalshiApiResponse[Dict[str, Any]]:
        kwargs.setdefault('timeout', self.timeout)
        if self.mock_responder:
            start = time.monotonic()
            mock_response = await self.mock_responder(method, endpoint, kwargs)
            metrics = RequestMetrics(
                latency=time.monotonic() - start,
                retries_attempted=0,
            )
            self.logger.info(
                "Kalshi async mock request completed",
                extra={
                    'endpoint': endpoint,
                    'status_code': mock_response.status_code,
                    'latency': metrics.latency,
                    'retries_attempted': metrics.retries_attempted,
                },
            )
            if not (200 <= mock_response.status_code < 300):
                raise KalshiApiError(
                    "Mock request failed",
                    endpoint=endpoint,
                    status_code=mock_response.status_code,
                    response_body=mock_response.body,
                    retries_attempted=0,
                )
            return KalshiApiResponse(
                endpoint=endpoint,
                status_code=mock_response.status_code,
                data=mock_response.data,
                metrics=metrics,
            )

        retries_attempted = 0
        start = time.monotonic()
        last_exception: Optional[BaseException] = None
        while True:
            try:
                response = await self.client.request(method, endpoint, **kwargs)
                status_code = response.status_code
                if status_code in self.retry_statuses and retries_attempted < self.max_retries:
                    retries_attempted += 1
                    backoff = self.backoff_factor * (2 ** (retries_attempted - 1))
                    self.logger.warning(
                        "Kalshi async request retrying",
                        extra={
                            'endpoint': endpoint,
                            'status_code': status_code,
                            'retries_attempted': retries_attempted,
                            'backoff': backoff,
                        },
                    )
                    await asyncio.sleep(backoff)
                    continue
                if not (200 <= status_code < 300):
                    raise KalshiApiError(
                        "Async API request returned non-success status",
                        endpoint=endpoint,
                        status_code=status_code,
                        response_body=response.text,
                        retries_attempted=retries_attempted,
                    )
                metrics = RequestMetrics(
                    latency=time.monotonic() - start,
                    retries_attempted=retries_attempted,
                )
                self.logger.info(
                    "Kalshi async request completed",
                    extra={
                        'endpoint': endpoint,
                        'status_code': status_code,
                        'latency': metrics.latency,
                        'retries_attempted': retries_attempted,
                    },
                )
                return KalshiApiResponse(
                    endpoint=endpoint,
                    status_code=status_code,
                    data=response.json(),
                    metrics=metrics,
                )
            except httpx.HTTPError as exc:
                last_exception = exc
                if retries_attempted < self.max_retries:
                    retries_attempted += 1
                    backoff = self.backoff_factor * (2 ** (retries_attempted - 1))
                    self.logger.warning(
                        "Kalshi async request retrying after exception",
                        extra={
                            'endpoint': endpoint,
                            'exception': str(exc),
                            'retries_attempted': retries_attempted,
                            'backoff': backoff,
                        },
                    )
                    await asyncio.sleep(backoff)
                    continue
                break

        self.logger.error(
            "Async API request failed",
            extra={
                'endpoint': endpoint,
                'exception': str(last_exception),
                'retries_attempted': retries_attempted,
            },
        )
        raise KalshiApiError(
            "Async API request failed",
            endpoint=endpoint,
            status_code=None,
            response_body=str(last_exception),
            retries_attempted=retries_attempted,
        )

    async def get_markets(
        self,
        status: str = 'active',
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        try:
            response = await self._request(
                'GET',
                '/trade-api/v2/markets',
                params={'status': status, 'limit': limit},
            )
            return response.data.get('markets', [])
        except KalshiApiError as exc:
            self.logger.error(
                "Failed to fetch markets (async)",
                extra={
                    'endpoint': exc.endpoint,
                    'status_code': exc.status_code,
                    'response_body': exc.response_body,
                },
            )
            return []

    async def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            response = await self._request('GET', f'/trade-api/v2/markets/{ticker}')
            return response.data.get('market')
        except KalshiApiError as exc:
            self.logger.error(
                "Failed to fetch market (async)",
                extra={
                    'endpoint': exc.endpoint,
                    'status_code': exc.status_code,
                    'response_body': exc.response_body,
                    'ticker': ticker,
                },
            )
            return None

    async def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[datetime] = None,
        max_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        try:
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
            trades = response.data.get('trades', [])
        except KalshiApiError as exc:
            self.logger.error(
                "Failed to fetch trades (async)",
                extra={
                    'endpoint': exc.endpoint,
                    'status_code': exc.status_code,
                    'response_body': exc.response_body,
                    'ticker': ticker,
                },
            )
            return pd.DataFrame()

        return self._format_trades(trades)

    async def get_market_summary(self, ticker: str) -> Dict[str, Any]:
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
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> 'AsyncKalshiClient':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = [
    'KalshiApiError',
    'KalshiApiResponse',
    'KalshiClient',
    'AsyncKalshiClient',
    'KalshiMockResponse',
    'RequestMetrics',
]
