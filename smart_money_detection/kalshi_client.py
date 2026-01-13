"""Kalshi API clients providing sync and async access layers."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import requests


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
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base)
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
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            self.logger.error(f"API request failed: {exc}")
            raise

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
            return response.get('markets', [])
        except Exception as exc:
            self.logger.error(f"Failed to fetch markets: {exc}")
            return []

    def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._request('GET', f'/trade-api/v2/markets/{ticker}')
            return response.get('market')
        except Exception as exc:
            self.logger.error(f"Failed to fetch market {ticker}: {exc}")
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
            trades = response.get('trades', [])
        except Exception as exc:
            self.logger.error(f"Failed to fetch trades for {ticker}: {exc}")
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
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base)
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
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            self.logger.error(f"Async API request failed: {exc}")
            raise

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
            return response.get('markets', [])
        except Exception as exc:
            self.logger.error(f"Failed to fetch markets (async): {exc}")
            return []

    async def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            response = await self._request('GET', f'/trade-api/v2/markets/{ticker}')
            return response.get('market')
        except Exception as exc:
            self.logger.error(f"Failed to fetch market {ticker} (async): {exc}")
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
            trades = response.get('trades', [])
        except Exception as exc:
            self.logger.error(f"Failed to fetch trades for {ticker} (async): {exc}")
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


class SyncKalshiClientAdapter:
    """Sync adapter for the async Kalshi client."""

    def __init__(self, async_client: AsyncKalshiClient) -> None:
        self._client = async_client

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "Sync adapter cannot be used within a running event loop. "
            "Use AsyncKalshiClient directly in async contexts."
        )

    def get_markets(self, *args, **kwargs):
        return self._run(self._client.get_markets(*args, **kwargs))

    def get_market(self, *args, **kwargs):
        return self._run(self._client.get_market(*args, **kwargs))

    def get_trades(self, *args, **kwargs):
        return self._run(self._client.get_trades(*args, **kwargs))

    def get_market_summary(self, *args, **kwargs):
        return self._run(self._client.get_market_summary(*args, **kwargs))

    def close(self) -> None:
        self._run(self._client.aclose())

    def __enter__(self) -> "SyncKalshiClientAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ['KalshiClient', 'AsyncKalshiClient', 'SyncKalshiClientAdapter']
