"""Kalshi API clients providing sync and async access layers."""
from __future__ import annotations

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
        demo_mode: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        resolved_api_base = api_base if api_base is not None else os.getenv(
            'KALSHI_API_BASE', 'https://api.elections.kalshi.com'
        )
        self.api_base = resolved_api_base.rstrip('/')
        env_demo = os.getenv('KALSHI_DEMO_MODE', 'false').lower() == 'true'
        self.demo_mode = demo_mode or env_demo
        self.logger = logging.getLogger(__name__)

    def _build_url(self, endpoint: str) -> str:
        return f"{self.api_base}/{endpoint.lstrip('/')}"

    def _get_mock_data(self, endpoint: str) -> Dict[str, Any]:
        if 'markets' in endpoint and endpoint.endswith('markets'):
            return self._mock_markets_list()
        if 'market' in endpoint and 'trades' in endpoint:
            return self._mock_trades()
        if 'market' in endpoint:
            return self._mock_market_details()
        return {}

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

    def _mock_markets_list(self) -> Dict[str, Any]:
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
        import numpy as np

        np.random.seed(42)
        n_trades = 500

        timestamps = pd.date_range(
            end=datetime.now(), periods=n_trades, freq='5min'
        ).tolist()

        volumes = np.random.lognormal(mean=5, sigma=2, size=n_trades)
        smart_money_indices = np.random.choice(n_trades, size=int(n_trades * 0.05), replace=False)
        volumes[smart_money_indices] *= 8

        base_price = 52
        price_changes = np.cumsum(np.random.randn(n_trades) * 0.5)
        prices = base_price + price_changes
        prices = np.clip(prices, 1, 99)

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


class KalshiClient(_KalshiBase):
    """Synchronous Kalshi REST client using ``requests``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        demo_mode: bool = False,
        *,
        session: Optional[requests.Session] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base, demo_mode=demo_mode)
        self.timeout = timeout or 10.0
        self.session = session or requests.Session()
        if self.api_key and not self.demo_mode:
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
            if self.demo_mode:
                return self._get_mock_data(endpoint)
            raise

    def get_markets(
        self,
        status: str = 'active',
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if self.demo_mode:
            data = self._mock_markets_list()
            return data.get('markets', [])

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
        if self.demo_mode:
            data = self._mock_market_details()
            market = data.get('market', {})
            market['ticker'] = ticker
            return market

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
        if self.demo_mode:
            data = self._mock_trades()
            trades = data.get('trades', [])
        else:
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
        demo_mode: bool = False,
        *,
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base, demo_mode=demo_mode)
        self.timeout = timeout or 10.0
        self._owns_client = client is None

        headers = {'Content-Type': 'application/json'}
        if self.api_key and not self.demo_mode:
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
            if self.demo_mode:
                return self._get_mock_data(endpoint)
            raise

    async def get_markets(
        self,
        status: str = 'active',
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if self.demo_mode:
            data = self._mock_markets_list()
            return data.get('markets', [])

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
        if self.demo_mode:
            data = self._mock_market_details()
            market = data.get('market', {})
            market['ticker'] = ticker
            return market

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
        if self.demo_mode:
            data = self._mock_trades()
            trades = data.get('trades', [])
        else:
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

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> 'AsyncKalshiClient':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ['KalshiClient', 'AsyncKalshiClient']
