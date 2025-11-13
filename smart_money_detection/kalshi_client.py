"""
Kalshi API client for fetching real market and trade data

Official Kalshi API documentation: https://trading-api.readme.io/reference/getting-started
"""
import os
import requests
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import logging


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
    ):
        """
        Initialize Kalshi API client

        Args:
            api_key: Kalshi API key (or set KALSHI_API_KEY env var)
            api_base: API base URL
            demo_mode: If True, use demo/mock data instead of real API
        """
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        resolved_api_base = api_base if api_base is not None else os.getenv(
            'KALSHI_API_BASE', 'https://api.elections.kalshi.com'
        )
        self.api_base = resolved_api_base.rstrip('/')
        self.demo_mode = demo_mode or os.getenv('KALSHI_DEMO_MODE', 'false').lower() == 'true'

        self.logger = logging.getLogger(__name__)

        # Session for connection pooling
        self.session = requests.Session()
        if self.api_key and not self.demo_mode:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            })

    def verify_authentication(self) -> bool:
        """Check that the client can authenticate against the API."""
        if self.demo_mode:
            return True

        try:
            response = self._request(
                'GET',
                '/trade-api/v2/markets',
                params={'status': 'active', 'limit': 1},
            )
        except Exception as exc:  # pragma: no cover - exercised in integration tests
            self.logger.error("Authentication verification failed: %s", exc)
            return False

        return bool(response.get('markets'))

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request"""
        url = f"{self.api_base}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            if self.demo_mode:
                return self._get_mock_data(endpoint)
            raise

    def _get_mock_data(self, endpoint: str) -> Dict[str, Any]:
        """Return mock data for demo mode"""
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
        if self.demo_mode:
            data = self._mock_markets_list()
            return data.get('markets', [])

        try:
            response = self._request(
                'GET',
                '/trade-api/v2/markets',
                params={'status': status, 'limit': limit}
            )
            return response.get('markets', [])
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
        if self.demo_mode:
            data = self._mock_market_details()
            market = data.get('market', {})
            market['ticker'] = ticker  # Use requested ticker
            return market

        try:
            response = self._request('GET', f'/trade-api/v2/markets/{ticker}')
            return response.get('market')
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
                    params=params
                )
                trades = response.get('trades', [])
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
