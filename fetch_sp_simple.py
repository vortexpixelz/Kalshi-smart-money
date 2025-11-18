#!/usr/bin/env python3
"""
Simple Kalshi S&P Price data fetcher
Fetches market data directly from Kalshi API
"""
import os
import requests
import json
from datetime import datetime, timedelta, timezone


def fetch_kalshi_sp_data():
    """Fetch S&P price markets from Kalshi"""

    # Get API credentials from environment
    api_key = os.getenv('KALSHI_API_KEY')
    api_base = os.getenv('KALSHI_API_BASE', 'https://api.elections.kalshi.com')
    demo_mode = os.getenv('KALSHI_DEMO_MODE', 'false').lower() == 'true'

    print("="*80)
    print("Kalshi S&P Price Data Fetcher")
    print("="*80)

    if not api_key and not demo_mode:
        print("\n⚠️  No API key found. Please set KALSHI_API_KEY environment variable")
        print("\nExample:")
        print("  export KALSHI_API_KEY='your_api_key_here'")
        print("\nContinuing without authentication (limited access)...")
    else:
        if demo_mode:
            print("\n🔧 Running in DEMO MODE")
        else:
            print(f"\n✓ Connected to Kalshi API")
            print(f"  Base URL: {api_base}")

    # Calculate tomorrow at 4pm EST
    est_offset = timezone(timedelta(hours=-5))
    now_est = datetime.now(est_offset)
    tomorrow_4pm = now_est.replace(hour=16, minute=0, second=0, microsecond=0) + timedelta(days=1)

    print(f"\nTarget time: {tomorrow_4pm.strftime('%Y-%m-%d %I:%M %p EST')}")

    # Set up headers
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    # Fetch markets
    print("\nFetching markets from Kalshi API...")
    try:
        url = f"{api_base}/trade-api/v2/markets"
        params = {'status': 'active', 'limit': 1000}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        markets = data.get('markets', [])
        print(f"✓ Found {len(markets)} active markets")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error fetching markets: {e}")
        print("\nNote: You may need valid API credentials to access Kalshi data.")
        print("Visit https://kalshi.com/ to get an API key")
        return

    # Filter for S&P related markets
    print("\nFiltering for S&P related markets...")
    sp_keywords = ['s&p', 'sp500', 'spx', 's&p 500', 'sp-', 'inx-', 'sp500']
    sp_markets = []

    for market in markets:
        ticker = market.get('ticker', '').lower()
        title = market.get('title', '').lower()
        subtitle = market.get('subtitle', '').lower()

        if any(keyword in ticker or keyword in title or keyword in subtitle for keyword in sp_keywords):
            sp_markets.append(market)

    print(f"✓ Found {len(sp_markets)} S&P related markets\n")

    if not sp_markets:
        print("❌ No S&P related markets found")
        print("\nShowing sample of available markets:")
        for i, market in enumerate(markets[:10]):
            print(f"  {i+1}. {market.get('ticker', 'N/A')}: {market.get('title', 'N/A')[:60]}")
        return

    # Display S&P markets
    print("="*80)
    print("S&P RELATED MARKETS")
    print("="*80)

    for i, market in enumerate(sp_markets, 1):
        print(f"\n{i}. Ticker: {market.get('ticker', 'N/A')}")
        print(f"   Title: {market.get('title', 'N/A')}")

        if 'subtitle' in market:
            print(f"   Subtitle: {market['subtitle']}")

        if 'yes_bid' in market:
            print(f"   Yes Bid: {market['yes_bid']}¢")
        if 'yes_ask' in market:
            print(f"   Yes Ask: {market['yes_ask']}¢")
        if 'yes_price' in market:
            print(f"   Last Price: {market['yes_price']}¢")

        if 'volume' in market:
            print(f"   Volume: ${market['volume']:,}")
        if 'volume_24h' in market:
            print(f"   24h Volume: ${market['volume_24h']:,}")
        if 'open_interest' in market:
            print(f"   Open Interest: {market['open_interest']:,}")

        if 'close_time' in market:
            close_time = market['close_time']
            print(f"   Close Time: {close_time}")

        if 'expiration_time' in market:
            expiration_time = market['expiration_time']
            print(f"   Expiration: {expiration_time}")

        print(f"   Status: {market.get('status', 'N/A')}")

    # Get detailed data for first market
    if sp_markets:
        print("\n" + "="*80)
        print("DETAILED DATA FOR TOP S&P MARKET")
        print("="*80)

        ticker = sp_markets[0]['ticker']
        print(f"\nFetching detailed data for {ticker}...\n")

        try:
            # Get market details
            market_url = f"{api_base}/trade-api/v2/markets/{ticker}"
            response = requests.get(market_url, headers=headers, timeout=30)
            response.raise_for_status()
            market_data = response.json()

            market = market_data.get('market', {})

            print(json.dumps(market, indent=2))

            # Try to get recent trades
            print(f"\n\nFetching recent trades for {ticker}...")
            trades_url = f"{api_base}/trade-api/v2/markets/{ticker}/trades"
            trade_params = {'limit': 100}

            response = requests.get(trades_url, headers=headers, params=trade_params, timeout=30)
            response.raise_for_status()
            trades_data = response.json()

            trades = trades_data.get('trades', [])
            if trades:
                print(f"✓ Found {len(trades)} recent trades\n")
                print("Last 5 Trades:")
                print(f"{'Timestamp':<25} {'Price':<10} {'Volume':<15} {'Side':<10}")
                print("-" * 65)

                for trade in trades[-5:]:
                    ts = trade.get('created_time', 'N/A')
                    price = trade.get('yes_price', trade.get('price', 'N/A'))
                    count = trade.get('count', trade.get('volume', 'N/A'))
                    side = trade.get('side', 'N/A')

                    print(f"{ts:<25} {price}¢{' ':<7} {count:<15} {side:<10}")
            else:
                print("No recent trades found")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching detailed data: {e}")

    print("\n" + "="*80)
    print("Fetch complete!")
    print("="*80)


if __name__ == '__main__':
    fetch_kalshi_sp_data()
