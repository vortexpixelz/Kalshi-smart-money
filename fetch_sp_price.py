#!/usr/bin/env python3
"""
Fetch Kalshi S&P Price data for tomorrow at 4pm EST

This script connects to the Kalshi API and fetches market data
for S&P 500 price prediction markets.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from smart_money_detection.kalshi_client import KalshiClient
import json


def format_market_data(market):
    """Format market data for display"""
    print("\n" + "="*80)
    print(f"Market: {market.get('title', 'N/A')}")
    print("="*80)
    print(f"Ticker: {market.get('ticker', 'N/A')}")
    print(f"Status: {market.get('status', 'N/A')}")

    # Price information
    if 'yes_price' in market:
        yes_price = market['yes_price']
        print(f"\nCurrent Yes Price: {yes_price}¢ ({yes_price}%)")
    if 'no_price' in market:
        no_price = market['no_price']
        print(f"Current No Price: {no_price}¢ ({no_price}%)")

    # Volume and liquidity
    if 'volume' in market:
        print(f"\nTotal Volume: ${market['volume']:,.0f}")
    if 'volume_24h' in market:
        print(f"24h Volume: ${market['volume_24h']:,.0f}")
    if 'open_interest' in market:
        print(f"Open Interest: {market['open_interest']:,}")

    # Timing information
    if 'close_time' in market:
        print(f"\nClose Time: {market['close_time']}")
    if 'expiration_time' in market:
        print(f"Expiration Time: {market['expiration_time']}")

    # Strike/threshold information
    if 'strike_price' in market:
        print(f"\nStrike Price: {market['strike_price']}")
    if 'floor_strike' in market:
        print(f"Floor Strike: {market['floor_strike']}")
    if 'cap_strike' in market:
        print(f"Cap Strike: {market['cap_strike']}")

    # Additional details
    if 'liquidity' in market:
        print(f"\nLiquidity: {market['liquidity']}")
    if 'yes_bid' in market and 'yes_ask' in market:
        spread = market['yes_ask'] - market['yes_bid']
        print(f"Bid-Ask Spread: {spread}¢ (Bid: {market['yes_bid']}¢, Ask: {market['yes_ask']}¢)")

    print("="*80)


def main():
    """Main function to fetch S&P price data"""
    print("Kalshi S&P Price Data Fetcher")
    print("="*80)

    # Initialize Kalshi client
    api_key = os.getenv('KALSHI_API_KEY')
    demo_mode = os.getenv('KALSHI_DEMO_MODE', 'false').lower() == 'true'

    if not api_key and not demo_mode:
        print("\n⚠️  Warning: KALSHI_API_KEY not found in environment")
        print("Set KALSHI_API_KEY environment variable or enable demo mode")
        print("\nTo use real API:")
        print("  export KALSHI_API_KEY='your_api_key'")
        print("\nTo use demo mode:")
        print("  export KALSHI_DEMO_MODE='true'")
        print("\nProceeding with demo mode for now...")
        demo_mode = True

    client = KalshiClient(api_key=api_key, demo_mode=demo_mode)

    if demo_mode:
        print("\n🔧 Running in DEMO MODE (using mock data)")
    else:
        print(f"\n✓ Connected to Kalshi API")

    # Calculate tomorrow at 4pm EST (EST is UTC-5)
    est_offset = timezone(timedelta(hours=-5))
    now_est = datetime.now(est_offset)
    tomorrow_4pm = now_est.replace(hour=16, minute=0, second=0, microsecond=0) + timedelta(days=1)

    print(f"\nSearching for S&P Price markets...")
    print(f"Target time: {tomorrow_4pm.strftime('%Y-%m-%d %I:%M %p %Z')}")

    # Get all active markets
    print("\nFetching active markets from Kalshi...")
    markets = client.get_markets(status='active', limit=1000)

    if not markets:
        print("\n❌ No markets found")
        return

    print(f"✓ Found {len(markets)} active markets")

    # Filter for S&P related markets
    sp_keywords = ['s&p', 'sp500', 'spx', 's&p 500', 'sp-', 'inx-']
    sp_markets = []

    for market in markets:
        ticker = market.get('ticker', '').lower()
        title = market.get('title', '').lower()

        if any(keyword in ticker or keyword in title for keyword in sp_keywords):
            sp_markets.append(market)

    print(f"\nFiltered to {len(sp_markets)} S&P related markets")

    if not sp_markets:
        print("\n❌ No S&P related markets found")
        print("\nAvailable market samples:")
        for i, market in enumerate(markets[:5]):
            print(f"  {i+1}. {market.get('ticker', 'N/A')}: {market.get('title', 'N/A')[:60]}...")
        return

    # Display S&P markets
    print("\n" + "="*80)
    print("S&P RELATED MARKETS")
    print("="*80)

    for i, market in enumerate(sp_markets, 1):
        print(f"\n{i}. {market.get('ticker', 'N/A')}")
        print(f"   {market.get('title', 'N/A')}")
        if 'yes_price' in market:
            print(f"   Current Price: {market['yes_price']}¢")
        if 'volume' in market:
            print(f"   Volume: ${market.get('volume', 0):,.0f}")
        if 'close_time' in market:
            print(f"   Closes: {market['close_time']}")

    # Get detailed data for the first S&P market
    if sp_markets:
        print("\n" + "="*80)
        print("DETAILED DATA FOR TOP S&P MARKET")
        print("="*80)

        ticker = sp_markets[0]['ticker']
        print(f"\nFetching detailed data for {ticker}...")

        market_details = client.get_market(ticker)
        if market_details:
            format_market_data(market_details)

            # Get recent trades
            print(f"\nFetching recent trades for {ticker}...")
            trades_df = client.get_trades(ticker, limit=100)

            if not trades_df.empty:
                print(f"\n✓ Fetched {len(trades_df)} recent trades")
                print("\nRecent Trade Summary:")
                print(f"  Average Price: {trades_df['price'].mean():.2f}¢")
                print(f"  Price Range: {trades_df['price'].min():.2f}¢ - {trades_df['price'].max():.2f}¢")
                print(f"  Average Volume: ${trades_df['volume'].mean():,.2f}")
                print(f"  Total Volume: ${trades_df['volume'].sum():,.2f}")

                print("\nLast 5 Trades:")
                print(f"{'Timestamp':<20} {'Price':<10} {'Volume':<15}")
                print("-" * 50)
                for _, trade in trades_df.tail(5).iterrows():
                    ts = trade['timestamp'].strftime('%Y-%m-%d %H:%M') if 'timestamp' in trade else 'N/A'
                    price = f"{trade['price']:.2f}¢" if 'price' in trade else 'N/A'
                    volume = f"${trade['volume']:,.2f}" if 'volume' in trade else 'N/A'
                    print(f"{ts:<20} {price:<10} {volume:<15}")
            else:
                print("\n⚠️  No recent trades found")
        else:
            print(f"\n❌ Could not fetch detailed data for {ticker}")

    # Get market summary
    if sp_markets:
        print("\n" + "="*80)
        print("MARKET SUMMARY")
        print("="*80)

        ticker = sp_markets[0]['ticker']
        summary = client.get_market_summary(ticker)

        if summary:
            print(f"\nTicker: {summary.get('ticker', 'N/A')}")
            print(f"Title: {summary.get('title', 'N/A')}")
            print(f"\nPrice: {summary.get('current_price', 0)}¢")
            print(f"Volume: ${summary.get('volume', 0):,.0f}")
            print(f"Open Interest: {summary.get('open_interest', 0):,}")
            print(f"Status: {summary.get('status', 'N/A')}")

            if 'n_trades' in summary:
                print(f"\nTrade Statistics:")
                print(f"  Number of Trades: {summary['n_trades']:,}")
                print(f"  Average Trade Size: ${summary.get('avg_trade_size', 0):,.2f}")
                print(f"  Median Trade Size: ${summary.get('median_trade_size', 0):,.2f}")
                print(f"  Max Trade Size: ${summary.get('max_trade_size', 0):,.2f}")
                print(f"  24h Volume: ${summary.get('total_volume_24h', 0):,.2f}")

    print("\n" + "="*80)
    print("Fetch complete!")
    print("="*80)

    client.close()


if __name__ == '__main__':
    main()
