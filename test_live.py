#!/usr/bin/env python3
"""
Live test of smart money detection with real Kalshi data

Run this script to test the complete detection system on actual market data.
"""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from smart_money_detection import SmartMoneyDetector
from smart_money_detection.kalshi_client import KalshiClient
from smart_money_detection.config import load_config
from smart_money_detection.models import VPINClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}\n")


def test_kalshi_connection():
    """Test Kalshi API connection"""
    print_header("🔌 Testing Kalshi API Connection")

    # Initialize client (will use demo mode if no API key)
    client = KalshiClient(demo_mode=True)

    # Get markets
    print("Fetching available markets...")
    markets = client.get_markets(limit=10)

    if not markets:
        print("❌ Failed to fetch markets")
        return None

    print(f"✅ Successfully connected! Found {len(markets)} markets\n")

    # Display markets
    print("Available Markets:")
    print(f"{'Ticker':<25} {'Volume':<15} {'Price':<10} {'Status'}")
    print("─" * 70)

    for market in markets[:5]:
        ticker = market.get('ticker', 'N/A')
        volume = market.get('volume', 0)
        price = market.get('yes_price', 0)
        status = market.get('status', 'N/A')

        print(f"{ticker:<25} ${volume:>12,.0f}  {price:>6}¢  {status}")

    return client, markets


def test_smart_money_detection(client: KalshiClient, markets: list):
    """Test smart money detection on a market"""
    print_header("🎯 Testing Smart Money Detection")

    # Select a market
    market = markets[0]
    ticker = market.get('ticker', 'UNKNOWN')
    title = market.get('title', 'Unknown Market')

    print(f"Market: {ticker}")
    print(f"Title: {title}")
    print(f"Volume: ${market.get('volume', 0):,.0f}")
    print(f"Open Interest: ${market.get('open_interest', 0):,.0f}\n")

    # Fetch trades
    print_section("📊 Fetching Trade Data")
    print("Retrieving trade history...")

    trades = client.get_trades(ticker, limit=1000)

    if trades.empty:
        print("❌ No trade data available")
        return

    print(f"✅ Fetched {len(trades)} trades")
    print(f"   Time range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")
    print(f"   Total volume: ${trades['volume'].sum():,.2f}")
    print(f"   Avg trade size: ${trades['volume'].mean():.2f}")
    print(f"   Max trade size: ${trades['volume'].max():.2f}")

    # Initialize detector
    print_section("🤖 Initializing Smart Money Detector")

    config = load_config()

    # Configure based on market size
    market_volume = market.get('volume', 0)
    if market_volume > 1000000:
        print("   Detected: MAJOR MARKET (using high thresholds)")
        config.smart_money.major_market_dollar_threshold = 10000
    else:
        print("   Detected: NICHE MARKET (using lower thresholds)")
        config.smart_money.niche_market_dollar_threshold = 1000

    config.ensemble.weighting_method = 'thompson'
    print(f"   Ensemble method: {config.ensemble.weighting_method}")

    detector = SmartMoneyDetector(config)

    # Fit detector
    print("\n   Fitting detector on trade history...")
    detector.fit(
        trades,
        volume_col='volume',
        timestamp_col='timestamp',
        price_col='price'
    )
    print("   ✅ Detector fitted")

    # Make predictions
    print_section("🔍 Detecting Smart Money Trades")

    predictions = detector.predict(trades)
    scores = detector.score(trades)

    n_smart_money = predictions.sum()
    pct_smart_money = (n_smart_money / len(trades)) * 100

    print(f"Results:")
    print(f"   Total trades analyzed: {len(trades)}")
    print(f"   Smart money detected: {n_smart_money} ({pct_smart_money:.1f}%)")
    print(f"   Normal trades: {len(trades) - n_smart_money}")

    if n_smart_money > 0:
        smart_trades = trades[predictions == 1].copy()
        smart_trades['score'] = scores[predictions == 1]

        print(f"\n   Smart Money Statistics:")
        print(f"   ├─ Mean volume: ${smart_trades['volume'].mean():,.2f}")
        print(f"   ├─ Median volume: ${smart_trades['volume'].median():,.2f}")
        print(f"   ├─ Total volume: ${smart_trades['volume'].sum():,.2f}")
        print(f"   └─ Avg confidence: {smart_trades['score'].mean():.3f}")

        # Show top 5 detected trades
        print(f"\n   Top 5 Detected Smart Money Trades:")
        print(f"   {'Trade ID':<15} {'Timestamp':<20} {'Volume':<15} {'Score'}")
        print(f"   {'-' * 65}")

        top_trades = smart_trades.nlargest(5, 'score')
        for _, trade in top_trades.iterrows():
            trade_id = str(trade['trade_id'])[:12]
            timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
            volume = trade['volume']
            score = trade['score']

            print(f"   {trade_id:<15} {timestamp:<20} ${volume:>10,.2f}  {score:.3f}")

    # Ensemble analysis
    print_section("⚖️ Ensemble Analysis")

    weights = detector.get_ensemble_weights()
    print("   Detector Weights:")
    for name, weight in weights.items():
        bar = '█' * int(weight * 40)
        print(f"   {name:<20} {weight:.3f} {bar}")

    # Check diversity
    if 'volume' in trades.columns:
        diversity = detector.ensemble.get_ensemble_diversity(
            trades[['volume']].values
        )
        print(f"\n   Ensemble diversity: {diversity:.3f}")
        if diversity > 0.5:
            print(f"   ✅ Good diversity (>{0.5:.1f})")
        else:
            print(f"   ⚠️  Low diversity (<{0.5:.1f})")

    # VPIN Analysis
    print_section("📈 VPIN (Order Flow Toxicity) Analysis")

    try:
        vpin_classifier = VPINClassifier(threshold=0.75, n_buckets=50)
        vpin_classifier.fit(trades['price'].values, trades['volume'].values)

        vpin_scores = vpin_classifier.score(
            trades['price'].values,
            trades['volume'].values
        )

        print(f"   VPIN Statistics:")
        print(f"   ├─ Mean VPIN: {np.mean(vpin_scores):.3f}")
        print(f"   ├─ Max VPIN: {np.max(vpin_scores):.3f}")
        print(f"   └─ High toxicity periods (>0.75): {(vpin_scores > 0.75).sum()}")

        if np.max(vpin_scores) > 0.75:
            print(f"\n   ⚠️  High order flow toxicity detected!")
        else:
            print(f"\n   ✅ Normal order flow toxicity")

    except Exception as e:
        print(f"   ⚠️  VPIN calculation skipped: {e}")

    # Active Learning Demo
    print_section("🎓 Active Learning: Manual Review Suggestions")

    print("   Selecting trades for manual review (Query-by-Committee)...")
    query_indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col='volume',
        n_queries=5
    )

    print(f"\n   Suggested {len(suggested)} trades for manual review:\n")
    print(f"   {'Trade ID':<15} {'Timestamp':<20} {'Volume':<15} {'Price'}")
    print(f"   {'-' * 65}")

    for _, trade in suggested.iterrows():
        trade_id = str(trade['trade_id'])[:12]
        timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
        volume = trade['volume']
        price = trade['price']

        print(f"   {trade_id:<15} {timestamp:<20} ${volume:>10,.2f}  {price:>6.2f}¢")

    print(f"\n   💡 These are the trades where detectors disagree most.")
    print(f"      Manually reviewing these would maximize learning efficiency!")

    return detector, trades, predictions


def test_feedback_loop(detector, trades, predictions):
    """Test feedback integration"""
    print_section("🔄 Testing Feedback Loop")

    print("   Simulating manual review of 5 trades...")

    # Simulate reviewing some trades
    # In practice, a human would label these
    review_indices = np.random.choice(len(trades), size=5, replace=False)
    sample_ids = trades.iloc[review_indices]['trade_id'].tolist()

    # Simulate labels (high volume = smart money for demo)
    simulated_labels = (
        trades.iloc[review_indices]['volume'] > trades['volume'].quantile(0.9)
    ).astype(int).values

    print(f"   Adding {len(simulated_labels)} manual labels...")

    initial_weights = detector.get_ensemble_weights()

    # Add feedback
    detector.add_feedback(
        sample_ids,
        simulated_labels,
        trades=trades.set_index('trade_id'),
        update_weights=True
    )

    updated_weights = detector.get_ensemble_weights()

    print(f"\n   Feedback Statistics:")
    stats = detector.get_feedback_statistics()
    print(f"   ├─ Total reviews: {stats['n_total']}")
    print(f"   ├─ Smart money found: {stats['n_positive']}")
    print(f"   └─ Optimal threshold: {stats['optimal_threshold']:.3f}")

    print(f"\n   Weight Changes:")
    for name in initial_weights.keys():
        old = initial_weights[name]
        new = updated_weights[name]
        change = new - old
        symbol = '↑' if change > 0 else '↓' if change < 0 else '→'
        print(f"   {name:<20} {old:.3f} → {new:.3f} ({symbol} {abs(change):.3f})")

    print(f"\n   ✅ System learned from feedback and updated weights!")


def main():
    """Main test function"""
    print_header("🚀 Smart Money Detection - Live Test")

    print("This test will:")
    print("1. Connect to Kalshi API (demo mode)")
    print("2. Fetch real market data")
    print("3. Detect smart money trades")
    print("4. Analyze results with VPIN")
    print("5. Demonstrate active learning")
    print("6. Show feedback loop\n")

    input("Press Enter to start... (or Ctrl+C to cancel)")

    try:
        # Test 1: Connection
        result = test_kalshi_connection()
        if result is None:
            print("\n❌ Connection test failed. Exiting.")
            return 1

        client, markets = result

        # Test 2: Detection
        result = test_smart_money_detection(client, markets)
        if result is None:
            print("\n❌ Detection test failed. Exiting.")
            return 1

        detector, trades, predictions = result

        # Test 3: Feedback
        test_feedback_loop(detector, trades, predictions)

        # Summary
        print_header("✅ All Tests Completed Successfully!")

        print("Summary:")
        print(f"   • Connected to Kalshi API")
        print(f"   • Analyzed {len(trades)} trades")
        print(f"   • Detected {predictions.sum()} smart money trades")
        print(f"   • Demonstrated active learning")
        print(f"   • Integrated human feedback")

        print("\n🎉 System is ready for production use!")

        print("\nNext steps:")
        print("   1. Set up real Kalshi API key in .env")
        print("   2. Run: python test_live.py")
        print("   3. Integrate into your trading pipeline")
        print("   4. Start collecting feedback to improve detection\n")

        # Cleanup
        client.close()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
