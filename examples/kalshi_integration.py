"""
Example integration with Kalshi API for real prediction market data

This demonstrates how to:
1. Connect to Kalshi API
2. Fetch market and trade data
3. Detect smart money trades
4. Use VPIN for informed trading detection
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from smart_money_detection import SmartMoneyDetector
from smart_money_detection.config import load_config
from smart_money_detection.models import VPINClassifier


class KalshiDataFetcher:
    """
    Mock Kalshi API client for demonstration

    In practice, replace with actual Kalshi API calls
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def get_market(self, market_id: str) -> dict:
        """Get market information"""
        # Mock data
        return {
            'market_id': market_id,
            'title': 'Will candidate win election?',
            'close_time': datetime.now() + timedelta(days=30),
            'volume': 1000000,
            'open_interest': 500000,
        }

    def get_trades(self, market_id: str, limit: int = 1000) -> pd.DataFrame:
        """Get recent trades for a market"""
        # Generate mock trade data
        np.random.seed(42)

        n_trades = limit
        timestamps = pd.date_range(
            end=datetime.now(), periods=n_trades, freq='5min'
        )

        # Volume distribution: mostly small, some large
        volumes = np.random.lognormal(mean=4, sigma=2, size=n_trades)

        # Add some very large trades (potential smart money)
        n_large = int(n_trades * 0.05)
        large_indices = np.random.choice(n_trades, size=n_large, replace=False)
        volumes[large_indices] *= 5

        # Prices (binary market: 0-100 cents)
        prices = 45 + np.cumsum(np.random.randn(n_trades) * 0.5)
        prices = np.clip(prices, 1, 99)

        trades = pd.DataFrame({
            'timestamp': timestamps,
            'volume': volumes,
            'price': prices,
            'trade_id': range(n_trades),
            'market_id': market_id,
        })

        return trades


def detect_smart_money_kalshi(market_id: str, api_key: str = None):
    """
    Detect smart money trades in a Kalshi market

    Args:
        market_id: Kalshi market identifier
        api_key: Kalshi API key (optional for demo)
    """
    print(f"=== Smart Money Detection for Kalshi Market: {market_id} ===\n")

    # 1. Initialize Kalshi client
    print("1. Connecting to Kalshi API...")
    client = KalshiDataFetcher(api_key)

    market = client.get_market(market_id)
    print(f"   Market: {market['title']}")
    print(f"   Volume: ${market['volume']:,.0f}")
    print(f"   Open Interest: ${market['open_interest']:,.0f}\n")

    # 2. Fetch recent trades
    print("2. Fetching recent trades...")
    trades = client.get_trades(market_id, limit=1000)
    print(f"   Fetched {len(trades)} trades\n")

    # 3. Initialize detector with market-specific config
    print("3. Initializing smart money detector...")
    config = load_config()

    # Adjust thresholds based on market size
    if market['volume'] > 500000:
        # Large market: higher absolute thresholds
        config.smart_money.major_market_dollar_threshold = 10000
        config.smart_money.major_market_oi_percent = 0.01
    else:
        # Smaller market: lower thresholds, higher multipliers
        config.smart_money.niche_market_dollar_threshold = 1000
        config.smart_money.niche_market_oi_percent = 0.05

    detector = SmartMoneyDetector(config)
    print("   Detector initialized\n")

    # 4. Fit and detect
    print("4. Fitting detector on trade history...")
    detector.fit(trades, volume_col='volume', timestamp_col='timestamp', price_col='price')

    print("5. Detecting smart money trades...")
    predictions = detector.predict(trades, volume_col='volume', timestamp_col='timestamp')
    scores = detector.score(trades, volume_col='volume', timestamp_col='timestamp')

    # 6. Analyze results
    print(f"\n6. Detection Results:")
    smart_money_trades = trades[predictions == 1].copy()
    smart_money_trades['anomaly_score'] = scores[predictions == 1]

    print(f"   Total trades: {len(trades)}")
    print(f"   Smart money trades detected: {len(smart_money_trades)}")
    print(f"   Detection rate: {len(smart_money_trades)/len(trades)*100:.1f}%\n")

    # 7. Show top detected trades
    print("7. Top 10 Detected Smart Money Trades:")
    top_trades = smart_money_trades.nlargest(10, 'anomaly_score')

    print(f"{'Trade ID':<10} {'Timestamp':<20} {'Volume':<12} {'Price':<8} {'Score':<8}")
    print("-" * 70)
    for _, trade in top_trades.iterrows():
        print(
            f"{trade['trade_id']:<10} "
            f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
            f"${trade['volume']:>10.2f}  "
            f"{trade['price']:>6.2f}  "
            f"{trade['anomaly_score']:>6.3f}"
        )

    print()

    # 8. Use VPIN for order flow toxicity
    print("8. Computing VPIN (order flow toxicity)...")
    vpin_classifier = VPINClassifier(threshold=0.75, n_buckets=50)

    vpin_scores = vpin_classifier.score(
        trades['price'].values, trades['volume'].values
    )

    print(f"   VPIN computed for {len(vpin_scores)} volume buckets")
    print(f"   Mean VPIN: {np.mean(vpin_scores):.3f}")
    print(f"   Max VPIN: {np.max(vpin_scores):.3f}")
    print(f"   High toxicity periods (VPIN > 0.75): {(vpin_scores > 0.75).sum()}\n")

    # 9. Suggest manual reviews
    print("9. Suggesting trades for manual review...")
    n_reviews = 5
    query_indices, suggested = detector.suggest_manual_reviews(
        trades, volume_col='volume', n_queries=n_reviews
    )

    print(f"   Selected {len(suggested)} trades for review:")
    for _, trade in suggested.iterrows():
        print(
            f"   - Trade {trade['trade_id']}: ${trade['volume']:.2f} "
            f"at {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        )

    print()

    # 10. Ensemble analysis
    print("10. Ensemble Detector Analysis:")
    weights = detector.get_ensemble_weights()
    print("    Current weights:")
    for name, weight in weights.items():
        print(f"    - {name}: {weight:.3f}")

    print()

    print("=== Detection Complete ===")

    return detector, smart_money_trades


def main():
    """Run Kalshi integration example"""

    # Example market ID (replace with actual Kalshi market)
    market_id = "PRES-2024-WINNER"

    # Run detection
    detector, smart_money_trades = detect_smart_money_kalshi(market_id)

    # Optional: Save results
    smart_money_trades.to_csv(f'smart_money_{market_id}.csv', index=False)
    print(f"\nSaved results to smart_money_{market_id}.csv")


if __name__ == '__main__':
    main()
