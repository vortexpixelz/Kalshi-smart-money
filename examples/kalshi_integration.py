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
from smart_money_detection import SmartMoneyDetector
from smart_money_detection.config import load_config
from smart_money_detection.kalshi_client import KalshiClient
from smart_money_detection.models import VPINClassifier

def detect_smart_money_kalshi(ticker: str, api_key: str | None = None):
    """
    Detect smart money trades in a Kalshi market

    Args:
        ticker: Kalshi market ticker (e.g., "PRES-2024-WINNER")
        api_key: Kalshi API key (optional; falls back to KALSHI_API_KEY env var)
    """
    print(f"=== Smart Money Detection for Kalshi Market: {ticker} ===\n")

    # 1. Initialize Kalshi client
    print("1. Connecting to Kalshi API...")
    with KalshiClient(api_key=api_key) as client:
        market = client.get_market(ticker)
        if not market:
            raise RuntimeError(f"No market found for ticker {ticker}")

        print(f"   Market: {market['title']}")
        print(f"   Volume: ${market['volume']:,.0f}")
        print(f"   Open Interest: ${market['open_interest']:,.0f}\n")

        # 2. Fetch recent trades
        print("2. Fetching recent trades...")
        trades = client.get_trades(ticker, limit=1000)
        if trades.empty:
            raise RuntimeError(f"No trades returned for ticker {ticker}")
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

    # Example market ticker (replace with actual Kalshi market)
    ticker = "PRES-2024-WINNER"

    # Run detection
    detector, smart_money_trades = detect_smart_money_kalshi(ticker)

    # Optional: Save results
    smart_money_trades.to_csv(f'smart_money_{ticker}.csv', index=False)
    print(f"\nSaved results to smart_money_{ticker}.csv")


if __name__ == '__main__':
    main()
