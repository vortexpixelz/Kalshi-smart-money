"""
Basic usage example for smart money detection

This example demonstrates:
1. Loading trade data
2. Fitting the detector
3. Making predictions
4. Using active learning to select manual reviews
5. Adding feedback and updating weights
"""
import numpy as np
import pandas as pd
from smart_money_detection import SmartMoneyDetector
from smart_money_detection.config import Config

# Example: Generate synthetic trade data
def generate_synthetic_trades(n_trades=1000, n_informed=50):
    """Generate synthetic trade data with some informed trades"""
    np.random.seed(42)

    # Normal trades: moderate volumes
    normal_volumes = np.random.lognormal(mean=3, sigma=1, size=n_trades - n_informed)

    # Informed trades: larger volumes
    informed_volumes = np.random.lognormal(mean=5, sigma=0.5, size=n_informed)

    # Combine
    volumes = np.concatenate([normal_volumes, informed_volumes])

    # Timestamps (one per hour)
    timestamps = pd.date_range('2024-01-01', periods=n_trades, freq='1H')

    # Prices (random walk)
    prices = 50 + np.cumsum(np.random.randn(n_trades) * 0.1)

    # Create DataFrame
    trades = pd.DataFrame({
        'timestamp': timestamps,
        'volume': volumes,
        'price': prices,
        'trade_id': range(n_trades),
    })

    # True labels (for evaluation only - not available in practice)
    true_labels = np.zeros(n_trades)
    true_labels[-n_informed:] = 1  # Last n_informed are smart money

    return trades, true_labels


def main():
    print("=== Smart Money Detection Example ===\n")

    # 1. Generate synthetic data
    print("1. Generating synthetic trade data...")
    trades, true_labels = generate_synthetic_trades(n_trades=1000, n_informed=50)
    print(f"   Generated {len(trades)} trades ({true_labels.sum():.0f} informed)\n")

    # 2. Initialize detector with custom config
    print("2. Initializing smart money detector...")
    config = Config()
    config.ensemble.weighting_method = 'thompson'  # Thompson Sampling
    config.active_learning.query_strategy = 'qbc'  # Query-by-Committee

    detector = SmartMoneyDetector(config)
    print(f"   Using {config.ensemble.weighting_method} weighting\n")

    # 3. Fit detector on historical data
    print("3. Fitting detector on historical trades...")
    detector.fit(trades, volume_col='volume', timestamp_col='timestamp', price_col='price')
    print("   Detector fitted successfully\n")

    # 4. Make predictions
    print("4. Making predictions on new trades...")
    predictions = detector.predict(trades, volume_col='volume', timestamp_col='timestamp')
    scores = detector.score(trades, volume_col='volume', timestamp_col='timestamp')

    n_detected = predictions.sum()
    print(f"   Detected {n_detected} potential smart money trades\n")

    # 5. Check initial ensemble weights
    print("5. Initial ensemble weights:")
    weights = detector.get_ensemble_weights()
    for name, weight in weights.items():
        print(f"   {name}: {weight:.3f}")
    print()

    # 6. Suggest trades for manual review (active learning)
    print("6. Suggesting trades for manual review...")
    n_reviews = 10
    query_indices, suggested_trades = detector.suggest_manual_reviews(
        trades, volume_col='volume', n_queries=n_reviews
    )

    print(f"   Selected {len(query_indices)} trades with highest disagreement")
    print(f"   Trade IDs: {suggested_trades['trade_id'].tolist()}\n")

    # 7. Simulate manual reviews (in practice, human would label these)
    print("7. Simulating manual reviews...")
    # Get true labels for suggested trades (cheating for demo)
    sample_ids = suggested_trades['trade_id'].tolist()
    manual_labels = true_labels[query_indices]

    # Add feedback
    detector.add_feedback(
        sample_ids,
        manual_labels,
        trades=trades.set_index('trade_id'),
        volume_col='volume',
        update_weights=True,
    )

    print(f"   Added {len(manual_labels)} manual reviews")
    print(f"   Found {manual_labels.sum():.0f} informed trades in review batch\n")

    # 8. Check updated weights
    print("8. Updated ensemble weights after feedback:")
    updated_weights = detector.get_ensemble_weights()
    for name, weight in updated_weights.items():
        change = weight - weights[name]
        print(f"   {name}: {weight:.3f} ({change:+.3f})")
    print()

    # 9. Get feedback statistics
    print("9. Feedback statistics:")
    stats = detector.get_feedback_statistics()
    print(f"   Total reviews: {stats['n_total']}")
    print(f"   Positive (smart money): {stats['n_positive']}")
    print(f"   Negative (normal): {stats['n_negative']}")
    print(f"   Optimal threshold: {stats['optimal_threshold']:.3f}\n")

    # 10. Evaluate performance (using true labels - only for demo)
    print("10. Evaluating performance on all trades:")
    metrics = detector.evaluate(trades, true_labels, volume_col='volume')

    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")

    if 'roc_auc' in metrics:
        print(f"   ROC AUC: {metrics['roc_auc']:.3f}")

    print()

    # 11. Save detector state
    print("11. Saving detector state...")
    detector.save_state('detector_state.pkl')
    print("    Saved to detector_state.pkl\n")

    print("=== Example Complete ===")


if __name__ == '__main__':
    main()
