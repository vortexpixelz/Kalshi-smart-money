"""
Basic usage example for smart money detection

This example demonstrates:
1. Loading trade data from CSV
2. Fitting the detector
3. Making predictions
4. Using active learning to select manual reviews
5. Adding feedback and updating weights (if labels are available)
"""
import pandas as pd
from smart_money_detection import SmartMoneyDetector
from smart_money_detection.config import load_config

def load_trades(csv_path: str) -> pd.DataFrame:
    """Load trade data from a CSV file."""
    trades = pd.read_csv(csv_path, parse_dates=["timestamp"])
    required_columns = {"timestamp", "volume", "price"}
    missing = required_columns - set(trades.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "trade_id" not in trades.columns:
        trades["trade_id"] = range(len(trades))
    return trades


def main():
    print("=== Smart Money Detection Example ===\n")

    # 1. Load trade data
    print("1. Loading trade data...")
    csv_path = "trades.csv"
    trades = load_trades(csv_path)
    print(f"   Loaded {len(trades)} trades from {csv_path}\n")

    # 2. Initialize detector with custom config
    print("2. Initializing smart money detector...")
    config = load_config()
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

    # 7. Add feedback (requires labels in the CSV)
    if "label" in trades.columns:
        print("7. Adding manual reviews from labels column...")
        sample_ids = suggested_trades['trade_id'].tolist()
        manual_labels = trades.loc[query_indices, "label"].values

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

        # 10. Evaluate performance
        print("10. Evaluating performance on all trades:")
        metrics = detector.evaluate(trades, trades["label"].values, volume_col='volume')

        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")

        if 'roc_auc' in metrics:
            print(f"   ROC AUC: {metrics['roc_auc']:.3f}")

        print()
    else:
        print("7. Skipping feedback/evaluation (no 'label' column in CSV)\n")

    # 11. Save detector state
    print("11. Saving detector state...")
    detector.save_state('detector_state.pkl')
    print("    Saved to detector_state.pkl\n")

    print("=== Example Complete ===")


if __name__ == '__main__':
    main()
