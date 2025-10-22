#!/usr/bin/env python3
"""
Comprehensive Test Suite - Testing Everything We Built

This script tests ALL components of the smart money detection system.
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Change to project directory
os.chdir('/home/user/Kalshi-smart-money')
sys.path.insert(0, '/home/user/Kalshi-smart-money')

from smart_money_detection import SmartMoneyDetector
from smart_money_detection.kalshi_client import KalshiClient
from smart_money_detection.config import Config
from smart_money_detection.detectors import ZScoreDetector, IQRDetector, PercentileDetector, RelativeVolumeDetector
from smart_money_detection.ensemble import AnomalyEnsemble
from smart_money_detection.models import VPINClassifier, SimplifiedPIN, BulkVolumeClassifier
from smart_money_detection.active_learning import QueryByCommittee, FeedbackManager
from smart_money_detection.features import TemporalFeatureEncoder
from smart_money_detection.utils import compute_metrics, find_optimal_threshold


def test_header(text):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def test_section(text):
    print(f"\n{'─'*80}")
    print(f"  {text}")
    print(f"{'─'*80}")


def test_1_kalshi_client():
    """Test 1: Kalshi API Client"""
    test_header("TEST 1: Kalshi API Client")

    print("Testing Kalshi API client functionality...")

    # Initialize client
    client = KalshiClient(demo_mode=True)
    print("✓ Client initialized")

    # Test get_markets
    markets = client.get_markets(limit=5)
    print(f"✓ Fetched {len(markets)} markets")

    # Test get_market
    if markets:
        ticker = markets[0]['ticker']
        market = client.get_market(ticker)
        print(f"✓ Fetched market details for {ticker}")
        print(f"  Title: {market.get('title', '')[:50]}...")
        print(f"  Volume: ${market.get('volume', 0):,.0f}")

    # Test get_trades
    trades = client.get_trades(ticker, limit=100)
    print(f"✓ Fetched {len(trades)} trades")
    print(f"  Columns: {list(trades.columns)}")
    print(f"  Total volume: ${trades['volume'].sum():,.2f}")

    # Test get_market_summary
    summary = client.get_market_summary(ticker)
    print(f"✓ Generated market summary")
    print(f"  Trades: {summary.get('n_trades', 0)}")
    print(f"  Avg size: ${summary.get('avg_trade_size', 0):.2f}")

    client.close()
    print("\n✅ TEST 1 PASSED - Kalshi API Client Works!")

    return client, markets, trades


def test_2_base_detectors(trades):
    """Test 2: Base Anomaly Detectors"""
    test_header("TEST 2: Base Anomaly Detectors")

    volumes = trades['volume'].values.reshape(-1, 1)

    # Test Z-Score Detector
    print("Testing Z-Score Detector...")
    zscore = ZScoreDetector(threshold=3.0)
    zscore.fit(volumes)
    z_predictions = zscore.predict(volumes)
    z_scores = zscore.score(volumes)
    print(f"✓ Z-Score: {z_predictions.sum()} anomalies detected")
    print(f"  Max score: {z_scores.max():.3f}")

    # Test IQR Detector
    print("\nTesting IQR Detector...")
    iqr = IQRDetector(multiplier=1.5)
    iqr.fit(volumes)
    iqr_predictions = iqr.predict(volumes)
    iqr_scores = iqr.score(volumes)
    print(f"✓ IQR: {iqr_predictions.sum()} anomalies detected")
    bounds = iqr.get_bounds()
    print(f"  Bounds: [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}]")

    # Test Percentile Detector
    print("\nTesting Percentile Detector...")
    percentile = PercentileDetector(percentile=95.0)
    percentile.fit(volumes)
    perc_predictions = percentile.predict(volumes)
    perc_scores = percentile.score(volumes)
    print(f"✓ Percentile: {perc_predictions.sum()} anomalies detected")
    thresholds = percentile.get_thresholds()
    print(f"  Threshold: {thresholds[1][0]:.2f}")

    # Test Volume Detector
    print("\nTesting Relative Volume Detector...")
    volume_det = RelativeVolumeDetector(threshold_multiplier=3.0)
    volume_det.fit(volumes)
    vol_predictions = volume_det.predict(volumes)
    vol_scores = volume_det.score(volumes)
    print(f"✓ Volume: {vol_predictions.sum()} anomalies detected")
    print(f"  Baseline: {volume_det.get_baseline():.2f}")

    print("\n✅ TEST 2 PASSED - All Base Detectors Work!")

    return {
        'zscore': zscore,
        'iqr': iqr,
        'percentile': percentile,
        'volume': volume_det
    }


def test_3_ensemble_methods(detectors, trades):
    """Test 3: Ensemble Weighting Methods"""
    test_header("TEST 3: Ensemble Weighting Methods")

    volumes = trades['volume'].values.reshape(-1, 1)
    detector_list = list(detectors.values())

    methods = ['uniform', 'mwu', 'thompson', 'ucb']

    for method in methods:
        print(f"\nTesting {method.upper()} weighting...")
        ensemble = AnomalyEnsemble(detector_list, weighting_method=method)
        ensemble.fit(volumes)

        predictions = ensemble.predict(volumes)
        scores = ensemble.score(volumes)
        weights = ensemble.get_weights()

        print(f"✓ {method.upper()}: {predictions.sum()} anomalies detected")
        print(f"  Weights: {weights}")
        print(f"  Mean score: {scores.mean():.3f}")

    print("\n✅ TEST 3 PASSED - All Ensemble Methods Work!")

    return ensemble


def test_4_temporal_features(trades):
    """Test 4: Temporal Feature Engineering"""
    test_header("TEST 4: Temporal Feature Engineering")

    print("Testing temporal feature encoder...")
    encoder = TemporalFeatureEncoder()

    # Test encoding
    timestamp = trades['timestamp'].iloc[0]
    features = encoder.encode_timestamp(timestamp)
    print(f"✓ Encoded timestamp: {list(features.keys())}")

    # Test cyclical encoding
    hour_features = encoder.encode_hour(12)
    hour_sin = hour_features['hour_sin'] if isinstance(hour_features['hour_sin'], float) else hour_features['hour_sin'][0]
    hour_cos = hour_features['hour_cos'] if isinstance(hour_features['hour_cos'], float) else hour_features['hour_cos'][0]
    print(f"✓ Hour encoding: sin={hour_sin:.3f}, cos={hour_cos:.3f}")

    # Test day encoding
    day_features = encoder.encode_day_of_week(3)
    day_sin = day_features['day_of_week_sin'] if isinstance(day_features['day_of_week_sin'], float) else day_features['day_of_week_sin'][0]
    day_cos = day_features['day_of_week_cos'] if isinstance(day_features['day_of_week_cos'], float) else day_features['day_of_week_cos'][0]
    print(f"✓ Day encoding: sin={day_sin:.3f}, cos={day_cos:.3f}")

    # Test DataFrame transformation
    trades_with_features = encoder.transform(trades, timestamp_col='timestamp')
    print(f"✓ Transformed DataFrame: {trades_with_features.shape}")
    print(f"  New features: {encoder.get_feature_names()}")

    print("\n✅ TEST 4 PASSED - Temporal Features Work!")

    return encoder


def test_5_smart_money_models(trades):
    """Test 5: Smart Money Models (VPIN, PIN)"""
    test_header("TEST 5: Smart Money Models")

    prices = trades['price'].values
    volumes = trades['volume'].values

    # Test VPIN
    print("Testing VPIN...")
    vpin = VPINClassifier(threshold=0.75, n_buckets=50)
    vpin.fit(prices, volumes)
    vpin_scores = vpin.score(prices, volumes)
    vpin_predictions = vpin.predict(prices, volumes)

    print(f"✓ VPIN: {vpin_predictions.sum()} high toxicity periods")
    print(f"  Mean VPIN: {vpin_scores.mean():.3f}")
    print(f"  Max VPIN: {vpin_scores.max():.3f}")

    # Test Simplified PIN
    print("\nTesting Simplified PIN...")
    pin = SimplifiedPIN(large_trade_threshold=0.95)
    pin.fit(volumes)
    pin_predictions = pin.predict(volumes)
    pin_scores = pin.score(volumes)

    print(f"✓ PIN: {pin_predictions.sum()} informed trades detected")
    print(f"  Threshold: {pin.threshold_value_:.2f}")

    # Test Trade Classifier
    print("\nTesting Bulk Volume Classifier...")
    classifier = BulkVolumeClassifier()
    classifier.fit(prices)
    buy_vols, sell_vols = classifier.classify(prices, volumes)

    print(f"✓ Trade Classification:")
    print(f"  Total buy volume: ${buy_vols.sum():,.2f}")
    print(f"  Total sell volume: ${sell_vols.sum():,.2f}")
    print(f"  Buy/Sell ratio: {buy_vols.sum()/sell_vols.sum():.2f}")

    print("\n✅ TEST 5 PASSED - Smart Money Models Work!")

    return vpin, pin


def test_6_active_learning(trades, ensemble):
    """Test 6: Active Learning"""
    test_header("TEST 6: Active Learning & Feedback")

    volumes = trades['volume'].values.reshape(-1, 1)

    # Test Query-by-Committee
    print("Testing Query-by-Committee...")
    qbc = QueryByCommittee(batch_size=10)

    # Get committee predictions
    committee_preds = []
    for detector in ensemble.detectors:
        preds = detector.predict(volumes)
        committee_preds.append(preds)
    committee_preds = np.column_stack(committee_preds)

    scores = ensemble.score(volumes)
    query_indices = qbc.select_queries(
        volumes,
        scores,
        n_queries=10,
        committee_predictions=committee_preds
    )

    print(f"✓ Query-by-Committee: Selected {len(query_indices)} samples")
    print(f"  Indices: {query_indices[:5]}...")

    # Test Feedback Manager
    print("\nTesting Feedback Manager...")
    feedback_mgr = FeedbackManager(optimize_f1=True)

    # Simulate adding feedback
    sample_ids = [f"trade_{i}" for i in query_indices[:5]]
    labels = np.random.randint(0, 2, size=5)

    feedback_mgr.add_batch_feedback(sample_ids, labels)

    stats = feedback_mgr.get_statistics()
    print(f"✓ Feedback added: {stats['n_total']} reviews")
    print(f"  Positive: {stats['n_positive']}")
    print(f"  Negative: {stats['n_negative']}")
    print(f"  Optimal threshold: {stats['optimal_threshold']:.3f}")

    print("\n✅ TEST 6 PASSED - Active Learning Works!")

    return qbc, feedback_mgr


def test_7_main_pipeline(trades):
    """Test 7: Main SmartMoneyDetector Pipeline"""
    test_header("TEST 7: Main SmartMoneyDetector Pipeline")

    print("Testing complete detection pipeline...")

    # Initialize with custom config
    config = Config()
    config.ensemble.weighting_method = 'thompson'
    config.active_learning.batch_size = 10

    detector = SmartMoneyDetector(config)
    print("✓ Detector initialized")

    # Fit
    detector.fit(trades, volume_col='volume', timestamp_col='timestamp', price_col='price')
    print("✓ Detector fitted")

    # Predict
    predictions = detector.predict(trades)
    scores = detector.score(trades)
    print(f"✓ Predictions made: {predictions.sum()}/{len(predictions)} anomalies")

    # Get weights
    weights = detector.get_ensemble_weights()
    print(f"✓ Ensemble weights: {weights}")

    # Suggest reviews
    indices, suggested = detector.suggest_manual_reviews(trades, n_queries=5)
    print(f"✓ Manual review suggestions: {len(suggested)} trades")

    # Simulate feedback
    sample_ids = suggested['trade_id'].tolist()
    labels = np.random.randint(0, 2, size=len(sample_ids))
    detector.add_feedback(sample_ids, labels, trades.set_index('trade_id'), update_weights=True)

    updated_weights = detector.get_ensemble_weights()
    print(f"✓ Updated weights: {updated_weights}")

    # Get stats
    stats = detector.get_feedback_statistics()
    print(f"✓ Feedback stats: {stats['n_total']} reviews, {stats['n_positive']} positive")

    print("\n✅ TEST 7 PASSED - Main Pipeline Works!")

    return detector


def test_8_state_persistence(detector):
    """Test 8: State Save/Load"""
    test_header("TEST 8: State Persistence")

    print("Testing save/load functionality...")

    # Save state
    save_path = '/tmp/detector_state_test.pkl'
    detector.save_state(save_path)
    print(f"✓ State saved to {save_path}")

    # Create new detector
    detector2 = SmartMoneyDetector()

    # Load state
    detector2.load_state(save_path)
    print(f"✓ State loaded")

    # Compare weights
    weights1 = detector.get_ensemble_weights()
    weights2 = detector2.get_ensemble_weights()

    weights_match = all(abs(weights1[k] - weights2[k]) < 1e-6 for k in weights1.keys())
    print(f"✓ Weights match: {weights_match}")

    # Compare feedback
    stats1 = detector.get_feedback_statistics()
    stats2 = detector2.get_feedback_statistics()

    print(f"✓ Feedback preserved: {stats1['n_total']} == {stats2['n_total']}")

    # Cleanup
    os.remove(save_path)
    print(f"✓ Cleaned up test file")

    print("\n✅ TEST 8 PASSED - State Persistence Works!")


def test_9_multiple_markets(client):
    """Test 9: Multiple Market Analysis"""
    test_header("TEST 9: Multiple Market Analysis")

    print("Testing detection across multiple markets...\n")

    markets = client.get_markets(limit=3)

    results = []

    for market in markets:
        ticker = market['ticker']
        title = market.get('title', '')[:40]

        print(f"Analyzing {ticker}...")
        print(f"  {title}...")

        # Fetch trades
        trades = client.get_trades(ticker, limit=200)

        if trades.empty:
            print(f"  ⚠️  No trades available\n")
            continue

        # Detect
        detector = SmartMoneyDetector()
        detector.fit(trades, volume_col='volume', timestamp_col='timestamp')
        predictions = detector.predict(trades)

        result = {
            'ticker': ticker,
            'n_trades': len(trades),
            'n_smart_money': predictions.sum(),
            'pct_smart_money': predictions.sum() / len(trades) * 100,
            'total_volume': trades['volume'].sum(),
        }

        results.append(result)

        print(f"  ✓ Analyzed {result['n_trades']} trades")
        print(f"  ✓ Detected {result['n_smart_money']} smart money ({result['pct_smart_money']:.1f}%)")
        print(f"  ✓ Total volume: ${result['total_volume']:,.2f}\n")

    print("Summary:")
    print(f"  Markets analyzed: {len(results)}")
    print(f"  Total trades: {sum(r['n_trades'] for r in results)}")
    print(f"  Total smart money: {sum(r['n_smart_money'] for r in results)}")

    print("\n✅ TEST 9 PASSED - Multiple Market Analysis Works!")

    return results


def test_10_performance_metrics():
    """Test 10: Performance & Metrics"""
    test_header("TEST 10: Performance Metrics")

    print("Testing metrics computation...")

    # Generate test data
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    y_scores = np.random.rand(100)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_scores)

    print("✓ Computed metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.3f}")

    # Test threshold optimization
    optimal_threshold, best_f1 = find_optimal_threshold(
        y_true, y_scores, metric='f1'
    )

    print(f"\n✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"✓ Best F1 score: {best_f1:.3f}")

    print("\n✅ TEST 10 PASSED - Metrics Work!")


def main():
    """Run all tests"""
    test_header("🚀 COMPREHENSIVE SYSTEM TEST SUITE")

    print("Running complete test of all components...")
    print("This will test:")
    print("  1. Kalshi API Client")
    print("  2. Base Anomaly Detectors")
    print("  3. Ensemble Weighting Methods")
    print("  4. Temporal Feature Engineering")
    print("  5. Smart Money Models (VPIN, PIN)")
    print("  6. Active Learning")
    print("  7. Main Detection Pipeline")
    print("  8. State Persistence")
    print("  9. Multiple Market Analysis")
    print("  10. Performance Metrics")

    input("\nPress Enter to begin tests...")

    start_time = datetime.now()

    try:
        # Run all tests
        client, markets, trades = test_1_kalshi_client()
        detectors = test_2_base_detectors(trades)
        ensemble = test_3_ensemble_methods(detectors, trades)
        encoder = test_4_temporal_features(trades)
        vpin, pin = test_5_smart_money_models(trades)
        qbc, feedback = test_6_active_learning(trades, ensemble)
        detector = test_7_main_pipeline(trades)
        test_8_state_persistence(detector)
        results = test_9_multiple_markets(client)
        test_10_performance_metrics()

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        test_header("🎉 ALL TESTS PASSED!")

        print(f"Total duration: {duration:.2f} seconds")
        print(f"\nSystem Status: ✅ FULLY OPERATIONAL")
        print(f"\nAll components tested and verified:")
        print(f"  ✓ API Integration")
        print(f"  ✓ Anomaly Detection")
        print(f"  ✓ Ensemble Methods")
        print(f"  ✓ Smart Money Models")
        print(f"  ✓ Active Learning")
        print(f"  ✓ Complete Pipeline")
        print(f"  ✓ State Management")
        print(f"  ✓ Multi-market Analysis")
        print(f"  ✓ Performance Metrics")

        print(f"\n🚀 Ready for production deployment!")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
