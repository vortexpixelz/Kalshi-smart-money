import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_money_detection.active_learning.feedback import FeedbackManager


def legacy_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.3, 0.71, 0.01):
        predictions = (scores >= threshold).astype(int)

        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        if (tp + fp) > 0 and (tp + fn) > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    return best_threshold


def build_manager(labels: np.ndarray, scores: np.ndarray) -> FeedbackManager:
    manager = FeedbackManager(optimize_f1=False)

    for idx, (label, score) in enumerate(zip(labels, scores)):
        manager.feedback_data.append(
            {
                'sample_id': idx,
                'y_true': int(label),
                'y_pred': None,
                'score': float(score),
                'timestamp': '1970-01-01T00:00:00',
                'metadata': {},
            }
        )

    manager.n_total = len(labels)
    manager.n_positive = int(labels.sum())
    manager.n_negative = manager.n_total - manager.n_positive

    return manager


def test_update_optimal_threshold_matches_legacy():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=50)
    scores = rng.random(50)

    manager = build_manager(labels, scores)
    manager._update_optimal_threshold()

    expected_threshold = legacy_optimal_threshold(labels, scores)

    assert np.isclose(manager.optimal_threshold, expected_threshold)


def test_update_optimal_threshold_handles_zero_denominator():
    labels = np.zeros(20, dtype=int)
    scores = np.linspace(0, 1, 20)

    manager = build_manager(labels, scores)
    manager.optimal_threshold = 0.5
    manager._update_optimal_threshold()

    expected_threshold = legacy_optimal_threshold(labels, scores)

    assert manager.optimal_threshold == expected_threshold == 0.5
