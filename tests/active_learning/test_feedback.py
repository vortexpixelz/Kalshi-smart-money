import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from smart_money_detection.active_learning.feedback import FeedbackManager


def test_get_labeled_data_marks_missing_scores_with_nan():
    manager = FeedbackManager(optimize_f1=False)
    manager.add_feedback('sample_with_score', 1, score=0.9)
    manager.add_feedback('sample_without_score', 0)

    sample_ids, labels, scores = manager.get_labeled_data()

    assert sample_ids == ['sample_with_score', 'sample_without_score']
    np.testing.assert_array_equal(labels, np.array([1, 0]))
    assert np.isnan(scores[1])
    assert scores[0] == pytest.approx(0.9)


def _compute_expected_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.3, 0.71, 0.01):
        predictions = (scores >= threshold).astype(int)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        if (tp + fp) == 0 or (tp + fn) == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def test_threshold_optimization_ignores_missing_scores():
    manager = FeedbackManager(optimize_f1=True)

    valid_scores = [0.9, 0.8, 0.7, 0.6, 0.2]
    valid_labels = [1, 1, 1, 0, 0]
    for idx, (score, label) in enumerate(zip(valid_scores, valid_labels)):
        manager.add_feedback(f'valid_{idx}', label, score=score)

    # Add positive labels without scores - these should not bias the threshold
    for idx in range(5):
        manager.add_feedback(f'missing_{idx}', 1, score=None)

    # Manually update the threshold to ensure the latest data is used
    manager._update_optimal_threshold()

    _, labels, scores = manager.get_labeled_data()
    mask = ~np.isnan(scores)
    expected_threshold = _compute_expected_threshold(labels[mask], scores[mask])

    assert mask.sum() == len(valid_scores)
    assert manager.optimal_threshold == pytest.approx(expected_threshold)

    # Ensure the expected threshold would differ if missing scores were imputed with 0.5
    imputed_threshold = _compute_expected_threshold(labels, np.nan_to_num(scores, nan=0.5))
    assert imputed_threshold != pytest.approx(expected_threshold)
