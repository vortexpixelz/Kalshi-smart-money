import numpy as np
import pytest

from smart_money_detection.utils.metrics import (
    compute_metrics,
    compute_precision_recall,
    compute_f1_score,
    find_optimal_threshold,
)


def test_compute_metrics_basic():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_scores = np.array([0.1, 0.8, 0.3, 0.4, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_scores)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["tn"] == 2
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 2 / 3
    assert metrics["f1_score"] == pytest.approx(0.8, rel=1e-6)
    assert metrics["specificity"] == 1.0
    assert metrics["sensitivity"] == 2 / 3
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == pytest.approx(1 / 3)
    assert "roc_auc" in metrics
    assert "avg_precision" in metrics


def test_compute_precision_recall_matches_helpers():
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    precision, recall = compute_precision_recall(y_true, y_pred)

    assert precision == compute_metrics(y_true, y_pred)["precision"]
    assert recall == compute_metrics(y_true, y_pred)["recall"]
    assert compute_f1_score(y_true, y_pred) == pytest.approx(
        2 * (precision * recall) / (precision + recall)
    )


def test_find_optimal_threshold_returns_best_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, metric_value = find_optimal_threshold(
        y_true, y_scores, metric="f1", min_threshold=0.0, max_threshold=1.0, step=0.1
    )

    assert threshold == pytest.approx(0.3, rel=1e-6)
    assert metric_value == pytest.approx(1.0, rel=1e-6)
