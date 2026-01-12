"""
Evaluation metrics for anomaly detection and smart money identification
"""
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _ensure_1d_boolean(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(array).reshape(-1)
    if arr.dtype != np.bool_:
        if not np.isin(np.unique(arr), [0, 1]).all():
            raise ValueError(f"{name} must contain binary values")
        arr = arr.astype(bool)
    return arr


def _ensure_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_scores: Predicted scores (for AUC metrics)

    Returns:
        Dictionary of metrics
    """
    y_true_bool = _ensure_1d_boolean(y_true, name="y_true")
    y_pred_bool = _ensure_1d_boolean(y_pred, name="y_pred")
    _ensure_same_length(y_true_bool, y_pred_bool)

    metrics: Dict[str, float] = {}

    tp = np.logical_and(y_pred_bool, y_true_bool).sum()
    fp = np.logical_and(y_pred_bool, np.logical_not(y_true_bool)).sum()
    tn = np.logical_and(np.logical_not(y_pred_bool), np.logical_not(y_true_bool)).sum()
    fn = np.logical_and(np.logical_not(y_pred_bool), y_true_bool).sum()

    metrics['tp'] = float(tp)
    metrics['fp'] = float(fp)
    metrics['tn'] = float(tn)
    metrics['fn'] = float(fn)

    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true_bool, y_pred_bool))
    metrics['precision'] = float(precision_score(y_true_bool, y_pred_bool, zero_division=0))
    metrics['recall'] = float(recall_score(y_true_bool, y_pred_bool, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true_bool, y_pred_bool, zero_division=0))

    # Specificity and sensitivity
    metrics['specificity'] = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    metrics['sensitivity'] = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    # False positive/negative rates
    metrics['fpr'] = float(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    metrics['fnr'] = float(fn / (fn + tp) if (fn + tp) > 0 else 0.0)

    # Score-based metrics if available
    if y_scores is not None:
        scores = np.asarray(y_scores).reshape(-1)
        _ensure_same_length(y_true_bool, scores)
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true_bool, scores))
            metrics['avg_precision'] = float(average_precision_score(y_true_bool, scores))
        except ValueError:
            # Handle case where only one class is present
            pass

    return metrics


def compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute F1 score

    F1 = 2·TP / (2·TP + FP + FN)

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        F1 score
    """
    y_true_bool = _ensure_1d_boolean(y_true, name="y_true")
    y_pred_bool = _ensure_1d_boolean(y_pred, name="y_pred")
    _ensure_same_length(y_true_bool, y_pred_bool)
    return float(f1_score(y_true_bool, y_pred_bool, zero_division=0))


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute precision and recall

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Tuple of (precision, recall)
    """
    y_true_bool = _ensure_1d_boolean(y_true, name="y_true")
    y_pred_bool = _ensure_1d_boolean(y_pred, name="y_pred")
    _ensure_same_length(y_true_bool, y_pred_bool)

    precision = float(precision_score(y_true_bool, y_pred_bool, zero_division=0))
    recall = float(recall_score(y_true_bool, y_pred_bool, zero_division=0))

    return precision, recall


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f1',
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    step: float = 0.01,
) -> Tuple[float, float]:
    """
    Find optimal decision threshold for a given metric

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        min_threshold: Minimum threshold to search
        max_threshold: Maximum threshold to search
        step: Step size for grid search

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    y_true_bool = _ensure_1d_boolean(y_true, name="y_true")
    scores = np.asarray(y_scores, dtype=float).reshape(-1)
    _ensure_same_length(y_true_bool, scores)

    thresholds = np.arange(min_threshold, max_threshold + step, step)

    metric_lookup = {
        'f1': lambda yt, yp: f1_score(yt, yp, zero_division=0),
        'precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
        'recall': lambda yt, yp: recall_score(yt, yp, zero_division=0),
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
    }

    if metric not in metric_lookup:
        raise ValueError(f"Unknown metric: {metric}")

    best_threshold = 0.5
    best_metric = -np.inf

    for threshold in thresholds:
        y_pred_bool = scores >= threshold
        current_metric = metric_lookup[metric](y_true_bool, y_pred_bool)
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold

    return float(best_threshold), float(best_metric)


def compute_ensemble_diversity(predictions: np.ndarray) -> float:
    """
    Compute ensemble diversity using pairwise disagreement

    Args:
        predictions: Binary predictions from ensemble members
                    Shape: (n_samples, n_members)

    Returns:
        Diversity score in [0, 1] (higher = more diverse)
    """
    array = np.asarray(predictions)
    if array.ndim != 2:
        raise ValueError("predictions must be a 2-D array")

    n_samples, n_members = array.shape
    if n_members < 2:
        return 0.0

    disagreement_matrix = array[:, :, None] != array[:, None, :]
    upper_triangle = np.triu_indices(n_members, k=1)
    pairwise_disagreement = disagreement_matrix[:, upper_triangle[0], upper_triangle[1]].mean(axis=0)
    return float(pairwise_disagreement.mean())
