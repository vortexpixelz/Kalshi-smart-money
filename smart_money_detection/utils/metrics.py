"""
Evaluation metrics for anomaly detection and smart money identification
"""
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


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
    metrics = {}

    # Confusion matrix components
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # False positive/negative rates
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Score-based metrics if available
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['avg_precision'] = average_precision_score(y_true, y_scores)
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
    return f1_score(y_true, y_pred, zero_division=0)


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
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

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
    best_metric = 0
    best_threshold = 0.5

    thresholds = np.arange(min_threshold, max_threshold + step, step)

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if metric == 'f1':
            current_metric = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            current_metric = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            current_metric = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            current_metric = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold

    return best_threshold, best_metric


def compute_ensemble_diversity(predictions: np.ndarray) -> float:
    """
    Compute ensemble diversity using pairwise disagreement

    Args:
        predictions: Binary predictions from ensemble members
                    Shape: (n_samples, n_members)

    Returns:
        Diversity score in [0, 1] (higher = more diverse)
    """
    n_samples, n_members = predictions.shape

    if n_members < 2:
        return 0.0

    # Compute pairwise disagreement
    total_disagreement = 0
    n_pairs = 0

    for i in range(n_members):
        for j in range(i + 1, n_members):
            # Disagreement = fraction of samples where predictions differ
            disagreement = (predictions[:, i] != predictions[:, j]).mean()
            total_disagreement += disagreement
            n_pairs += 1

    diversity = total_disagreement / n_pairs if n_pairs > 0 else 0.0

    return diversity
