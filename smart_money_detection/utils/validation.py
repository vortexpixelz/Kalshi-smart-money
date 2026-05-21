"""Cross-validation utilities with explicit typing and validation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut


def _to_numpy(array: Any, *, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return result.astype(float, copy=False)


def _validate_targets(y: Any) -> np.ndarray:
    arr = np.asarray(y).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    return arr


def loocv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, Any]:
    """Perform leave-one-out cross-validation."""

    features = _to_numpy(X, name="X")
    targets = _validate_targets(y)

    if features.shape[0] != targets.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    predictions = np.zeros(targets.shape[0])
    metrics: List[float] = []

    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        model = model_fn(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test)).reshape(-1)
        predictions[test_idx] = y_pred
        metrics.append(float(metric_fn(y_test, y_pred)))

    return {
        'predictions': predictions,
        'mean_metric': float(np.mean(metrics)),
        'std_metric': float(np.std(metrics)),
        'all_metrics': metrics,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_folds: int = 5,
    n_repeats: int = 1,
    stratified: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run (optionally repeated) k-fold cross-validation."""

    features = _to_numpy(X, name="X")
    targets = _validate_targets(y)

    if features.shape[0] != targets.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    all_metrics: List[float] = []
    all_predictions: List[np.ndarray] = []

    for repeat in range(n_repeats):
        seed = random_state + repeat

        if stratified:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(features, targets)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(features)

        fold_predictions = np.zeros(targets.shape[0])

        for train_idx, test_idx in split_iter:
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]

            model = model_fn(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test)).reshape(-1)
            fold_predictions[test_idx] = y_pred
            all_metrics.append(float(metric_fn(y_test, y_pred)))

        all_predictions.append(fold_predictions.copy())

    return {
        'predictions': all_predictions,
        'mean_metric': float(np.mean(all_metrics)) if all_metrics else 0.0,
        'std_metric': float(np.std(all_metrics)) if all_metrics else 0.0,
        'all_metrics': all_metrics,
        'n_folds': n_folds,
        'n_repeats': n_repeats,
    }


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Any],
    param_search_fn: Callable[[np.ndarray, np.ndarray, int], Dict[str, Any]],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform nested cross-validation for model selection."""

    features = _to_numpy(X, name="X")
    targets = _validate_targets(y)

    if features.shape[0] != targets.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    outer_kfold = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    outer_metrics: List[float] = []
    best_params_per_fold: List[Dict[str, Any]] = []

    for outer_train_idx, outer_test_idx in outer_kfold.split(features):
        X_outer_train, X_outer_test = features[outer_train_idx], features[outer_test_idx]
        y_outer_train, y_outer_test = targets[outer_train_idx], targets[outer_test_idx]

        best_params = param_search_fn(X_outer_train, y_outer_train, inner_folds)
        best_params_per_fold.append(best_params)

        model = model_fn(X_outer_train, y_outer_train, best_params)
        y_pred = np.asarray(model.predict(X_outer_test)).reshape(-1)
        outer_metrics.append(float(metric_fn(y_outer_test, y_pred)))

    return {
        'mean_metric': float(np.mean(outer_metrics)) if outer_metrics else 0.0,
        'std_metric': float(np.std(outer_metrics)) if outer_metrics else 0.0,
        'outer_metrics': outer_metrics,
        'best_params_per_fold': best_params_per_fold,
    }


def bootstrap_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 100,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Estimate performance uncertainty via bootstrap resampling."""

    features = _to_numpy(X, name="X")
    targets = _validate_targets(y)

    if features.shape[0] != targets.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    rng = np.random.default_rng(random_state)

    n_samples = features.shape[0]
    sample_size = sample_size or n_samples

    metrics: List[float] = []
    models: List[Any] = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=sample_size, replace=True)
        X_bootstrap = features[indices]
        y_bootstrap = targets[indices]

        model = model_fn(X_bootstrap, y_bootstrap)
        models.append(model)

        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False

        if not np.any(oob_mask):
            continue

        X_oob = features[oob_mask]
        y_oob = targets[oob_mask]

        y_pred = np.asarray(model.predict(X_oob)).reshape(-1)
        metrics.append(float(metric_fn(y_oob, y_pred)))

    return {
        'metrics': metrics,
        'mean_metric': float(np.mean(metrics)) if metrics else None,
        'std_metric': float(np.std(metrics)) if metrics else None,
        'n_bootstrap': n_bootstrap,
        'models': models,
    }
