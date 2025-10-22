"""
Validation utilities for models with minimal labeled data
"""
import numpy as np
from typing import Callable, Dict, Any, List, Optional
from sklearn.model_selection import KFold, LeaveOneOut


def loocv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    metric_fn: Callable,
) -> Dict[str, Any]:
    """
    Leave-One-Out Cross-Validation

    Recommended when n < 50 samples. Provides nearly unbiased performance
    estimate but high variance.

    Args:
        X: Feature data
        y: Labels
        model_fn: Function that takes (X_train, y_train) and returns fitted model
        metric_fn: Function that takes (y_true, y_pred) and returns metric value

    Returns:
        Dictionary with LOOCV results
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    metrics = []

    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = model_fn(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        predictions[test_idx] = y_pred

        # Compute metric
        metric_value = metric_fn(y_test, y_pred)
        metrics.append(metric_value)

    return {
        'predictions': predictions,
        'mean_metric': np.mean(metrics),
        'std_metric': np.std(metrics),
        'all_metrics': metrics,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    metric_fn: Callable,
    n_folds: int = 5,
    n_repeats: int = 1,
    stratified: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    K-Fold Cross-Validation with optional repetitions

    Repeated stratified k-fold CV recommended for n in [50, 200].

    Args:
        X: Feature data
        y: Labels
        model_fn: Function that takes (X_train, y_train) and returns fitted model
        metric_fn: Function that takes (y_true, y_pred) and returns metric value
        n_folds: Number of folds
        n_repeats: Number of repetitions
        stratified: If True, maintain class balance in folds
        random_state: Random seed

    Returns:
        Dictionary with CV results
    """
    all_metrics = []
    all_predictions = []

    for repeat in range(n_repeats):
        seed = random_state + repeat

        if stratified:
            from sklearn.model_selection import StratifiedKFold

            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_predictions = np.zeros(len(X))

        for train_idx, test_idx in kfold.split(X, y if stratified else None):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model = model_fn(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            fold_predictions[test_idx] = y_pred

            # Compute metric
            metric_value = metric_fn(y_test, y_pred)
            all_metrics.append(metric_value)

        all_predictions.append(fold_predictions)

    return {
        'predictions': all_predictions,
        'mean_metric': np.mean(all_metrics),
        'std_metric': np.std(all_metrics),
        'all_metrics': all_metrics,
        'n_folds': n_folds,
        'n_repeats': n_repeats,
    }


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    param_search_fn: Callable,
    metric_fn: Callable,
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Nested Cross-Validation for hyperparameter tuning with limited data

    Essential when optimizing weights: outer loop estimates performance,
    inner loop tunes weights, prevents optimistic bias.

    Args:
        X: Feature data
        y: Labels
        model_fn: Function that takes (X_train, y_train, params) and returns model
        param_search_fn: Function that takes (X_train, y_train) and returns best params
        metric_fn: Function that takes (y_true, y_pred) and returns metric value
        outer_folds: Number of outer CV folds
        inner_folds: Number of inner CV folds
        random_state: Random seed

    Returns:
        Dictionary with nested CV results
    """
    outer_kfold = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    outer_metrics = []
    best_params_per_fold = []

    for outer_train_idx, outer_test_idx in outer_kfold.split(X):
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        # Inner loop: find best parameters
        best_params = param_search_fn(X_outer_train, y_outer_train, inner_folds)
        best_params_per_fold.append(best_params)

        # Outer loop: evaluate with best parameters
        model = model_fn(X_outer_train, y_outer_train, best_params)
        y_pred = model.predict(X_outer_test)

        metric_value = metric_fn(y_outer_test, y_pred)
        outer_metrics.append(metric_value)

    return {
        'mean_metric': np.mean(outer_metrics),
        'std_metric': np.std(outer_metrics),
        'outer_metrics': outer_metrics,
        'best_params_per_fold': best_params_per_fold,
    }


def bootstrap_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    metric_fn: Callable,
    n_bootstrap: int = 100,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap validation for uncertainty estimation

    Args:
        X: Feature data
        y: Labels
        model_fn: Function that takes (X_train, y_train) and returns fitted model
        metric_fn: Function that takes (y_true, y_pred) and returns metric value
        n_bootstrap: Number of bootstrap samples
        sample_size: Size of bootstrap samples (default: same as original)
        random_state: Random seed

    Returns:
        Dictionary with bootstrap results
    """
    np.random.seed(random_state)

    n_samples = len(X)
    if sample_size is None:
        sample_size = n_samples

    metrics = []

    for i in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)

        if len(oob_indices) == 0:
            continue

        X_train, X_test = X[indices], X[oob_indices]
        y_train, y_test = y[indices], y[oob_indices]

        # Train and evaluate
        model = model_fn(X_train, y_train)
        y_pred = model.predict(X_test)

        metric_value = metric_fn(y_test, y_pred)
        metrics.append(metric_value)

    return {
        'mean_metric': np.mean(metrics),
        'std_metric': np.std(metrics),
        'confidence_interval_95': np.percentile(metrics, [2.5, 97.5]),
        'all_metrics': metrics,
    }
