"""Optimization utilities for combining detector outputs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

try:  # pragma: no cover - optional dependency
    from bayes_opt import BayesianOptimization

    BAYESOPT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BAYESOPT_AVAILABLE = False


def bayesian_optimize_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_iterations: int = 20,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """Perform Bayesian optimisation to learn detector weights."""

    if not BAYESOPT_AVAILABLE:
        raise ImportError("bayesian-optimization required. Install with: pip install bayesian-optimization")

    scores = np.asarray(detector_scores, dtype=float)
    labels = np.asarray(y_true).reshape(-1)

    if scores.shape[0] != labels.shape[0]:
        raise ValueError("detector_scores and y_true must contain the same number of samples")

    n_detectors = scores.shape[1]

    if metric_fn is None:
        from sklearn.metrics import f1_score

        metric_fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)

    def objective(**weights_dict: float) -> float:
        weights = np.array([weights_dict[f'w{i}'] for i in range(n_detectors)], dtype=float)
        weights = weights / weights.sum()
        ensemble_scores = np.dot(scores, weights)
        y_pred = (ensemble_scores > 0.5).astype(int)
        return float(metric_fn(labels, y_pred))

    pbounds = {f'w{i}': (0.0, 1.0) for i in range(n_detectors)}

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=0,
    )

    optimizer.maximize(init_points=5, n_iter=n_iterations)

    best_params = optimizer.max['params']
    optimal_weights = np.array([best_params[f'w{i}'] for i in range(n_detectors)], dtype=float)
    optimal_weights = optimal_weights / optimal_weights.sum()
    best_metric = float(optimizer.max['target'])

    return optimal_weights, best_metric


def gradient_optimize_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    initial_weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float]:
    """Optimise detector weights using constrained gradient descent."""

    scores = np.asarray(detector_scores, dtype=float)
    labels = np.asarray(y_true, dtype=float).reshape(-1)

    if scores.shape[0] != labels.shape[0]:
        raise ValueError("detector_scores and y_true must contain the same number of samples")

    _, n_detectors = scores.shape

    if initial_weights is None:
        initial_weights = np.ones(n_detectors) / n_detectors

    def loss_fn(weights: np.ndarray) -> float:
        ensemble_scores = np.dot(scores, weights)
        return float(np.mean((labels - ensemble_scores) ** 2))

    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_detectors)]

    result = minimize(
        loss_fn,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter},
    )

    return result.x, float(result.fun)


def grid_search_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_grid_points: int = 5,
) -> Tuple[np.ndarray, float]:
    """Brute-force search for detector weights (only practical for <= 3 detectors)."""

    scores = np.asarray(detector_scores, dtype=float)
    labels = np.asarray(y_true).reshape(-1)

    if scores.shape[0] != labels.shape[0]:
        raise ValueError("detector_scores and y_true must contain the same number of samples")

    n_detectors = scores.shape[1]

    if n_detectors > 3:
        raise ValueError("Grid search only practical for <= 3 detectors")

    grid_values = np.linspace(0, 1, n_grid_points)
    best_metric = -np.inf
    best_weights: Optional[np.ndarray] = None

    def enumerate_weights(depth: int, current_weights: List[float], remaining_mass: float) -> None:
        nonlocal best_metric, best_weights

        if depth == n_detectors - 1:
            weights = np.array(current_weights + [remaining_mass], dtype=float)
            ensemble_scores = np.dot(scores, weights)
            y_pred = (ensemble_scores > 0.5).astype(int)
            metric_value = metric_fn(labels, y_pred)

            if metric_value > best_metric:
                best_metric = metric_value
                best_weights = weights.copy()
            return

        for value in grid_values:
            if value > remaining_mass:
                continue
            enumerate_weights(depth + 1, current_weights + [float(value)], remaining_mass - value)

    enumerate_weights(0, [], 1.0)

    if best_weights is None:
        raise RuntimeError("Failed to identify optimal weights during grid search")

    return best_weights, float(best_metric)
