"""
Optimization methods for ensemble weight tuning with minimal labeled data

Implements:
- Bayesian optimization for sample-efficient weight search
- Gradient-based optimization for differentiable objectives
- Convex optimization for ensemble weights
"""
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple
from scipy.optimize import minimize
try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False


def bayesian_optimize_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Optional[Callable] = None,
    n_iterations: int = 20,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Bayesian optimization for ensemble weights

    Sample-efficient optimization when validation set has only 10-20 examples.
    Typically finds near-optimal weights in 15-30 evaluations.

    Args:
        detector_scores: Detector scores of shape (n_samples, n_detectors)
        y_true: True labels
        metric_fn: Metric to maximize (default: F1 score)
        n_iterations: Number of optimization iterations
        random_state: Random seed

    Returns:
        Tuple of (optimal_weights, best_metric_value)
    """
    if not BAYESOPT_AVAILABLE:
        raise ImportError("bayesian-optimization required. Install with: pip install bayesian-optimization")

    n_detectors = detector_scores.shape[1]

    if metric_fn is None:
        # Default: F1 score
        from sklearn.metrics import f1_score
        metric_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0)

    # Define objective function
    def objective(**weights_dict):
        # Extract weights from dictionary
        weights = np.array([weights_dict[f'w{i}'] for i in range(n_detectors)])

        # Normalize weights
        weights = weights / weights.sum()

        # Compute ensemble predictions
        ensemble_scores = np.dot(detector_scores, weights)
        y_pred = (ensemble_scores > 0.5).astype(int)

        # Compute metric
        metric_value = metric_fn(y_true, y_pred)

        return metric_value

    # Define parameter bounds (each weight in [0, 1])
    pbounds = {f'w{i}': (0.0, 1.0) for i in range(n_detectors)}

    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=0,
    )

    # Optimize
    optimizer.maximize(
        init_points=5,  # Random exploration
        n_iter=n_iterations,  # Optimization iterations
    )

    # Extract optimal weights
    best_params = optimizer.max['params']
    optimal_weights = np.array([best_params[f'w{i}'] for i in range(n_detectors)])

    # Normalize
    optimal_weights = optimal_weights / optimal_weights.sum()

    best_metric = optimizer.max['target']

    return optimal_weights, best_metric


def gradient_optimize_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    initial_weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Gradient-based optimization for ensemble weights

    Solves convex optimization problem:
    Minimize: L(w) = (1/n)Σ(y_i - Σw_j·ŷ_ij)²
    Subject to: Σw_j = 1, w_j ≥ 0

    Args:
        detector_scores: Detector scores of shape (n_samples, n_detectors)
        y_true: True labels
        initial_weights: Initial weight guess (default: uniform)
        max_iter: Maximum optimization iterations

    Returns:
        Tuple of (optimal_weights, final_loss)
    """
    n_samples, n_detectors = detector_scores.shape

    if initial_weights is None:
        initial_weights = np.ones(n_detectors) / n_detectors

    # Define loss function (mean squared error)
    def loss_fn(weights):
        ensemble_scores = np.dot(detector_scores, weights)
        mse = np.mean((y_true - ensemble_scores) ** 2)
        return mse

    # Constraints: sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}

    # Bounds: [0, 1] for each weight
    bounds = [(0.0, 1.0) for _ in range(n_detectors)]

    # Optimize
    result = minimize(
        loss_fn,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter},
    )

    optimal_weights = result.x
    final_loss = result.fun

    return optimal_weights, final_loss


def grid_search_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable,
    n_grid_points: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Grid search for ensemble weights (small ensembles only)

    Only practical for 2-3 detectors due to exponential complexity.

    Args:
        detector_scores: Detector scores of shape (n_samples, n_detectors)
        y_true: True labels
        metric_fn: Metric to maximize
        n_grid_points: Number of grid points per dimension

    Returns:
        Tuple of (optimal_weights, best_metric_value)
    """
    n_detectors = detector_scores.shape[1]

    if n_detectors > 3:
        raise ValueError("Grid search only practical for <= 3 detectors")

    # Generate grid
    grid_values = np.linspace(0, 1, n_grid_points)

    best_metric = -np.inf
    best_weights = None

    # Enumerate all weight combinations
    def enumerate_weights(depth, current_weights, remaining_mass):
        nonlocal best_metric, best_weights

        if depth == n_detectors - 1:
            # Last weight is determined by constraint
            weights = np.array(current_weights + [remaining_mass])

            # Compute metric
            ensemble_scores = np.dot(detector_scores, weights)
            y_pred = (ensemble_scores > 0.5).astype(int)
            metric_value = metric_fn(y_true, y_pred)

            if metric_value > best_metric:
                best_metric = metric_value
                best_weights = weights.copy()

        else:
            # Try different values for this weight
            for w in grid_values:
                if w <= remaining_mass:
                    enumerate_weights(
                        depth + 1, current_weights + [w], remaining_mass - w
                    )

    enumerate_weights(0, [], 1.0)

    return best_weights, best_metric


def evolutionary_optimize_weights(
    detector_scores: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable,
    population_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Evolutionary algorithm for ensemble weights

    Gradient-free optimization for non-differentiable objectives like F1 score.
    Handles multi-objective formulations naturally.

    Args:
        detector_scores: Detector scores of shape (n_samples, n_detectors)
        y_true: True labels
        metric_fn: Metric to maximize
        population_size: Number of individuals in population
        n_generations: Number of evolution generations
        mutation_rate: Probability of mutation
        random_state: Random seed

    Returns:
        Tuple of (optimal_weights, best_metric_value)
    """
    np.random.seed(random_state)

    n_detectors = detector_scores.shape[1]

    def evaluate_fitness(weights):
        """Evaluate fitness of weight configuration"""
        # Normalize weights
        weights = weights / weights.sum()

        # Compute predictions
        ensemble_scores = np.dot(detector_scores, weights)
        y_pred = (ensemble_scores > 0.5).astype(int)

        # Compute metric
        return metric_fn(y_true, y_pred)

    # Initialize random population
    population = np.random.rand(population_size, n_detectors)
    population = population / population.sum(axis=1, keepdims=True)

    best_weights = None
    best_fitness = -np.inf

    for generation in range(n_generations):
        # Evaluate fitness
        fitness = np.array([evaluate_fitness(ind) for ind in population])

        # Track best
        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_weights = population[gen_best_idx].copy()

        # Selection (tournament)
        new_population = []
        for _ in range(population_size):
            # Tournament selection
            tournament_indices = np.random.choice(
                population_size, size=3, replace=False
            )
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            new_population.append(population[winner_idx].copy())

        population = np.array(new_population)

        # Crossover
        for i in range(0, population_size - 1, 2):
            if np.random.rand() < 0.7:  # Crossover probability
                # Single-point crossover
                point = np.random.randint(1, n_detectors)
                temp = population[i, :point].copy()
                population[i, :point] = population[i + 1, :point]
                population[i + 1, :point] = temp

        # Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                # Mutate one gene
                gene_idx = np.random.randint(n_detectors)
                population[i, gene_idx] += np.random.randn() * 0.1
                population[i, gene_idx] = np.clip(population[i, gene_idx], 0, 1)

        # Normalize
        population = population / population.sum(axis=1, keepdims=True)

    return best_weights, best_fitness


def tune_detection_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    metric: str = 'f1',
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Tune detection threshold to optimize a metric

    Args:
        scores: Anomaly scores
        y_true: True labels
        metric: Metric to optimize ('f1', 'precision', 'recall', etc.)
        n_thresholds: Number of thresholds to try

    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    metric_fn_map = {
        'f1': lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
    }

    if metric not in metric_fn_map:
        raise ValueError(f"Unknown metric: {metric}")

    metric_fn = metric_fn_map[metric]

    # Try different thresholds
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    best_metric = -np.inf
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metric_value = metric_fn(y_true, y_pred)

        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold

    return best_threshold, best_metric
