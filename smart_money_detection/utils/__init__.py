"""
Utility functions for smart money detection
"""

from .metrics import compute_metrics, compute_f1_score, compute_precision_recall, find_optimal_threshold
from .pandas_utils import coerce_trade_dataframe, trades_from_records
from .validation import cross_validate, loocv, nested_cv, bootstrap_validation
from .optimization import (
    bayesian_optimize_weights,
    gradient_optimize_weights,
    grid_search_weights,
)
from .performance import (
    PerformanceCollector,
    PerformanceMetric,
    TelemetryLogger,
    get_performance_collector,
    get_telemetry_logger,
    reset_performance_collector,
    track_performance,
)

__all__ = [
    "compute_metrics",
    "compute_f1_score",
    "compute_precision_recall",
    "find_optimal_threshold",
    "cross_validate",
    "loocv",
    "nested_cv",
    "bootstrap_validation",
    "bayesian_optimize_weights",
    "gradient_optimize_weights",
    "grid_search_weights",
    "coerce_trade_dataframe",
    "trades_from_records",
]
