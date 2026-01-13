"""Performance instrumentation utilities."""
from __future__ import annotations

import functools
import inspect
import json
import logging
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class PerformanceMetric:
    """Single performance measurement record."""

    label: str
    duration_ms: float
    memory_kb: Optional[float]
    timestamp: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceCollector:
    """Collects performance metrics in-memory for downstream reporting."""

    def __init__(self) -> None:
        self._records: List[PerformanceMetric] = []

    @property
    def records(self) -> List[PerformanceMetric]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()

    def record(self, metric: PerformanceMetric) -> None:
        self._records.append(metric)

    def summary(self) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        by_label: Dict[str, List[PerformanceMetric]] = {}
        for record in self._records:
            by_label.setdefault(record.label, []).append(record)

        for label, entries in by_label.items():
            durations = [entry.duration_ms for entry in entries]
            memory_values = [
                entry.memory_kb for entry in entries if entry.memory_kb is not None
            ]
            summaries[label] = {
                "count": float(len(entries)),
                "avg_ms": float(statistics.mean(durations)) if durations else 0.0,
                "p95_ms": float(_percentile(durations, 95)) if durations else 0.0,
                "max_ms": float(max(durations)) if durations else 0.0,
                "avg_mem_kb": float(statistics.mean(memory_values))
                if memory_values
                else 0.0,
            }
        return summaries


_COLLECTOR = PerformanceCollector()


def get_performance_collector() -> PerformanceCollector:
    return _COLLECTOR


def reset_performance_collector() -> None:
    _COLLECTOR.reset()


def _percentile(values: Iterable[float], percentile: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * percentile / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


class TelemetryLogger:
    """Emit structured JSON telemetry logs."""

    def __init__(self, logger_name: str = "smart_money_detection.telemetry") -> None:
        self._logger = logging.getLogger(logger_name)

    def log_event(self, event: Dict[str, Any]) -> None:
        self._logger.info(json.dumps(event, default=str))


_TELEMETRY_LOGGER = TelemetryLogger()


def get_telemetry_logger() -> TelemetryLogger:
    return _TELEMETRY_LOGGER


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _collect_memory_start() -> tuple[bool, int]:
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    current, _ = tracemalloc.get_traced_memory()
    return tracing, current


def _collect_memory_end(start_current: int, was_tracing: bool) -> float:
    _, peak = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()
    delta = max(peak - start_current, 0)
    return float(delta) / 1024


def track_performance(
    label: Optional[str] = None,
    *,
    include_memory: bool = True,
    telemetry_logger: Optional[TelemetryLogger] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to record timing/memory metrics and emit telemetry events."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_label = label or f"{func.__module__}.{func.__qualname__}"
        telemetry = telemetry_logger or get_telemetry_logger()
        meta = metadata or {}

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                was_tracing = False
                start_memory = 0
                if include_memory:
                    was_tracing, start_memory = _collect_memory_start()
                start_time = time.perf_counter()
                success = True
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - telemetry only
                    success = False
                    _record_metric(
                        func_label,
                        start_time,
                        include_memory,
                        was_tracing,
                        start_memory,
                        success,
                        meta,
                        error=str(exc),
                    )
                    raise
                _record_metric(
                    func_label,
                    start_time,
                    include_memory,
                    was_tracing,
                    start_memory,
                    success,
                    meta,
                )
                return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            was_tracing = False
            start_memory = 0
            if include_memory:
                was_tracing, start_memory = _collect_memory_start()
            start_time = time.perf_counter()
            success = True
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - telemetry only
                success = False
                _record_metric(
                    func_label,
                    start_time,
                    include_memory,
                    was_tracing,
                    start_memory,
                    success,
                    meta,
                    error=str(exc),
                )
                raise
            _record_metric(
                func_label,
                start_time,
                include_memory,
                was_tracing,
                start_memory,
                success,
                meta,
            )
            return result

        return wrapper

    return decorator


def _record_metric(
    label: str,
    start_time: float,
    include_memory: bool,
    was_tracing: bool,
    start_memory: int,
    success: bool,
    metadata: Dict[str, Any],
    *,
    error: Optional[str] = None,
) -> None:
    duration_ms = (time.perf_counter() - start_time) * 1000
    memory_kb = None
    if include_memory:
        memory_kb = _collect_memory_end(start_memory, was_tracing)

    metric = PerformanceMetric(
        label=label,
        duration_ms=duration_ms,
        memory_kb=memory_kb,
        timestamp=_now_iso(),
        success=success,
        metadata=metadata,
    )
    _COLLECTOR.record(metric)

    payload = {
        "event": "performance",
        "label": label,
        "duration_ms": round(duration_ms, 3),
        "memory_kb": round(memory_kb, 3) if memory_kb is not None else None,
        "timestamp": metric.timestamp,
        "success": success,
    }
    if metadata:
        payload["metadata"] = metadata
    if error:
        payload["error"] = error

    get_telemetry_logger().log_event(payload)
