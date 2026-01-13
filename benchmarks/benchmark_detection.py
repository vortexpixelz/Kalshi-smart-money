"""Benchmark script to simulate detector inference workloads."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smart_money_detection.detectors.iqr import IQRDetector
from smart_money_detection.detectors.percentile import PercentileDetector
from smart_money_detection.detectors.volume import RelativeVolumeDetector
from smart_money_detection.detectors.zscore import ZScoreDetector
from smart_money_detection.utils.performance import (
    get_performance_collector,
    get_telemetry_logger,
    reset_performance_collector,
)


def run_benchmark(
    *,
    samples: int,
    features: int,
    iterations: int,
    label: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=20.0, size=(samples, features))
    volumes = rng.lognormal(mean=4.0, sigma=0.8, size=(samples, 1))

    detectors = [
        ZScoreDetector(threshold=2.5, rolling_window=20),
        IQRDetector(multiplier=1.5, rolling_window=20),
        PercentileDetector(percentile=0.95, rolling_window=20),
        RelativeVolumeDetector(threshold_multiplier=3.0),
    ]

    reset_performance_collector()
    telemetry = get_telemetry_logger()

    for detector in detectors:
        detector.fit(data)

    start = time.perf_counter()
    for _ in range(iterations):
        for detector in detectors:
            detector.predict(data)
            detector.predict_rolling(volumes)
    duration = time.perf_counter() - start

    collector = get_performance_collector()
    summary = collector.summary()

    total_predictions = samples * iterations * len(detectors)
    throughput = total_predictions / duration if duration > 0 else 0

    payload = {
        "label": label,
        "duration_s": round(duration, 4),
        "iterations": iterations,
        "samples": samples,
        "features": features,
        "throughput_samples_per_second": round(throughput, 3),
        "metrics": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    telemetry.log_event({"event": "benchmark.detection", **payload})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detector inference.")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--label", type=str, default="baseline")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    results = run_benchmark(
        samples=args.samples,
        features=args.features,
        iterations=args.iterations,
        label=args.label,
    )

    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
