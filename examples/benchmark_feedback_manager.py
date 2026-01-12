"""Benchmark script comparing individual and batch feedback ingestion."""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Iterable, Tuple

import numpy as np

from smart_money_detection.active_learning.feedback import FeedbackManager


def _generate_feedback(num_samples: int) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    sample_ids = [f"sample-{i}" for i in range(num_samples)]
    y_true = rng.integers(0, 2, size=num_samples)
    y_pred = rng.integers(0, 2, size=num_samples)
    scores = rng.random(size=num_samples)
    return sample_ids, y_true, y_pred, scores


def _time_function(func, repeats: int) -> Iterable[float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return durations


def benchmark(num_samples: int, repeats: int) -> None:
    sample_ids, y_true, y_pred, scores = _generate_feedback(num_samples)

    def run_individual():
        manager = FeedbackManager(optimize_f1=False)
        for sid, label, pred, score in zip(sample_ids, y_true, y_pred, scores):
            manager.add_feedback(sid, int(label), int(pred), float(score))

    def run_batch():
        manager = FeedbackManager(optimize_f1=False)
        manager.add_batch_feedback(sample_ids, y_true, y_pred, scores)

    individual_times = list(_time_function(run_individual, repeats))
    batch_times = list(_time_function(run_batch, repeats))

    print(f"Samples per batch: {num_samples}")
    print(f"Repeats: {repeats}")
    print()
    print("Individual add_feedback timing (seconds):")
    print(f"  min/median/max: {min(individual_times):.6f} / {statistics.median(individual_times):.6f} / {max(individual_times):.6f}")
    print("Batch add_batch_feedback timing (seconds):")
    print(f"  min/median/max: {min(batch_times):.6f} / {statistics.median(batch_times):.6f} / {max(batch_times):.6f}")
    improvement = statistics.median(individual_times) / statistics.median(batch_times)
    print()
    print(f"Median speedup: {improvement:.2f}x faster")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000, help="Number of feedback samples per run")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions for each method")
    args = parser.parse_args()
    benchmark(args.samples, args.repeats)
