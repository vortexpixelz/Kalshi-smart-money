"""Generate performance trend plots from benchmark JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_latency_frame(label: str, payload: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric_label, metrics in payload.get("metrics", {}).items():
        rows.append(
            {
                "run": label,
                "metric": metric_label,
                "avg_ms": metrics.get("avg_ms", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _build_throughput_frame(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run": label,
        "throughput": payload.get("cycles_per_second")
        or payload.get("throughput_samples_per_second")
        or 0,
        "workload": payload.get("workload", payload.get("label", label)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance plots.")
    parser.add_argument("--market-baseline", required=True)
    parser.add_argument("--market-post", required=True)
    parser.add_argument("--detection-baseline", required=True)
    parser.add_argument("--detection-post", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    market_baseline = _load(Path(args.market_baseline))
    market_post = _load(Path(args.market_post))
    detection_baseline = _load(Path(args.detection_baseline))
    detection_post = _load(Path(args.detection_post))

    latency_frames = pd.concat(
        [
            _build_latency_frame("market_baseline", market_baseline),
            _build_latency_frame("market_post", market_post),
            _build_latency_frame("detection_baseline", detection_baseline),
            _build_latency_frame("detection_post", detection_post),
        ],
        ignore_index=True,
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=latency_frames, x="metric", y="avg_ms", hue="run")
    plt.xticks(rotation=45, ha="right")
    plt.title("Average response time by instrumented method")
    plt.ylabel("Average latency (ms)")
    plt.tight_layout()
    latency_path = output_dir / "response_times.png"
    plt.savefig(latency_path)
    plt.close()

    throughput_rows = [
        {
            **_build_throughput_frame("market_baseline", market_baseline),
            "workload": "market_polling",
        },
        {
            **_build_throughput_frame("market_post", market_post),
            "workload": "market_polling",
        },
        {
            **_build_throughput_frame("detection_baseline", detection_baseline),
            "workload": "detection",
        },
        {
            **_build_throughput_frame("detection_post", detection_post),
            "workload": "detection",
        },
    ]
    throughput_frame = pd.DataFrame(throughput_rows)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=throughput_frame, x="workload", y="throughput", hue="run")
    plt.title("Throughput comparison")
    plt.ylabel("Throughput (ops/sec)")
    plt.tight_layout()
    throughput_path = output_dir / "throughput.png"
    plt.savefig(throughput_path)
    plt.close()


if __name__ == "__main__":
    main()
