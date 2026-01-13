# Performance report

## Overview
This report captures baseline measurements for Kalshi market polling and detector inference workloads, along with a post-instrumentation re-run to confirm timing/memory telemetry behavior. The workloads are simulated with mock/historical-style data to avoid external dependencies while still exercising the critical paths.

## Benchmark configuration
- **Market polling**: 15 polling cycles, 8 markets, 80 trades per market, 4 ms simulated API latency.
- **Detection**: 500 samples, 4 features, 40 iterations per detector with rolling windows enabled.

## Market polling results (baseline vs. post)
| Metric | Baseline | Post | Notes |
| --- | --- | --- | --- |
| Cycles/sec | 4.907 | 5.613 | Overall polling throughput. |
| Avg request latency (ms) | 4.615 | 4.346 | Low latency consistent with simulated 4 ms network delay. |
| Avg get_trades latency (ms) | 16.353 | 13.991 | Includes DataFrame formatting overhead. |
| Avg get_market_summary latency (ms) | 24.640 | 21.702 | Includes trade aggregation over last 24h. |

## Detection results (baseline vs. post)
| Metric | Baseline | Post | Notes |
| --- | --- | --- | --- |
| Throughput (samples/sec) | 91,866.384 | 92,814.268 | Aggregate inference throughput across detectors. |
| Avg ZScore predict latency (ms) | 0.263 | 0.268 | Simple vectorized computation. |
| Avg IQR predict latency (ms) | 0.341 | 0.341 | Additional quantile distance checks. |
| Avg Percentile predict latency (ms) | 0.385 | 0.358 | Threshold check against percentile baseline. |
| Avg Volume predict latency (ms) | 0.151 | 0.138 | Relative volume ratio check. |

## Trends
- **Response times**: The response time plot highlights the consistent latency profiles across pre/post runs, confirming the instrumentation overhead is minimal for the simulated workloads (visualized with seaborn).
- **Throughput**: Throughput is effectively stable between runs, with minor variance expected from random data generation and interpreter scheduling.

Plots are generated via `python benchmarks/generate_performance_plots.py` and should be attached to the PR (not committed) due to binary file restrictions.

## Notes
- Metrics are recorded via the new timing/memory decorators and emitted as structured JSON telemetry events.
- Re-run benchmarks after optimization changes to track improvements over time.
