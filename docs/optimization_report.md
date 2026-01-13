# Smart Money Detection Optimization Report

## Pipeline Hotspots
- **`smart_money_detection/pipeline.py::SmartMoneyDetector.score`** repeatedly constructs temporal context features and re-normalizes detector scores. Context generation performs timestamp parsing plus trigonometric feature expansion that scales with every scoring call, even when the underlying timestamps repeat.
- **`SmartMoneyDetector.suggest_manual_reviews`** previously walked each detector twice (once for predictions, once for scores). This doubled detector compute and repeated numpy conversions for the same volumes.
- **`smart_money_detection/ensemble/ensemble.py::AnomalyEnsemble.score`** normalized each detector output after per-detector data conversion. The redundant DataFrame-to-numpy conversions and repeated min/max scans add up for large batches.
- **Temporal feature generation** in `smart_money_detection/features/temporal.py` still allocates multiple intermediate Series/arrays per call. When scoring and manual-review suggestion run in quick succession, the encoder recomputes identical context features.

## Implemented Optimizations
- Introduced a typed `BaseDetector` with centralised logging, error propagation, and shared numpy conversion utilities. Detectors (`zscore.py`, `iqr.py`, `percentile.py`, `volume.py`) now operate on validated numpy inputs with consistent prediction pathways.
- Refactored `AnomalyEnsemble` to convert the input matrix once per call, reuse normalized score matrices for contributions, and expose `score_with_components` plus calibration hooks to attach post-processing of ensemble probabilities.
- Simplified committee scoring via single-pass `predict_with_scores` and cached normalized score matrices inside `SmartMoneyDetector.score`, eliminating redundant detector passes for manual review suggestions and weight optimization.
- Hardened `KalshiClient`/`AsyncKalshiClient` with retry/backoff, typed exceptions, and DataFrame hygiene via the new `utils.pandas_utils` helpers (vectorized `assign`, categorical conversion) to reduce repeated downstream cleanup.

## Residual Technical Debt
- Temporal encoding still rebuilds feature matrices on every request. Caching encoded timestamps or exposing incremental update APIs would reduce overhead for streaming ticks or backtesting.
- Ensemble calibration expects external calibrators to provide a `transform` method. Shipping a default calibrator (e.g., isotonic regression or Platt scaling) would simplify adoption and make probability semantics uniform by default.
- Detector score caching depends on trade indices; duplicated or unstable indices can lead to cache misses. Introducing explicit sample IDs in the data model would make feedback-driven optimization more robust.
