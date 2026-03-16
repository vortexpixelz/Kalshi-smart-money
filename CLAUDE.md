# CLAUDE.md — AI Assistant Guide for Kalshi Smart Money Detection

This document describes the codebase structure, development workflows, and conventions for AI assistants (Claude and others) working on this project.

---

## Project Overview

**Kalshi-smart-money** is a Python library for detecting "smart money" — informed traders — in prediction markets using ensemble anomaly detection with adaptive weighting. It is designed to work with minimal labeled data (0–100 examples) and integrates with the Kalshi prediction market API.

**Core Research Foundations:**
- SEAD (ICML 2025): Unsupervised ensemble weighting with O(√T) regret
- VPIN (Easley et al., 2012): Volume-Synchronized Probability of Informed Trading
- TEUCB (2024): Tree Ensemble UCB with lower regret bounds
- SPADE (Google, TMLR 2024): Semi-supervised anomaly detection

---

## Repository Structure

```
Kalshi-smart-money/
├── smart_money_detection/        # Main Python package
│   ├── __init__.py               # Lazy-loading package exports
│   ├── config.py                 # Configuration dataclasses and loading logic
│   ├── pipeline.py               # SmartMoneyDetector — main orchestrator (683 lines)
│   ├── kalshi_client.py          # Kalshi REST API client with retry/backoff
│   ├── detectors/                # Individual anomaly detectors
│   │   ├── base.py               # DetectorProtocol and BaseDetector ABC
│   │   ├── zscore.py             # Z-Score statistical outlier detection
│   │   ├── iqr.py                # Interquartile Range detector
│   │   ├── percentile.py         # Percentile threshold detector
│   │   └── volume.py             # Relative volume anomaly detector
│   ├── ensemble/                 # Ensemble coordination and adaptive weighting
│   │   ├── base.py               # EnsembleProtocol interface
│   │   ├── ensemble.py           # AnomalyEnsemble class
│   │   └── weighting.py          # Weighting strategies (Thompson, UCB, MWU, etc.)
│   ├── models/                   # Smart money models
│   │   ├── vpin.py               # VPIN: order flow toxicity score [0,1]
│   │   ├── pin.py                # PIN: Probability of Informed Trading
│   │   └── trade_classifier.py   # BulkVolumeClassifier, TickRuleClassifier
│   ├── active_learning/          # Human-in-the-loop active learning
│   │   ├── feedback.py           # FeedbackManager — tracks labels, optimizes F1
│   │   └── query_strategies.py   # QueryByCommittee, BALD, UncertaintySampling
│   ├── features/
│   │   └── temporal.py           # Cyclical time encoding (hour/day/week)
│   ├── services/                 # Service layer
│   │   ├── data_ingestion.py     # DataIngestionService — trade data extraction
│   │   └── detection.py          # DetectionService — ensemble coordination
│   └── utils/
│       ├── metrics.py            # F1, precision, recall, ROC-AUC utilities
│       ├── optimization.py       # Bayesian, gradient, grid search weight optimization
│       ├── validation.py         # Cross-validation (LOOCV, K-Fold, bootstrap)
│       ├── pandas_utils.py       # DataFrame coercion helpers
│       └── performance.py        # Performance telemetry
├── tests/                        # Pytest test suite
│   ├── conftest.py               # Shared fixtures (mock/live client, sandbox flags)
│   ├── data/
│   │   └── sandbox_trades_snapshot.json  # Static mock trade data
│   ├── detection/
│   ├── active_learning/
│   ├── ensemble/
│   └── test_*.py                 # Top-level test modules
├── examples/                     # Usage examples
├── docs/                         # Supplementary documentation
├── benchmarks/                   # Performance benchmarks
├── config/
│   └── default.yaml              # Minimal default config (log_level: INFO)
├── Dockerfile                    # Container build
├── docker-compose.yml            # Docker Compose for deployment
├── requirements.txt              # Python dependencies (40 packages)
├── setup.py                      # Package installation
└── pytest.ini                    # Pytest configuration
```

---

## Core Architecture

### Primary Class: `SmartMoneyDetector`

**Location:** `smart_money_detection/pipeline.py`

The main orchestrator. All public API flows through this class.

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `fit(trades_df)` | Train detectors on historical trade data |
| `predict(trades_df)` | Binary predictions — 0=normal, 1=smart money |
| `score(trades_df)` | Anomaly scores [0, 1] |
| `suggest_manual_reviews(trades_df, n)` | Active learning query selection |
| `add_feedback(trade_id, label)` | Human-in-the-loop weight updates |
| `evaluate(trades_df, labels)` | Performance metrics (F1, precision, recall) |
| `save_state(path)` / `load_state(path)` | State persistence |
| `get_ensemble_weights()` | Interpretability — detector weight distribution |
| `optimize_weights(labeled_data)` | Bayesian/gradient weight optimization |
| `get_feedback_statistics()` | Review tracking statistics |

### Weighting Strategies (Adaptive)

Located in `smart_money_detection/ensemble/weighting.py`:

| Strategy | Class | Best For |
|----------|-------|---------|
| `uniform` | `UniformWeighting` | Cold start / baseline |
| `mwu` | `MultiplicativeWeightsUpdate` | Adversarial / fast adaptation |
| `thompson` | `ThompsonSamplingWeighting` | Early labels (3–10) |
| `ucb` | `UCBWeighting` | Upper Confidence Bound exploration |
| `irt` | IRT-based | Item Response Theory quality scoring |

**Lifecycle recommendation:**
- **0 labels:** `uniform`
- **3–10 labels:** `thompson`
- **10–50 labels:** `mwu` or `ucb`
- **50+ labels:** Bayesian optimization via `optimize_weights()`

### Data Flow

```
Kalshi API → DataIngestionService → Feature Engineering → Detectors
                                                               ↓
                                              AnomalyEnsemble (weighted combination)
                                                               ↓
                                              SmartMoneyDetector.predict() / .score()
                                                               ↓
                                              QueryByCommittee → Manual Review
                                                               ↓
                                              FeedbackManager → Weight Update
```

---

## Configuration System

**Location:** `smart_money_detection/config.py`

Configuration uses dataclasses with layered loading:

1. Code defaults (dataclass field defaults)
2. `config/*.yaml` files
3. Environment variables: `SMART_MONEY_DETECTION__<SECTION>__<FIELD>`
4. Runtime overrides: `load_config(overrides={})`

**Key Config Dataclasses:**
- `DetectorConfig` — thresholds for each detector type
- `EnsembleConfig` — `weighting_method`, ensemble settings
- `SmartMoneyConfig` — VPIN buckets, dollar thresholds, trade size percentiles
- `ActiveLearningConfig` — query strategy, batch size, confidence thresholds
- `KalshiConfig` — API credentials and endpoints

**Example environment variable override:**
```bash
export SMART_MONEY_DETECTION__ENSEMBLE__WEIGHTING_METHOD=thompson
export SMART_MONEY_DETECTION__ACTIVE_LEARNING__BATCH_SIZE=20
```

**Simple env vars (from `.env`):**
```
KALSHI_API_KEY=your_api_key_here
KALSHI_API_BASE=https://api.elections.kalshi.com
DETECTION_THRESHOLD=0.75
MIN_VOLUME_THRESHOLD=100
ENABLE_ACTIVE_LEARNING=true
LOG_LEVEL=INFO
```

---

## Development Workflows

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Copy and fill environment variables
cp .env.example .env
```

### Running Tests

```bash
# All unit tests (no external services needed)
pytest

# With coverage report
pytest --cov=smart_money_detection

# Specific test file or marker
pytest tests/detection/test_pipeline.py -v
pytest -m integration -v

# Sandbox integration tests (requires Kalshi sandbox credentials)
pytest --live-sandbox -vv
```

**Pytest markers** (defined in `pytest.ini`):
- `integration` — tests requiring external services or long runtime
- `sandbox` — tests requiring Kalshi sandbox API connectivity
- `regression` — snapshot tests for pipeline output stability

### Running Examples

```bash
python examples/basic_usage.py
python examples/kalshi_integration.py
python test_live.py   # Full integration test
```

### Docker

```bash
docker build -t kalshi-smart-money .
docker-compose up
```

---

## Testing Conventions

**Test fixtures** (`tests/conftest.py`):
- `mocked_kalshi_client` — Uses `tests/data/sandbox_trades_snapshot.json` (no API key needed)
- `live_kalshi_client` — Real Kalshi sandbox connection (requires env vars)
- `kalshi_client` — Auto-selects mock vs live based on `--live-sandbox` flag
- `sandbox_retry` — Retry helper for flaky network tests

**Convention:** All unit tests must pass without any API keys using the mocked client. Use `@pytest.mark.sandbox` only for tests that genuinely need live connectivity.

**Mock data:** `tests/data/sandbox_trades_snapshot.json` contains representative trade snapshots for offline testing. Do not modify this file unless updating the snapshot intentionally.

---

## Key Conventions

### Code Style

- **Python version:** 3.9+ (tested on 3.9, 3.10, 3.11, 3.12 via GitHub Actions)
- **Formatting:** `black` (line length 88)
- **Linting:** `flake8`
- **Type checking:** `mypy`
- **Type hints:** Required on all public methods
- **Docstrings:** Required on all public classes and methods

### Extending Detectors

To add a new detector:
1. Subclass `BaseDetector` from `smart_money_detection/detectors/base.py`
2. Implement `fit(X: pd.DataFrame) -> None` and `score(X: pd.DataFrame) -> np.ndarray`
3. The `DetectorProtocol` uses structural subtyping — also works via duck typing
4. Register in `AnomalyEnsemble` initialization in `ensemble/ensemble.py`

### Extending Weighting Strategies

Implement the `EnsembleProtocol` interface from `smart_money_detection/ensemble/base.py`:
- `update_weights(scores, labels)` — online weight update
- `get_weights()` — return current weight vector

### DataFrame Conventions

Trade data DataFrames are expected to have these columns:
- `trade_id` — unique identifier
- `price` — trade price
- `count` — number of contracts
- `taker_side` — `"yes"` or `"no"` (or `"buy"`/`"sell"`)
- `created_time` — ISO timestamp string or datetime

Use helpers in `smart_money_detection/utils/pandas_utils.py` to coerce and validate DataFrames.

### VPIN Scores

- `0.0 – 0.50`: Normal trading activity
- `0.50 – 0.75`: Elevated informed trading risk
- `0.75+`: High probability of smart money activity

---

## CI/CD

**GitHub Actions:** `.github/workflows/django.yml`

- Triggers on push/PR to main
- Tests against Python 3.9, 3.10, 3.11, 3.12
- Runs `pytest` (unit tests only, no sandbox flag)

**No sandbox credentials are stored in CI.** Integration tests are run locally.

---

## Common Tasks for AI Assistants

### Adding a New Feature

1. Read the relevant existing module before editing
2. Follow type hints and docstring conventions
3. Add or update tests in the corresponding `tests/` subdirectory
4. Run `pytest` to verify nothing is broken
5. Use `black` to format new code

### Debugging Detection Issues

- Check `get_ensemble_weights()` for skewed weight distributions
- Inspect `get_feedback_statistics()` for label imbalance
- Use `score()` instead of `predict()` to see raw anomaly scores
- VPIN scores >0.75 in `models/vpin.py` indicate high-risk windows

### Modifying Configuration

- Prefer environment variable overrides for deployment changes
- For permanent defaults, edit `smart_money_detection/config.py` dataclass defaults
- Do not modify `config/default.yaml` for feature-specific settings — use env vars

### Performance Optimization

- See `docs/optimization_report.md` for existing benchmarks
- Run `benchmarks/benchmark_detection.py` before and after changes
- The Bayesian optimizer in `utils/optimization.py` is the preferred weight optimization method for 10+ labeled samples

---

## Important Files to Know

| File | Why It Matters |
|------|---------------|
| `smart_money_detection/pipeline.py` | Main public API — start here for any feature work |
| `smart_money_detection/config.py` | All configuration — check before adding new settings |
| `smart_money_detection/ensemble/weighting.py` | Weighting algorithms — core research implementation |
| `smart_money_detection/models/vpin.py` | VPIN model — primary smart money signal |
| `smart_money_detection/active_learning/feedback.py` | Feedback loop — modifies ensemble behavior |
| `tests/conftest.py` | Test fixtures — understand before writing new tests |
| `tests/data/sandbox_trades_snapshot.json` | Mock data — used in all offline tests |
| `.env.example` | All supported environment variables |

---

## Out of Scope

- This library does **not** place trades on Kalshi — it only reads market data and detects anomalies
- The Kalshi API client (`kalshi_client.py`) is read-only
- No financial advice is implied by detection outputs
