# Smart Money Detection for Prediction Markets

State-of-the-art ensemble anomaly detection system for identifying informed traders ("smart money") in prediction markets with minimal labeled data.

## Features

- **Minimal Labeled Data**: Operates effectively with just 2-10 labeled examples using advanced ensemble weighting
- **Adaptive Weighting**: Thompson Sampling, UCB, and Multiplicative Weights Update (MWU) for dynamic ensemble optimization
- **Smart Money Models**: VPIN (Volume-Synchronized PIN), PIN models adapted for prediction markets
- **Active Learning**: Query-by-Committee and BALD for optimal manual review selection
- **Temporal Context**: Cyclical encoding of time features for context-aware detection
- **Human-in-the-Loop**: Seamless feedback integration with F1 score optimization

## Installation

```bash
# Clone repository
git clone https://github.com/VirtualPixelz/Kalshi-smart-money.git
cd Kalshi-smart-money

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
from smart_money_detection import SmartMoneyDetector
import pandas as pd

# Load trade data
trades = pd.DataFrame({
    'timestamp': [...],
    'volume': [...],
    'price': [...],
})

# Initialize detector
detector = SmartMoneyDetector()

# Fit on historical data
detector.fit(trades, volume_col='volume', timestamp_col='timestamp')

# Detect smart money trades
predictions = detector.predict(trades)
scores = detector.score(trades)

# Suggest trades for manual review (active learning)
indices, suggested = detector.suggest_manual_reviews(trades, n_queries=10)

# Add feedback and update weights
detector.add_feedback(
    sample_ids=[...],
    labels=[1, 0, 1, ...],  # 1 = smart money, 0 = normal
    trades=trades,
    update_weights=True
)

# Get performance statistics
stats = detector.get_feedback_statistics()
print(f"F1 Score: {stats['f1_score']:.3f}")
```

## Architecture

### Core Components

1. **Base Detectors** (`smart_money_detection/detectors/`)
   - Z-Score: Statistical outlier detection
   - IQR: Interquartile range method
   - Percentile: Threshold-based detection
   - Relative Volume: Trade size relative to market

2. **Ensemble Weighting** (`smart_money_detection/ensemble/`)
   - **Uniform**: Baseline equal weighting
   - **MWU**: Multiplicative Weights Update (SEAD algorithm)
   - **Thompson Sampling**: Bayesian bandit with Beta posteriors
   - **UCB**: Upper Confidence Bound with exploration bonus
   - **Contextual UCB**: Context-aware weighting with temporal features

3. **Smart Money Models** (`smart_money_detection/models/`)
   - **VPIN**: Volume-Synchronized Probability of Informed Trading
   - **PIN**: Probability of Informed Trading with MLE
   - **Simplified PIN**: Heuristic-based informed trading detection
   - **Trade Classification**: Bulk volume classifier, tick rule, quote rule

4. **Active Learning** (`smart_money_detection/active_learning/`)
   - **Query-by-Committee**: Maximize ensemble disagreement
   - **BALD**: Bayesian Active Learning by Disagreement
   - **Uncertainty Sampling**: Select near decision boundary
   - **Feedback Manager**: Track reviews and optimize F1 threshold

5. **Temporal Features** (`smart_money_detection/features/`)
   - Cyclical encoding: `sin/cos` for hour, day, week
   - Time-to-resolution features
   - Transformer-style positional encoding

## Research Foundation

This implementation is based on cutting-edge research from 2020-2025:

- **SEAD** (Amazon, ICML 2025): Unsupervised ensemble weighting using MWU
- **VPIN** (Easley et al., 2012): Predicted Flash Crash with 0.40 correlation
- **TEUCB** (2024): Tree Ensemble UCB achieving 20-35% lower regret
- **SPADE** (Google, TMLR 2024): Semi-supervised anomaly detection
- **IRT** (2022): Item Response Theory for detector quality estimation

### Key Insights

✅ **Start with uniform weights** when you have <10 labeled examples
✅ **Use Thompson Sampling** for optimal exploration-exploitation
✅ **Apply active learning** (QBC) to reduce manual reviews by 50-80%
✅ **Encode time cyclically** to preserve periodicity
✅ **Optimize F1 threshold** as feedback accumulates
✅ **Transition to Bayesian optimization** after 10-50 labels

## Configuration

Customize detection parameters via the configuration loader:

```python
from smart_money_detection.config import load_config

config = load_config()

# Ensemble weighting
config.ensemble.weighting_method = 'thompson'  # 'uniform', 'mwu', 'ucb'
config.ensemble.thompson_alpha_prior = 1.0
config.ensemble.thompson_beta_prior = 1.0

# Smart money thresholds
config.smart_money.vpin_threshold = 0.75
config.smart_money.major_market_dollar_threshold = 10000
config.smart_money.niche_market_dollar_threshold = 1000

# Active learning
config.active_learning.query_strategy = 'qbc'  # 'bald', 'uncertainty'
config.active_learning.batch_size = 10
config.active_learning.optimize_f1 = True

detector = SmartMoneyDetector(config)
```

The loader automatically merges values from `config/*.yaml`, environment variables
(`SMART_MONEY_DETECTION__SECTION__FIELD`), and optional CLI overrides.

## Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

Demonstrates:
- Fitting detector on synthetic data
- Making predictions
- Active learning workflow
- Feedback integration and weight updates

### Kalshi Integration

```bash
python examples/kalshi_integration.py
```

Shows:
- API integration pattern
- Market-specific threshold configuration
- VPIN computation for order flow toxicity
- Real-time detection and manual review

## Validation with Minimal Data

When you have very few labeled examples:

### <10 Labels: Leave-One-Out CV

```python
from smart_money_detection.utils import loocv

results = loocv(X, y, model_fn=..., metric_fn=...)
print(f"LOOCV F1: {results['mean_metric']:.3f} ± {results['std_metric']:.3f}")
```

### 10-50 Labels: Repeated Stratified K-Fold

```python
from smart_money_detection.utils import cross_validate

results = cross_validate(X, y, n_folds=5, n_repeats=10)
```

### Weight Optimization: Nested CV

```python
from smart_money_detection.utils import nested_cv

results = nested_cv(
    X, y,
    model_fn=...,
    param_search_fn=...,
    outer_folds=5,
    inner_folds=3
)
```

## Performance Optimization

### Cold Start (0-2 Labels)

```python
# Use uniform weights + ensemble diversity
detector = SmartMoneyDetector()
detector.config.ensemble.weighting_method = 'uniform'

# Focus on maximizing detector diversity
contributions = detector.get_detector_contributions(trades)
```

### Early Stage (3-10 Labels)

```python
# Thompson Sampling for exploration-exploitation
detector.config.ensemble.weighting_method = 'thompson'

# Aggressive active learning
indices, suggested = detector.suggest_manual_reviews(trades, n_queries=20)
```

### Mature Stage (>50 Labels)

```python
# Bayesian optimization for meta-weights
from smart_money_detection.utils import bayesian_optimize_weights

optimal_weights, best_f1 = bayesian_optimize_weights(
    detector_scores, labels, n_iterations=30
)
```

## Kalshi-Specific Adaptations

### Market Size Thresholds

```python
# Major markets (presidential elections, Fed decisions)
config.smart_money.major_market_dollar_threshold = 10000
config.smart_money.major_market_oi_percent = 0.01  # 1% of open interest

# Niche markets
config.smart_money.niche_market_dollar_threshold = 1000
config.smart_money.niche_market_oi_percent = 0.05  # 5% of open interest
```

### Temporal Features

```python
# Time to market resolution
config.detector.use_time_to_resolution = True

# Time of day patterns (late night = more suspicious?)
config.ensemble.use_temporal_context = True
```

### VPIN for Order Flow

```python
from smart_money_detection.models import VPINClassifier

vpin = VPINClassifier(
    threshold=0.75,
    n_buckets=50,
    bucket_pct_of_daily=0.02  # 2% of daily volume per bucket
)

vpin.fit(prices, volumes)
toxicity_scores = vpin.score(new_prices, new_volumes)

# VPIN > 0.75 indicates elevated informed trading
high_toxicity = toxicity_scores > 0.75
```

## API Reference

### SmartMoneyDetector

Main detection pipeline coordinating all components.

**Methods:**
- `fit(trades, volume_col, timestamp_col, price_col)`: Fit detector
- `predict(trades, ...)`: Binary predictions (0=normal, 1=smart money)
- `score(trades, ...)`: Anomaly scores in [0, 1]
- `suggest_manual_reviews(trades, n_queries)`: Active learning query selection
- `add_feedback(sample_ids, labels, ...)`: Update with human feedback
- `evaluate(trades, labels, ...)`: Compute performance metrics
- `save_state(filepath)` / `load_state(filepath)`: Persistence

### Ensemble Weighting

**UniformWeighting**: Equal weights (baseline)

**MultiplicativeWeightsUpdate**:
- Learning rate η ∈ [0.1, 0.5]
- Regret bound: O(√(T log k))

**ThompsonSamplingWeighting**:
- Beta posteriors: Beta(α, β)
- Optimal regret: O(√T)

**UCBWeighting**:
- Exploration param c ∈ [0.5, 2.0]
- UCB = μ̂ + c√(ln(t)/n)

### Active Learning

**QueryByCommittee**:
- Vote entropy: H(y|x) = -Σ(V(y_i)/C)log(V(y_i)/C)
- Reduces manual reviews by 50-80%

**BALD**:
- Mutual information: I(y; θ | x, D)
- Captures epistemic uncertainty

## Metrics and Evaluation

```python
from smart_money_detection.utils import compute_metrics

metrics = compute_metrics(y_true, y_pred, y_scores)

# Available metrics:
# - accuracy, precision, recall, f1_score
# - specificity, sensitivity
# - fpr (false positive rate), fnr (false negative rate)
# - roc_auc, avg_precision (if scores provided)
# - tp, fp, tn, fn (confusion matrix)
```

## Citation

If you use this system in your research or trading, please cite:

```bibtex
@software{smart_money_detection_2024,
  title={Smart Money Detection: Ensemble Anomaly Detection for Prediction Markets},
  author={VirtualPixelz},
  year={2024},
  url={https://github.com/VirtualPixelz/Kalshi-smart-money}
}
```

### Key Research Papers

- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow toxicity and liquidity in a high-frequency world." *Review of Financial Studies*, 25(5), 1457-1493.
- Easley, D., Kiefer, N. M., O'Hara, M., & Paperman, J. B. (1996). "Liquidity, information, and infrequently traded stocks." *Journal of Finance*, 51(4), 1405-1436.
- Zhao, Y., et al. (2024). "SEAD: Streaming Ensemble of Anomaly Detectors." *ICML 2025*.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

- **Issues**: https://github.com/VirtualPixelz/Kalshi-smart-money/issues
- **Discussions**: https://github.com/VirtualPixelz/Kalshi-smart-money/discussions

## Roadmap

- [x] Base anomaly detectors
- [x] Ensemble weighting methods
- [x] VPIN and PIN models
- [x] Active learning
- [x] Temporal feature encoding
- [ ] Real-time streaming detection
- [ ] Multi-market correlation analysis
- [ ] Wallet/account tracking on blockchain markets
- [ ] Advanced meta-learning (MAML, Prototypical Networks)
- [ ] Drift detection and model updates
- [ ] API integrations (Polymarket, Manifold, PredictIt)
- [ ] Web dashboard for monitoring

## Acknowledgments

Research-based implementation drawing from:
- Market microstructure literature (Easley, O'Hara, Kyle)
- Machine learning conferences (ICML, NeurIPS, KDD 2020-2025)
- Fraud detection and financial anomaly systems
- Contextual bandits and online learning theory

Built with: NumPy, pandas, scikit-learn, SciPy, PyOD, bayesian-optimization
