# Quick Start Guide

Get started with smart money detection in 5 minutes!

## Installation

```bash
git clone https://github.com/VirtualPixelz/Kalshi-smart-money.git
cd Kalshi-smart-money
pip install -r requirements.txt
pip install -e .
```

## Minimal Example (5 lines)

```python
from smart_money_detection import SmartMoneyDetector
import pandas as pd

# Load your trade data
trades = pd.read_csv('trades.csv')  # Needs: timestamp, volume columns

# Detect smart money
detector = SmartMoneyDetector()
detector.fit(trades, volume_col='volume', timestamp_col='timestamp')
predictions = detector.predict(trades)

# Get detected smart money trades
smart_money = trades[predictions == 1]
print(f"Detected {len(smart_money)} smart money trades")
```

## Step-by-Step Tutorial

### 1. Prepare Your Data

Your trade data should be a pandas DataFrame with at minimum:
- `timestamp`: datetime
- `volume`: float (trade size in dollars)

Optional but recommended:
- `price`: float (for VPIN analysis)
- `trade_id`: unique identifier

```python
import pandas as pd

trades = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
    'volume': [100, 250, 500, ...],  # Your volume data
    'price': [50.5, 50.6, 50.4, ...],  # Optional
    'trade_id': range(1000),
})
```

### 2. Initialize and Fit Detector

```python
from smart_money_detection import SmartMoneyDetector

detector = SmartMoneyDetector()

# Fit on historical data
detector.fit(
    trades,
    volume_col='volume',
    timestamp_col='timestamp',
    price_col='price'  # Optional
)
```

### 3. Detect Smart Money

```python
# Get predictions (0 = normal, 1 = smart money)
predictions = detector.predict(trades)

# Get anomaly scores (0-1, higher = more suspicious)
scores = detector.score(trades)

# Analyze results
smart_money_trades = trades[predictions == 1]
print(f"Found {len(smart_money_trades)} smart money trades")
print(f"Mean score: {scores[predictions == 1].mean():.3f}")
```

### 4. Human-in-the-Loop Learning

The real power comes from active learning with minimal manual reviews:

```python
# Step 1: System suggests which trades to review
indices, suggested = detector.suggest_manual_reviews(trades, n_queries=10)

# Step 2: You manually label these 10 trades
# (in practice, you'd review and label these)
manual_labels = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # Your labels

# Step 3: System learns from your feedback
detector.add_feedback(
    sample_ids=suggested['trade_id'].tolist(),
    labels=manual_labels,
    trades=trades.set_index('trade_id'),
    update_weights=True  # Automatically improve detection
)

# Step 4: Check improved performance
stats = detector.get_feedback_statistics()
print(f"Current F1 Score: {stats.get('f1_score', 'N/A')}")
```

### 5. Configure for Your Market

Adjust thresholds based on market size:

```python
from smart_money_detection.config import load_config

config = load_config()

# For large markets (e.g., presidential elections)
config.smart_money.major_market_dollar_threshold = 10000
config.smart_money.major_market_oi_percent = 0.01

# For small markets
config.smart_money.niche_market_dollar_threshold = 1000
config.smart_money.niche_market_oi_percent = 0.05

# Choose ensemble method
config.ensemble.weighting_method = 'thompson'  # or 'ucb', 'mwu'

# Create detector with config
detector = SmartMoneyDetector(config)
```

## Understanding the Output

### Predictions

```python
predictions = detector.predict(trades)
# Array of 0s and 1s
# 0 = normal trade
# 1 = smart money (informed trade)
```

### Scores

```python
scores = detector.score(trades)
# Array of floats in [0, 1]
# Higher score = more anomalous
# Typical threshold: 0.5 (automatically optimized with feedback)
```

### Detector Weights

```python
weights = detector.get_ensemble_weights()
# {'ZScore': 0.25, 'IQR': 0.30, 'Percentile': 0.20, 'RelativeVolume': 0.25}
# Shows how much each detector contributes
```

## Common Workflows

### Workflow 1: Cold Start (No Labels)

```python
# Use uniform weighting + focus on diversity
detector = SmartMoneyDetector()
detector.config.ensemble.weighting_method = 'uniform'

detector.fit(trades)
predictions = detector.predict(trades)

# Start with active learning
indices, suggested = detector.suggest_manual_reviews(trades, n_queries=20)
```

### Workflow 2: With Some Labels (2-10)

```python
# Thompson Sampling for exploration
detector = SmartMoneyDetector()
detector.config.ensemble.weighting_method = 'thompson'

detector.fit(trades)

# Iterative improvement
for round in range(5):
    indices, suggested = detector.suggest_manual_reviews(trades, n_queries=10)
    labels = get_manual_labels(suggested)  # Your labeling function
    detector.add_feedback(suggested['trade_id'].tolist(), labels, trades, update_weights=True)
```

### Workflow 3: Production (>50 Labels)

```python
# Optimize weights with Bayesian optimization
from smart_money_detection.utils import bayesian_optimize_weights

detector = SmartMoneyDetector()
detector.fit(trades)

# After collecting 50+ labels
_, labels, _ = detector.feedback_manager.get_labeled_data()
# optimal_weights, _ = bayesian_optimize_weights(detector_scores, labels)

# Regular monitoring
predictions = detector.predict(new_trades)
detector.save_state('production_detector.pkl')
```

## VPIN for Order Flow Analysis

```python
from smart_money_detection.models import VPINClassifier

# Detect toxic order flow
vpin = VPINClassifier(threshold=0.75, n_buckets=50)
vpin.fit(trades['price'].values, trades['volume'].values)

vpin_scores = vpin.score(new_trades['price'].values, new_trades['volume'].values)

# High VPIN indicates informed trading
high_toxicity_periods = vpin_scores > 0.75
print(f"High toxicity detected in {high_toxicity_periods.sum()} periods")
```

## Troubleshooting

### Issue: Too many false positives

```python
# Increase threshold or adjust detector sensitivity
detector.config.detector.zscore_threshold = 3.5  # More conservative
detector.config.detector.volume_threshold_multiplier = 4.0
```

### Issue: Missing smart money trades

```python
# Decrease thresholds
detector.config.detector.percentile_threshold = 90.0  # Instead of 95
detector.config.smart_money.large_trade_percentile = 90.0
```

### Issue: Poor ensemble performance

```python
# Check ensemble diversity
diversity = detector.ensemble.get_ensemble_diversity(trades['volume'].values)
print(f"Diversity: {diversity:.3f}")  # Should be > 0.5

# Check detector contributions
contributions = detector.get_detector_contributions(trades)
for name, data in contributions.items():
    print(f"{name}: weight={data['weight']:.3f}")
```

## Next Steps

1. **Run Examples**: `python examples/basic_usage.py`
2. **Read Documentation**: See `README.md` for full API reference
3. **Integrate with Kalshi**: See `examples/kalshi_integration.py`
4. **Optimize Performance**: Use active learning + Bayesian optimization
5. **Monitor & Improve**: Regular feedback loop with manual reviews

## Key Recommendations

✅ Start with uniform weights if you have <10 labels
✅ Use Thompson Sampling for automatic exploration-exploitation
✅ Apply active learning (QBC) to reduce manual work by 50-80%
✅ Accumulate 10-50 labels before optimizing weights
✅ Save detector state regularly for reproducibility
✅ Monitor ensemble diversity (should be >0.5)

## Resources

- Full API: `README.md`
- Examples: `examples/`
- Research: See citations in `README.md`
- Issues: https://github.com/VirtualPixelz/Kalshi-smart-money/issues
