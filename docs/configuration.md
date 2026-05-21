# Configuration

The smart money detection system uses a layered configuration loader that
combines defaults, YAML, environment variables, and optional CLI overrides.
The resulting configuration is validated and returned as a fully typed
`Config` instance.

## Precedence

Configuration values are merged in order. Later sources override earlier ones:

1. **Defaults** from `smart_money_detection.config.Config`
2. **YAML** files in `config/*.yml` / `config/*.yaml` (merged in lexical order)
3. **Environment variables** prefixed with `SMART_MONEY_DETECTION__`
4. **CLI overrides** passed to `load_config`

The loader reads `.env` at the repo root before evaluating environment variables.

## YAML configuration

Create a file such as `config/local.yaml`:

```yaml
log_level: INFO
ensemble:
  weighting_method: thompson
smart_money:
  vpin_threshold: 0.75
active_learning:
  batch_size: 10
kalshi:
  enabled: true
  api_key: ${KALSHI_API_KEY}
```

## Environment variable overrides

Use double-underscore to define nested keys:

```bash
export SMART_MONEY_DETECTION__SMART_MONEY__VPIN_THRESHOLD=0.8
export SMART_MONEY_DETECTION__ENSEMBLE__WEIGHTING_METHOD=uniform
export SMART_MONEY_DETECTION__KALSHI__ENABLED=true
export SMART_MONEY_DETECTION__KALSHI__API_KEY=your-kalshi-key
```

## CLI overrides

Pass dotted keys in `cli_overrides`:

```python
from smart_money_detection.config import load_config

config = load_config(
    cli_overrides={
        "ensemble.weighting_method": "mwu",
        "smart_money.vpin_threshold": 0.7,
    }
)
```

## Validation

The loader validates numeric ranges (for example, VPIN thresholds, percentile
bounds, and rolling windows) and enforces required credentials. If
`kalshi.enabled` is `True`, `kalshi.api_key` and `kalshi.api_base` must be set.

If validation fails, `load_config` raises a `ValueError` with a list of issues.
