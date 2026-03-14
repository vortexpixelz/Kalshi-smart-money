# Configuration Guide

`smart_money_detection` uses a configuration loader that merges multiple sources so you
can tune detection behavior without editing code.

## Precedence

From lowest to highest priority:

1. Library defaults defined in `smart_money_detection.config.Config`
2. YAML file passed to `load_config(yaml_path=...)`
3. Environment variables (including values loaded from `.env`)
4. Programmatic overrides passed directly to `load_config(..., overrides=...)`

Later sources overwrite earlier ones.

## Loading configuration

```python
from smart_money_detection.config import load_config

config = load_config(yaml_path="config.yaml")
```

Pass `env_path=None` to skip loading a `.env` file, or use `overrides` to supply
one-off changes in code.

## Environment variables

Environment variables with the prefix `SMART_MONEY_` map to configuration fields using
`__` to separate nested sections:

- `SMART_MONEY_DETECTOR__ZSCORE_THRESHOLD=2.5`
- `SMART_MONEY_ENSEMBLE__WEIGHTING_METHOD=uniform`
- `SMART_MONEY_ACTIVE_LEARNING__BATCH_SIZE=5`

Dedicated Kalshi credentials can also be supplied without the prefix:

- `KALSHI_API_KEY`
- `KALSHI_EMAIL` and `KALSHI_PASSWORD`
- `KALSHI_API_BASE`
- `KALSHI_DEMO_MODE` (`true`/`false`)

By default `demo_mode` is `true`; set it to `false` and provide credentials to connect
to the real Kalshi API.

## YAML example

```yaml
# config.yaml
detector:
  zscore_threshold: 2.5
  percentile_threshold: 97.5
ensemble:
  weighting_method: thompson
kalshi:
  demo_mode: false
```

## Validation

The loader enforces numeric ranges (e.g., thresholds > 0, percentages between 0 and 1)
and ensures Kalshi credentials are present when `demo_mode` is disabled. Invalid
configurations raise `ValueError` with a helpful message.
