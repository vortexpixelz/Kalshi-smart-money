"""Configuration management for the smart money detection system."""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import yaml
from dotenv import load_dotenv


ENV_PREFIX = "SMART_MONEY_DETECTION"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class DetectorConfig:
    """Configuration for base anomaly detectors"""

    # Z-score detector
    zscore_threshold: float = 3.0
    zscore_rolling_window: int = 100

    # IQR detector
    iqr_multiplier: float = 1.5
    iqr_rolling_window: int = 100

    # Percentile detector
    percentile_threshold: float = 95.0  # 95th percentile
    percentile_rolling_window: int = 100

    # Relative volume detector
    volume_threshold_multiplier: float = 3.0  # 3x median volume
    volume_rolling_window: int = 100


@dataclass
class EnsembleConfig:
    """Configuration for ensemble weighting methods"""

    # Weighting method: 'uniform', 'mwu', 'thompson', 'ucb', 'irt'
    weighting_method: str = 'thompson'

    # Multiplicative Weights Update (MWU)
    mwu_learning_rate: float = 0.3

    # UCB parameters
    ucb_exploration_param: float = 1.0

    # Thompson Sampling
    thompson_alpha_prior: float = 1.0
    thompson_beta_prior: float = 1.0

    # Context encoding
    use_temporal_context: bool = True

    # Optimization
    bayesian_opt_iterations: int = 20
    bayesian_opt_trigger_samples: int = 50


@dataclass
class SmartMoneyConfig:
    """Configuration for smart money detection models"""

    # VPIN parameters
    vpin_buckets: int = 50
    vpin_volume_bucket_size: float = 0.02  # 2% of daily volume
    vpin_threshold: float = 0.75  # 75th percentile

    # Large trade thresholds
    major_market_dollar_threshold: float = 10000.0
    major_market_oi_percent: float = 0.01  # 1% of open interest
    niche_market_dollar_threshold: float = 1000.0
    niche_market_oi_percent: float = 0.05  # 5% of open interest

    # Trade size percentile
    large_trade_percentile: float = 95.0


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning and human-in-the-loop"""

    # Query strategy: 'qbc', 'bald', 'uncertainty', 'random'
    query_strategy: str = 'qbc'

    # Query-by-Committee
    qbc_committee_size: int = 4  # Use all base detectors

    # Budget
    manual_review_budget: int = 100
    batch_size: int = 10

    # Confidence thresholds
    high_confidence_threshold: float = 0.9
    low_confidence_threshold: float = 0.5

    # F1 optimization
    optimize_f1: bool = True
    threshold_search_min: float = 0.3
    threshold_search_max: float = 0.7
    threshold_search_step: float = 0.01


@dataclass
class KalshiConfig:
    """Configuration for Kalshi API access."""

    enabled: bool = False
    api_key: Optional[str] = None
    api_base: str = "https://api.elections.kalshi.com"


@dataclass
class Config:
    """Main configuration class"""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    smart_money: SmartMoneyConfig = field(default_factory=SmartMoneyConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)

    # Random seed for reproducibility
    random_seed: int = 42

    # Logging
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Create a config instance from a mapping."""
        instance = copy.deepcopy(_DEFAULT_CONFIG)
        _apply_updates(instance, data)
        return instance


def load_config(
    *,
    env_file: Optional[os.PathLike[str] | str] = DEFAULT_ENV_FILE,
    config_paths: Optional[Sequence[os.PathLike[str] | str]] = None,
    cli_overrides: Optional[Mapping[str, Any]] = None,
    env_prefix: str = ENV_PREFIX,
) -> Config:
    """Load configuration by merging .env, YAML, and CLI overrides.

    Args:
        env_file: Optional path to a .env file. If provided and exists, it
            will be loaded before reading environment variables.
        config_paths: Optional sequence of paths (files or directories) that
            contain YAML configuration. Directories are traversed and all
            ``*.yml``/``*.yaml`` files are merged in lexical order.
        cli_overrides: Optional mapping of overrides (e.g., parsed CLI args).
            Dotted-key notation (``section.option``) is supported.
        env_prefix: Prefix used to extract environment overrides. Variables
            must follow the ``PREFIX__SECTION__OPTION`` convention.

    Returns:
        A fully merged :class:`Config` instance.
    """

    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)

    merged_overrides: Dict[str, Any] = {}

    # YAML configuration
    yaml_overrides = _load_yaml_overrides(config_paths)
    _deep_update(merged_overrides, yaml_overrides)

    # Environment configuration
    env_overrides = _extract_env_overrides(env_prefix)
    _deep_update(merged_overrides, env_overrides)

    # CLI overrides
    if cli_overrides:
        normalized_cli = _normalize_overrides(cli_overrides)
        _deep_update(merged_overrides, normalized_cli)

    config = copy.deepcopy(_DEFAULT_CONFIG)
    _apply_updates(config, merged_overrides)
    _validate_config(config)
    return config


def _validate_config(config: Config) -> None:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    detector = config.detector
    require(detector.zscore_threshold > 0, "detector.zscore_threshold must be > 0")
    require(detector.zscore_rolling_window > 0, "detector.zscore_rolling_window must be > 0")
    require(detector.iqr_multiplier > 0, "detector.iqr_multiplier must be > 0")
    require(detector.iqr_rolling_window > 0, "detector.iqr_rolling_window must be > 0")
    require(
        0 < detector.percentile_threshold <= 100,
        "detector.percentile_threshold must be between 0 and 100",
    )
    require(
        detector.percentile_rolling_window > 0,
        "detector.percentile_rolling_window must be > 0",
    )
    require(
        detector.volume_threshold_multiplier > 0,
        "detector.volume_threshold_multiplier must be > 0",
    )
    require(
        detector.volume_rolling_window > 0,
        "detector.volume_rolling_window must be > 0",
    )

    ensemble = config.ensemble
    require(
        ensemble.weighting_method in {"uniform", "mwu", "thompson", "ucb", "irt"},
        "ensemble.weighting_method must be one of uniform, mwu, thompson, ucb, irt",
    )
    require(
        0 < ensemble.mwu_learning_rate <= 1,
        "ensemble.mwu_learning_rate must be between 0 and 1",
    )
    require(
        ensemble.ucb_exploration_param >= 0,
        "ensemble.ucb_exploration_param must be >= 0",
    )
    require(
        ensemble.thompson_alpha_prior > 0,
        "ensemble.thompson_alpha_prior must be > 0",
    )
    require(
        ensemble.thompson_beta_prior > 0,
        "ensemble.thompson_beta_prior must be > 0",
    )
    require(
        ensemble.bayesian_opt_iterations >= 0,
        "ensemble.bayesian_opt_iterations must be >= 0",
    )
    require(
        ensemble.bayesian_opt_trigger_samples >= 0,
        "ensemble.bayesian_opt_trigger_samples must be >= 0",
    )

    smart_money = config.smart_money
    require(smart_money.vpin_buckets > 0, "smart_money.vpin_buckets must be > 0")
    require(
        0 < smart_money.vpin_volume_bucket_size <= 1,
        "smart_money.vpin_volume_bucket_size must be between 0 and 1",
    )
    require(
        0 < smart_money.vpin_threshold <= 1,
        "smart_money.vpin_threshold must be between 0 and 1",
    )
    require(
        smart_money.major_market_dollar_threshold > 0,
        "smart_money.major_market_dollar_threshold must be > 0",
    )
    require(
        smart_money.major_market_oi_percent > 0,
        "smart_money.major_market_oi_percent must be > 0",
    )
    require(
        smart_money.niche_market_dollar_threshold > 0,
        "smart_money.niche_market_dollar_threshold must be > 0",
    )
    require(
        smart_money.niche_market_oi_percent > 0,
        "smart_money.niche_market_oi_percent must be > 0",
    )
    require(
        0 < smart_money.large_trade_percentile <= 100,
        "smart_money.large_trade_percentile must be between 0 and 100",
    )

    active_learning = config.active_learning
    require(
        active_learning.query_strategy in {"qbc", "bald", "uncertainty", "random"},
        "active_learning.query_strategy must be one of qbc, bald, uncertainty, random",
    )
    require(
        active_learning.qbc_committee_size > 0,
        "active_learning.qbc_committee_size must be > 0",
    )
    require(
        active_learning.manual_review_budget >= 0,
        "active_learning.manual_review_budget must be >= 0",
    )
    require(active_learning.batch_size > 0, "active_learning.batch_size must be > 0")
    require(
        0 <= active_learning.low_confidence_threshold <= 1,
        "active_learning.low_confidence_threshold must be between 0 and 1",
    )
    require(
        0 <= active_learning.high_confidence_threshold <= 1,
        "active_learning.high_confidence_threshold must be between 0 and 1",
    )
    require(
        active_learning.low_confidence_threshold
        <= active_learning.high_confidence_threshold,
        "active_learning.low_confidence_threshold must be <= high_confidence_threshold",
    )
    require(
        0 <= active_learning.threshold_search_min <= 1,
        "active_learning.threshold_search_min must be between 0 and 1",
    )
    require(
        0 <= active_learning.threshold_search_max <= 1,
        "active_learning.threshold_search_max must be between 0 and 1",
    )
    require(
        active_learning.threshold_search_min
        <= active_learning.threshold_search_max,
        "active_learning.threshold_search_min must be <= threshold_search_max",
    )
    require(
        active_learning.threshold_search_step > 0,
        "active_learning.threshold_search_step must be > 0",
    )

    kalshi = config.kalshi
    if kalshi.enabled:
        require(
            bool(kalshi.api_key and kalshi.api_key.strip()),
            "kalshi.api_key is required when kalshi.enabled is true",
        )
        require(
            bool(kalshi.api_base and kalshi.api_base.strip()),
            "kalshi.api_base is required when kalshi.enabled is true",
        )

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid configuration:\n{formatted}")


def _load_yaml_overrides(
    config_paths: Optional[Sequence[os.PathLike[str] | str]]
) -> Dict[str, Any]:
    paths: Sequence[os.PathLike[str] | str]
    if config_paths:
        paths = config_paths
    else:
        paths = [DEFAULT_CONFIG_DIR]

    overrides: Dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_dir():
            yaml_files = sorted(
                list(path.glob("*.yml")) + list(path.glob("*.yaml"))
            )
            for yaml_file in yaml_files:
                overrides = _merge_yaml_file(overrides, yaml_file)
        else:
            overrides = _merge_yaml_file(overrides, path)
    return overrides


def _merge_yaml_file(base: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Failed to parse YAML config: {file_path}") from exc

    if not isinstance(data, MutableMapping):
        raise ValueError(f"YAML config must be a mapping: {file_path}")

    result: Dict[str, Any] = dict(base)
    _deep_update(result, data)  # type: ignore[arg-type]
    return result


def _extract_env_overrides(prefix: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    prefix_with_sep = f"{prefix}__"
    for key, value in os.environ.items():
        if not key.startswith(prefix_with_sep):
            continue
        nested_key = key[len(prefix_with_sep) :].lower()
        segments = nested_key.split("__")
        _set_nested_value(overrides, segments, value)
    return overrides


def _normalize_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            normalized[key] = _normalize_overrides(value)
            continue
        segments = key.split(".") if "." in key else [key]
        _set_nested_value(normalized, segments, value)
    return normalized


def _set_nested_value(container: Dict[str, Any], keys: Sequence[str], value: Any):
    current = container
    for segment in keys[:-1]:
        current = current.setdefault(segment, {})  # type: ignore[assignment]
    current[keys[-1]] = value


def _deep_update(target: Dict[str, Any], overrides: Mapping[str, Any]):
    for key, value in overrides.items():
        if (
            key in target
            and isinstance(target[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _apply_updates(obj: Any, updates: Mapping[str, Any]):
    if not is_dataclass(obj):
        raise TypeError("Updates can only be applied to dataclass instances")

    field_map = {field.name: field for field in fields(obj)}
    for key, value in updates.items():
        if key not in field_map:
            continue
        field_info = field_map[key]
        current_value = getattr(obj, key)
        if is_dataclass(current_value) and isinstance(value, Mapping):
            _apply_updates(current_value, value)
        else:
            coerced = _coerce_value(value, field_info.type)
            setattr(obj, key, coerced)


def _coerce_value(value: Any, target_type: Any) -> Any:
    from typing import get_args, get_origin

    origin = get_origin(target_type)
    if origin is None:
        coerced = _coerce_simple(value, target_type)
        return coerced

    if origin is list:
        elem_type = get_args(target_type)[0] if get_args(target_type) else Any
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_coerce_value(v, elem_type) for v in value]
        return value

    if origin is dict:
        key_type, val_type = get_args(target_type) or (Any, Any)
        if isinstance(value, Mapping):
            return {
                _coerce_value(k, key_type): _coerce_value(v, val_type)
                for k, v in value.items()
            }
        return value

    if origin is Union:
        for arg in get_args(target_type):
            if arg is type(None):
                if value in (None, "None", "null"):
                    return None
                continue
            try:
                return _coerce_value(value, arg)
            except (TypeError, ValueError):
                continue
        return value

    return value


def _coerce_simple(value: Any, target_type: Any) -> Any:
    if target_type in (Any, object):
        return value
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


# Default configuration instance used for deep copies
_DEFAULT_CONFIG = Config()


__all__ = [
    "ActiveLearningConfig",
    "Config",
    "DetectorConfig",
    "EnsembleConfig",
    "KalshiConfig",
    "SmartMoneyConfig",
    "load_config",
]
