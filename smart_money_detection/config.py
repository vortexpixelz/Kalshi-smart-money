"""Configuration for smart money detection system"""
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv


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
    """Configuration for Kalshi API access"""

    api_key: str = ""
    email: str = ""
    password: str = ""
    api_base: str = "https://api.elections.kalshi.com"
    demo_mode: bool = True


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
        return {
            'detector': self.detector.__dict__,
            'ensemble': self.ensemble.__dict__,
            'smart_money': self.smart_money.__dict__,
            'active_learning': self.active_learning.__dict__,
            'kalshi': self.kalshi.__dict__,
            'random_seed': self.random_seed,
            'log_level': self.log_level,
        }


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base dict."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(config_dict: Dict[str, Any], path: Any, value: Any) -> None:
    """Set nested value given a path of keys."""

    current = config_dict
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _parse_env_value(value: str) -> Any:
    """Parse env string into bool/int/float when possible."""

    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _collect_env_overrides() -> Dict[str, Any]:
    """Collect overrides from environment variables."""

    overrides: Dict[str, Any] = {}

    env_mapping = {
        'KALSHI_API_KEY': ('kalshi', 'api_key'),
        'KALSHI_EMAIL': ('kalshi', 'email'),
        'KALSHI_PASSWORD': ('kalshi', 'password'),
        'KALSHI_API_BASE': ('kalshi', 'api_base'),
        'KALSHI_DEMO_MODE': ('kalshi', 'demo_mode'),
    }

    for env_var, path in env_mapping.items():
        if env_var in os.environ:
            _set_nested(overrides, path, _parse_env_value(os.environ[env_var]))

    prefix = "SMART_MONEY_"
    for env_var, raw_value in os.environ.items():
        if env_var.startswith(prefix):
            path = env_var[len(prefix):].split("__")
            if not path:
                continue
            normalized_path = [segment.lower() for segment in path]
            _set_nested(overrides, normalized_path, _parse_env_value(raw_value))

    return overrides


def _dict_to_config(config_data: Dict[str, Any]) -> Config:
    """Convert nested dictionary to Config dataclass."""

    return Config(
        detector=DetectorConfig(**config_data.get('detector', {})),
        ensemble=EnsembleConfig(**config_data.get('ensemble', {})),
        smart_money=SmartMoneyConfig(**config_data.get('smart_money', {})),
        active_learning=ActiveLearningConfig(**config_data.get('active_learning', {})),
        kalshi=KalshiConfig(**config_data.get('kalshi', {})),
        random_seed=config_data.get('random_seed', 42),
        log_level=config_data.get('log_level', "INFO"),
    )


def load_config(
    env_path: str = ".env",
    yaml_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load configuration from defaults, YAML, and environment variables.

    Precedence (lowest to highest):
    1. Library defaults
    2. YAML file values
    3. Environment variables (.env is loaded automatically)
    4. Programmatic overrides
    """

    defaults_dict = asdict(Config())

    if env_path:
        load_dotenv(env_path, override=False)

    merged_config = deepcopy(defaults_dict)

    if yaml_path:
        yaml_file = Path(yaml_path)
        if yaml_file.exists():
            with yaml_file.open("r", encoding="utf-8") as fh:
                yaml_config = yaml.safe_load(fh) or {}
            if not isinstance(yaml_config, dict):
                raise ValueError("YAML configuration must be a mapping")
            merged_config = _deep_update(merged_config, yaml_config)

    env_overrides = _collect_env_overrides()
    merged_config = _deep_update(merged_config, env_overrides)

    if overrides:
        merged_config = _deep_update(merged_config, overrides)

    config = _dict_to_config(merged_config)
    validate_config(config)
    return config


def validate_config(config: Config) -> Config:
    """Validate configuration values and credentials."""

    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")

    if config.detector.zscore_threshold <= 0:
        raise ValueError("zscore_threshold must be positive")
    if config.detector.zscore_rolling_window <= 0:
        raise ValueError("zscore_rolling_window must be positive")
    if config.detector.iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be positive")
    if config.detector.iqr_rolling_window <= 0:
        raise ValueError("iqr_rolling_window must be positive")
    if not 0 < config.detector.percentile_threshold <= 100:
        raise ValueError("percentile_threshold must be in (0, 100]")
    if config.detector.percentile_rolling_window <= 0:
        raise ValueError("percentile_rolling_window must be positive")
    if config.detector.volume_threshold_multiplier <= 0:
        raise ValueError("volume_threshold_multiplier must be positive")
    if config.detector.volume_rolling_window <= 0:
        raise ValueError("volume_rolling_window must be positive")

    if config.ensemble.weighting_method not in {'uniform', 'mwu', 'thompson', 'ucb', 'irt'}:
        raise ValueError("weighting_method must be one of: uniform, mwu, thompson, ucb, irt")
    if config.ensemble.mwu_learning_rate <= 0:
        raise ValueError("mwu_learning_rate must be positive")
    if config.ensemble.ucb_exploration_param < 0:
        raise ValueError("ucb_exploration_param cannot be negative")
    if config.ensemble.thompson_alpha_prior <= 0 or config.ensemble.thompson_beta_prior <= 0:
        raise ValueError("Thompson priors must be positive")
    if config.ensemble.bayesian_opt_iterations < 0:
        raise ValueError("bayesian_opt_iterations cannot be negative")
    if config.ensemble.bayesian_opt_trigger_samples <= 0:
        raise ValueError("bayesian_opt_trigger_samples must be positive")

    if config.smart_money.vpin_buckets <= 0:
        raise ValueError("vpin_buckets must be positive")
    if not 0 < config.smart_money.vpin_volume_bucket_size <= 1:
        raise ValueError("vpin_volume_bucket_size must be in (0, 1]")
    if not 0 < config.smart_money.vpin_threshold <= 1:
        raise ValueError("vpin_threshold must be in (0, 1]")
    if config.smart_money.major_market_dollar_threshold <= 0:
        raise ValueError("major_market_dollar_threshold must be positive")
    if not 0 < config.smart_money.major_market_oi_percent <= 1:
        raise ValueError("major_market_oi_percent must be in (0, 1]")
    if config.smart_money.niche_market_dollar_threshold <= 0:
        raise ValueError("niche_market_dollar_threshold must be positive")
    if not 0 < config.smart_money.niche_market_oi_percent <= 1:
        raise ValueError("niche_market_oi_percent must be in (0, 1]")
    if not 0 < config.smart_money.large_trade_percentile <= 100:
        raise ValueError("large_trade_percentile must be in (0, 100]")

    if config.active_learning.query_strategy not in {'qbc', 'bald', 'uncertainty', 'random'}:
        raise ValueError("query_strategy must be one of: qbc, bald, uncertainty, random")
    if config.active_learning.qbc_committee_size <= 0:
        raise ValueError("qbc_committee_size must be positive")
    if config.active_learning.manual_review_budget < 0:
        raise ValueError("manual_review_budget cannot be negative")
    if config.active_learning.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < config.active_learning.high_confidence_threshold <= 1:
        raise ValueError("high_confidence_threshold must be in (0, 1]")
    if not 0 < config.active_learning.low_confidence_threshold < config.active_learning.high_confidence_threshold:
        raise ValueError("low_confidence_threshold must be positive and below high_confidence_threshold")
    if not 0 < config.active_learning.threshold_search_min < config.active_learning.threshold_search_max <= 1:
        raise ValueError("threshold_search_min/max must be in (0, 1] and min < max")
    if config.active_learning.threshold_search_step <= 0:
        raise ValueError("threshold_search_step must be positive")

    if not config.kalshi.demo_mode:
        has_api_key = bool(config.kalshi.api_key)
        has_email_password = bool(config.kalshi.email and config.kalshi.password)
        if not (has_api_key or has_email_password):
            raise ValueError("Kalshi credentials required: set KALSHI_API_KEY or both email and password")
    if not config.kalshi.api_base:
        raise ValueError("kalshi.api_base cannot be empty")

    if config.random_seed < 0:
        raise ValueError("random_seed cannot be negative")
    if config.log_level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ValueError("log_level must be a valid logging level")

    return config
