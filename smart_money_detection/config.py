"""
Configuration for smart money detection system
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


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
class Config:
    """Main configuration class"""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    smart_money: SmartMoneyConfig = field(default_factory=SmartMoneyConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)

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
            'random_seed': self.random_seed,
            'log_level': self.log_level,
        }


# Default configuration instance
default_config = Config()
