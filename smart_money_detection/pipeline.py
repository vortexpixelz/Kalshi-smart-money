"""Main detection pipeline orchestrating all components."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .active_learning import FeedbackManager, QueryByCommittee
from .config import Config, load_config
from .detectors import (
    IQRDetector,
    PercentileDetector,
    RelativeVolumeDetector,
    ZScoreDetector,
)
from .ensemble import AnomalyEnsemble
 codex/refactor-config-and-orchestration-layers
from .models import SimplifiedPIN, VPINClassifier
from .services import DataIngestionService, DetectionService
from .utils import bayesian_optimize_weights, compute_metrics, find_optimal_threshold

from .features import TemporalFeatureEncoder
from .models import VPIN, VPINClassifier, SimplifiedPIN
from .active_learning import QueryByCommittee, FeedbackManager
from .utils import (
    compute_metrics,
    find_optimal_threshold,
    bayesian_optimize_weights,
    gradient_optimize_weights,
)
from .utils.optimization import grid_search_weights
 main


class SmartMoneyDetector:
    """
    Complete smart money detection system with minimal labeled data

    Combines:
    - Base anomaly detectors (z-score, IQR, percentile, volume)
    - Adaptive ensemble weighting (Thompson Sampling, UCB, MWU)
    - Smart money models (VPIN, PIN)
    - Active learning (Query-by-Committee)
    - Temporal feature encoding
    - Human-in-the-loop feedback

    Example:
        >>> detector = SmartMoneyDetector()
        >>> detector.fit(historical_trades)
        >>> anomalies = detector.predict(new_trades)
        >>> queries = detector.suggest_manual_reviews(new_trades, n=10)
        >>> detector.add_feedback(sample_ids, labels)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        data_service: Optional[DataIngestionService] = None,
        detection_service: Optional[DetectionService] = None,
    ):
        """
        Initialize smart money detector

        Args:
            config: Configuration object (default: use default config)
        """
        self.config = config or load_config()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.data_service = data_service or DataIngestionService()
        self.feature_encoder = self.data_service.feature_encoder
        self.detection_service = detection_service or self._build_detection_service()
        self.ensemble = self.detection_service.ensemble
        self.detectors = list(self.detection_service.detectors)

        # Initialize additional components
        self._init_smart_money_models()
        self._init_active_learning()

        # State tracking
        self.is_fitted = False
        self.n_samples_seen = 0
        self._detector_score_cache: Dict[Any, np.ndarray] = {}

    def _build_detection_service(self) -> DetectionService:
        """Construct detectors and the ensemble service."""
        cfg = self.config.detector

        detectors = [
            ZScoreDetector(
                threshold=cfg.zscore_threshold,
                rolling_window=cfg.zscore_rolling_window,
            ),
            IQRDetector(
                multiplier=cfg.iqr_multiplier,
                rolling_window=cfg.iqr_rolling_window,
            ),
            PercentileDetector(
                percentile=cfg.percentile_threshold,
                rolling_window=cfg.percentile_rolling_window,
            ),
            RelativeVolumeDetector(
                threshold_multiplier=cfg.volume_threshold_multiplier,
                rolling_window=cfg.volume_rolling_window,
            ),
        ]

        cfg = self.config.ensemble

        weighting_params = {
            'learning_rate': cfg.mwu_learning_rate,
            'exploration_param': cfg.ucb_exploration_param,
            'alpha_prior': cfg.thompson_alpha_prior,
            'beta_prior': cfg.thompson_beta_prior,
        }

        ensemble = AnomalyEnsemble(
            detectors=detectors,
            weighting_method=cfg.weighting_method,
            weighting_params=weighting_params,
        )

        return DetectionService(ensemble)

    def _init_smart_money_models(self):
        """Initialize smart money detection models"""
        cfg = self.config.smart_money

        self.vpin_model = VPINClassifier(
            threshold=cfg.vpin_threshold,
            n_buckets=cfg.vpin_buckets,
            bucket_pct_of_daily=cfg.vpin_volume_bucket_size,
        )

        self.pin_model = SimplifiedPIN(
            large_trade_threshold=cfg.large_trade_percentile / 100,
        )

    def _init_active_learning(self):
        """Initialize active learning components"""
        cfg = self.config.active_learning

        if cfg.query_strategy == 'qbc':
            self.query_strategy = QueryByCommittee(batch_size=cfg.batch_size)
        else:
            # Default to QBC
            self.query_strategy = QueryByCommittee(batch_size=cfg.batch_size)

        self.feedback_manager = FeedbackManager(optimize_f1=cfg.optimize_f1)

    def fit(
        self,
        trades: pd.DataFrame,
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp',
        price_col: Optional[str] = None,
    ):
        """
        Fit detector on historical trade data

        Args:
            trades: DataFrame with trade data
            volume_col: Name of volume column
            timestamp_col: Name of timestamp column
            price_col: Optional name of price column (for VPIN)

        Returns:
            self
        """
        if trades is None or trades.empty:
            self.logger.warning("Received empty trade dataset for fit; skipping training.")
            self.is_fitted = False
            self.n_samples_seen = 0
            return self

        if volume_col not in trades.columns:
            raise ValueError(f"Column '{volume_col}' not found in trade data")

        if timestamp_col not in trades.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in trade data")

        self.logger.info(f"Fitting smart money detector on {len(trades)} trades")

        # Extract volumes
        volumes = self.data_service.extract_volumes(trades, volume_col)

        # Fit detectors and ensemble
        self.detection_service.fit(volumes)

        # Fit smart money models
        self.pin_model.fit(volumes.flatten())

        if price_col is not None and price_col in trades.columns:
            prices = trades[price_col].values
            self.vpin_model.fit(prices, volumes.flatten())

        self.is_fitted = True
        self.n_samples_seen = len(trades)

        self.logger.info("Smart money detector fitted successfully")

        return self

    def predict(
        self,
        trades: pd.DataFrame,
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp',
        use_temporal_context: bool = True,
    ) -> np.ndarray:
        """
        Predict smart money (informed trading) on new trades

        Args:
            trades: DataFrame with trade data
            volume_col: Name of volume column
            timestamp_col: Name of timestamp column
            use_temporal_context: If True, use temporal features for contextual weighting

        Returns:
            Binary predictions (0 = normal, 1 = smart money)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        if trades is None or trades.empty:
            self.logger.info("No trades provided for prediction; returning empty result.")
            return np.array([], dtype=int)

        scores = self.score(trades, volume_col, timestamp_col, use_temporal_context)

        # Use optimal threshold from feedback if available
        threshold = self.feedback_manager.get_optimal_threshold()

        predictions = (scores >= threshold).astype(int)

        return predictions

    def score(
        self,
        trades: pd.DataFrame,
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp',
        use_temporal_context: bool = True,
    ) -> np.ndarray:
        """
        Compute anomaly scores for trades

        Args:
            trades: DataFrame with trade data
            volume_col: Name of volume column
            timestamp_col: Name of timestamp column
            use_temporal_context: If True, use temporal features

        Returns:
            Anomaly scores in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        volumes = self.data_service.extract_volumes(trades, volume_col)

        # Get temporal context if enabled
        context = self.data_service.build_temporal_context(
            trades, timestamp_col, use_temporal_context
        )

        # Get ensemble scores
 codex/refactor-config-and-orchestration-layers
        scores = self.detection_service.score(volumes, context)

        detector_scores = []
        for detector in self.detectors:
            raw_scores = np.asarray(detector.score(volumes)).flatten()
            if raw_scores.size != len(volumes):
                raise ValueError(
                    f"Detector {detector.name} returned unexpected score shape."
                )
            if raw_scores.max() > raw_scores.min():
                normalized = (raw_scores - raw_scores.min()) / (
                    raw_scores.max() - raw_scores.min()
                )
            else:
                normalized = raw_scores
            detector_scores.append(normalized)

        detector_scores = (
            np.column_stack(detector_scores) if detector_scores else np.empty((len(volumes), 0))
        )

        # Cache detector scores for later optimization
        for idx, sample_id in enumerate(trades.index):
            self._detector_score_cache[sample_id] = detector_scores[idx].copy()

        scores = self.ensemble.weighting.combine_scores(detector_scores)
 main

        return scores

    def suggest_manual_reviews(
        self,
        trades: pd.DataFrame,
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp',
        n_queries: int = 10,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Suggest which trades to manually review using active learning

        Uses Query-by-Committee to select trades where detectors disagree most.

        Args:
            trades: DataFrame with trade data
            volume_col: Name of volume column
            timestamp_col: Name of timestamp column
            n_queries: Number of trades to suggest for review

        Returns:
            Tuple of (indices, suggested_trades_df)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        volumes = self.data_service.extract_volumes(trades, volume_col)

        codex/add-context-features-to-manual-reviews
        context = None
        if timestamp_col in trades.columns:
            timestamps = trades[timestamp_col]
            if isinstance(timestamps, pd.Series) and timestamps.notna().any():
                context = self._get_temporal_context(timestamps)

        # Get predictions from all detectors
 codex/refactor-config-and-orchestration-layers
        committee_predictions, committee_scores = self.detection_service.committee_outputs(
            volumes
        )

        # Select queries using QBC
        ensemble_scores = self.detection_service.score(volumes)


        # Get predictions and scores from all detectors with single scoring pass
        main
        committee_predictions = []
        committee_scores = []

        for detector in self.detectors:
            predictions, scores = detector.predict_with_scores(volumes)
            committee_predictions.append(predictions)
            committee_scores.append(scores)

        committee_predictions = np.column_stack(committee_predictions)
        committee_scores = np.column_stack(committee_scores)

        # Select queries using QBC
        ensemble_scores = self.ensemble.score(volumes, context)
 main

        query_indices = self.query_strategy.select_queries(
            volumes,
            ensemble_scores,
            n_queries=n_queries,
            committee_predictions=committee_predictions,
            committee_scores=committee_scores,
        )

        suggested_trades = trades.iloc[query_indices].copy()

        self.logger.info(f"Suggested {len(query_indices)} trades for manual review")

        return query_indices, suggested_trades

    def add_feedback(
        self,
        sample_ids: List[Any],
        labels: np.ndarray,
        trades: Optional[pd.DataFrame] = None,
        volume_col: str = 'volume',
        update_weights: bool = True,
    ):
        """
        Add manual review feedback and update ensemble weights

        Args:
            sample_ids: List of sample identifiers
            labels: True labels (0 = normal, 1 = smart money)
            trades: Optional DataFrame with trade data for weight updates
            volume_col: Name of volume column
            update_weights: If True, update ensemble weights based on feedback
        """
        # Add to feedback manager
        weights = self.ensemble.get_weights()
        ensemble_scores: List[Optional[float]] = []
        predictions: List[Optional[int]] = []

        for sample_id in sample_ids:
            cached_scores = self._detector_score_cache.get(sample_id)
            if cached_scores is None:
                ensemble_scores.append(None)
                predictions.append(None)
            else:
                score = float(np.dot(cached_scores, weights))
                ensemble_scores.append(score)
                predictions.append(int(score >= 0.5))

        self.feedback_manager.add_batch_feedback(
            sample_ids,
            labels,
            y_pred=predictions,
            scores=ensemble_scores,
        )

        self.logger.info(f"Added feedback for {len(labels)} samples")

        # Update ensemble weights if requested and trade data provided
        if update_weights and trades is not None:
            subset = trades.loc[sample_ids, volume_col]
            volumes = subset.to_numpy(dtype=float).reshape(-1, 1)
            self.detection_service.update(volumes, labels)

            self.logger.info("Updated ensemble weights based on feedback")

        # Log statistics
        stats = self.feedback_manager.get_statistics()
        self.logger.info(f"Feedback statistics: {stats}")

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about manual reviews and performance"""
        return self.feedback_manager.get_statistics()

    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble detector weights"""
        weights = self.detection_service.get_weights()
        detector_names = [d.name for d in self.detection_service.detectors]

        return dict(zip(detector_names, weights))

    def get_detector_contributions(
        self, trades: pd.DataFrame, volume_col: str = 'volume'
    ) -> Dict[str, Any]:
        """
        Get individual detector contributions for interpretability

        Args:
            trades: DataFrame with trade data
            volume_col: Name of volume column

        Returns:
            Dictionary with detector contributions
        """
        volumes = self.data_service.extract_volumes(trades, volume_col)
        return self.detection_service.get_detector_contributions(volumes)

    def optimize_weights(
        self,
        method: str = 'bayesian',
        n_iterations: int = 20,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize ensemble weights using feedback data

        Should only be called after accumulating 10-50 labeled examples.

        Args:
            method: Optimization method ('bayesian', 'gradient', 'evolutionary')
            n_iterations: Number of optimization iterations

        Returns:
            Tuple of (optimal_weights, performance_metric)
        """
        # Get labeled data
        sample_ids, labels, _ = self.feedback_manager.get_labeled_data()

        if len(labels) < 10:
            self.logger.warning(
                f"Only {len(labels)} labeled samples available. "
                "Recommend waiting until 10-50 samples before optimizing weights."
            )

        self.logger.info(f"Optimizing ensemble weights using {method} method")

 codex/refactor-config-and-orchestration-layers
        # Return current weights as placeholder
        current_weights = self.detection_service.get_weights()

        # Gather cached detector scores for labeled samples
        cached_scores = []
        cached_labels = []
        missing_samples = []

        for sample_id, label in zip(sample_ids, labels):
            scores = self._detector_score_cache.get(sample_id)
            if scores is None:
                missing_samples.append(sample_id)
                continue
            cached_scores.append(scores)
            cached_labels.append(label)

        if missing_samples:
            self.logger.warning(
                "No cached detector scores available for samples: %s", missing_samples
            )

        if not cached_scores:
            self.logger.error("Cannot optimize weights without cached detector scores.")
            return self.ensemble.get_weights(), 0.0

        detector_scores = np.vstack(cached_scores)
        y_true = np.array(cached_labels)

        def evaluate_weights(weights: np.ndarray) -> float:
            ensemble_scores = np.dot(detector_scores, weights)
            y_pred = (ensemble_scores >= 0.5).astype(int)
            metrics = compute_metrics(y_true, y_pred, ensemble_scores)
            return metrics.get('f1_score', 0.0)

        baseline_weights = self.ensemble.get_weights()
        baseline_metric = evaluate_weights(baseline_weights)

        method = method.lower()
        if method == 'bayesian':
            optimal_weights, best_metric = bayesian_optimize_weights(
                detector_scores,
                y_true,
                n_iterations=n_iterations,
            )
        elif method == 'gradient':
            optimal_weights, _ = gradient_optimize_weights(
                detector_scores,
                y_true,
                max_iter=max(n_iterations, 1),
            )
            best_metric = evaluate_weights(optimal_weights)
        elif method == 'grid':
            optimal_weights, best_metric = grid_search_weights(
                detector_scores,
                y_true,
                metric_fn=lambda yt, yp: compute_metrics(yt, yp).get('f1_score', 0.0),
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        if optimal_weights is None:
            self.logger.error("Optimization did not return valid weights.")
            return baseline_weights, baseline_metric

        # Normalize weights to ensure they sum to 1
        if optimal_weights.sum() > 0:
            optimal_weights = optimal_weights / optimal_weights.sum()

        # Update ensemble weights
        self.ensemble.weighting.weights = optimal_weights

        improvement = best_metric - baseline_metric
        self.logger.info(
            "Weight optimization completed. Baseline F1: %.4f, Optimized F1: %.4f, "
            "Improvement: %.4f",
            baseline_metric,
            best_metric,
            improvement,
        )
 main

        return optimal_weights, best_metric

    def save_state(self, filepath: str):
        """
        Save detector state to file

        Args:
            filepath: Path to save state
        """
        import pickle

        state = {
            'config': self.config.to_dict(),
            'ensemble_state': self.ensemble.get_state(),
            'feedback_data': self.feedback_manager.feedback_data,
            'is_fitted': self.is_fitted,
            'n_samples_seen': self.n_samples_seen,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info(f"Saved detector state to {filepath}")

    def load_state(self, filepath: str):
        """
        Load detector state from file

        Args:
            filepath: Path to load state from
        """
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.ensemble.set_state(state['ensemble_state'])
        self.detectors = list(self.detection_service.detectors)
        self.feedback_manager.feedback_data = state['feedback_data']
        self.is_fitted = state['is_fitted']
        self.n_samples_seen = state['n_samples_seen']

        self.logger.info(f"Loaded detector state from {filepath}")

    def evaluate(
        self, trades: pd.DataFrame, labels: np.ndarray, volume_col: str = 'volume'
    ) -> Dict[str, float]:
        """
        Evaluate detector performance on labeled data

        Args:
            trades: DataFrame with trade data
            labels: True labels
            volume_col: Name of volume column

        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict(trades, volume_col)
        scores = self.score(trades, volume_col)

        metrics = compute_metrics(labels, predictions, scores)

        self.logger.info(f"Evaluation metrics: {metrics}")

        return metrics
