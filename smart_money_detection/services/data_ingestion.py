"""Services for preparing data inputs to the detection pipeline."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..features import TemporalFeatureEncoder

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Extracts core tensors and contextual features from trade data."""

    def __init__(self, feature_encoder: Optional[TemporalFeatureEncoder] = None):
        self.feature_encoder = feature_encoder or TemporalFeatureEncoder()

    def extract_volumes(self, trades: pd.DataFrame, volume_col: str) -> np.ndarray:
        """Return volumes as a 2D array suitable for detector input."""
        volumes = trades[volume_col].to_numpy(dtype=float).reshape(-1, 1)
        logger.debug(
            "Extracted volumes: shape=%s, mean=%.4f, max=%.4f",
            volumes.shape, float(volumes.mean()), float(volumes.max()),
        )
        return volumes

    def build_temporal_context(
        self,
        trades: pd.DataFrame,
        timestamp_col: str,
        use_temporal_context: bool = True,
    ) -> Optional[np.ndarray]:
        """Generate temporal context features if requested."""
        if not use_temporal_context or timestamp_col not in trades.columns:
            logger.debug(
                "Temporal context skipped (use_temporal_context=%s, col_present=%s)",
                use_temporal_context, timestamp_col in trades.columns,
            )
            return None

        timestamps = trades[timestamp_col]
        features_dict = self.feature_encoder.encode_timestamp(
            timestamps, include_all=False
        )
        feature_values = [values for values in features_dict.values() if values is not None]
        if not feature_values:
            logger.debug("No temporal features produced for column '%s'", timestamp_col)
            return None
        context = np.column_stack(feature_values)
        logger.debug(
            "Built temporal context: shape=%s from column '%s'", context.shape, timestamp_col
        )
        return context
