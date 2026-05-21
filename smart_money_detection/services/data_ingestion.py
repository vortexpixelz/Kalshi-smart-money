"""Services for preparing data inputs to the detection pipeline."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..features import TemporalFeatureEncoder


class DataIngestionService:
    """Extracts core tensors and contextual features from trade data."""

    def __init__(self, feature_encoder: Optional[TemporalFeatureEncoder] = None):
        self.feature_encoder = feature_encoder or TemporalFeatureEncoder()

    def extract_volumes(self, trades: pd.DataFrame, volume_col: str) -> np.ndarray:
        """Return volumes as a 2D array suitable for detector input."""
        volumes = trades[volume_col].to_numpy(dtype=float).reshape(-1, 1)
        return volumes

    def build_temporal_context(
        self,
        trades: pd.DataFrame,
        timestamp_col: str,
        use_temporal_context: bool = True,
    ) -> Optional[np.ndarray]:
        """Generate temporal context features if requested."""
        if not use_temporal_context or timestamp_col not in trades.columns:
            return None

        timestamps = trades[timestamp_col]
        features_dict = self.feature_encoder.encode_timestamp(
            timestamps, include_all=False
        )
        feature_values = [values for values in features_dict.values() if values is not None]
        if not feature_values:
            return None
        return np.column_stack(feature_values)
