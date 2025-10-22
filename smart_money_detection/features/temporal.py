"""
Temporal feature engineering with cyclical encoding

Implements cyclical encoding for time-based features to preserve periodicity,
as recommended by research: hour_sin = sin(2π × hour/24), hour_cos = cos(2π × hour/24)
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, List, Dict, Any


class TemporalFeatureEncoder:
    """
    Encodes temporal features using cyclical transformations to preserve periodicity.

    This is critical for time-based features where linear encoding would incorrectly
    treat 11pm and 1am as distant (23 vs 1) when they are temporally adjacent.

    Features:
        - Hour of day (0-23) -> sin/cos with period 24
        - Day of week (0-6) -> sin/cos with period 7
        - Day of month (1-31) -> sin/cos with period 31
        - Month of year (1-12) -> sin/cos with period 12
        - Time to resolution (continuous)
    """

    def __init__(self):
        """Initialize temporal feature encoder"""
        self.feature_names_ = []

    def encode_hour(self, hour: Union[int, np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        Encode hour of day (0-23) as cyclical features

        Args:
            hour: Hour value(s) in range [0, 23]

        Returns:
            Dictionary with 'hour_sin' and 'hour_cos' arrays
        """
        hour = np.asarray(hour)
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
        }

    def encode_day_of_week(
        self, day: Union[int, np.ndarray, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Encode day of week (0-6, Monday=0) as cyclical features

        Args:
            day: Day of week value(s) in range [0, 6]

        Returns:
            Dictionary with 'day_of_week_sin' and 'day_of_week_cos' arrays
        """
        day = np.asarray(day)
        return {
            'day_of_week_sin': np.sin(2 * np.pi * day / 7),
            'day_of_week_cos': np.cos(2 * np.pi * day / 7),
        }

    def encode_day_of_month(
        self, day: Union[int, np.ndarray, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Encode day of month (1-31) as cyclical features

        Args:
            day: Day of month value(s) in range [1, 31]

        Returns:
            Dictionary with 'day_of_month_sin' and 'day_of_month_cos' arrays
        """
        day = np.asarray(day)
        return {
            'day_of_month_sin': np.sin(2 * np.pi * day / 31),
            'day_of_month_cos': np.cos(2 * np.pi * day / 31),
        }

    def encode_month(
        self, month: Union[int, np.ndarray, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Encode month of year (1-12) as cyclical features

        Args:
            month: Month value(s) in range [1, 12]

        Returns:
            Dictionary with 'month_sin' and 'month_cos' arrays
        """
        month = np.asarray(month)
        return {
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
        }

    def encode_timestamp(
        self, timestamp: Union[datetime, pd.Timestamp, pd.Series], include_all: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode timestamp into multiple cyclical temporal features

        Args:
            timestamp: Timestamp value(s)
            include_all: If True, include all temporal features (hour, day, month)

        Returns:
            Dictionary with all encoded temporal features
        """
        if isinstance(timestamp, (datetime, pd.Timestamp)):
            timestamp = pd.Series([timestamp])
        elif not isinstance(timestamp, pd.Series):
            timestamp = pd.Series(timestamp)

        features = {}

        # Always encode hour and day of week (most important)
        hour = timestamp.dt.hour.values
        features.update(self.encode_hour(hour))

        day_of_week = timestamp.dt.dayofweek.values
        features.update(self.encode_day_of_week(day_of_week))

        if include_all:
            # Add day of month and month
            day_of_month = timestamp.dt.day.values
            features.update(self.encode_day_of_month(day_of_month))

            month = timestamp.dt.month.values
            features.update(self.encode_month(month))

        return features

    def encode_time_to_resolution(
        self,
        current_time: Union[datetime, pd.Timestamp, pd.Series],
        resolution_time: Union[datetime, pd.Timestamp, pd.Series],
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Encode time remaining until market resolution

        Args:
            current_time: Current timestamp(s)
            resolution_time: Resolution timestamp(s)
            normalize: If True, normalize by 24 hours (days)

        Returns:
            Dictionary with 'time_to_resolution_hours' and optionally cyclical encoding
        """
        if not isinstance(current_time, pd.Series):
            current_time = pd.Series([current_time])
        if not isinstance(resolution_time, pd.Series):
            resolution_time = pd.Series([resolution_time])

        # Calculate time difference in hours
        time_diff = (resolution_time - current_time).dt.total_seconds() / 3600
        time_diff = time_diff.values

        features = {'time_to_resolution_hours': time_diff}

        if normalize:
            # Normalize by 24 hours (convert to days as fraction)
            features['time_to_resolution_days'] = time_diff / 24

        return features

    def transform(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        resolution_col: str = None,
        include_all: bool = True,
    ) -> pd.DataFrame:
        """
        Transform a DataFrame by adding cyclical temporal features

        Args:
            df: Input DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            resolution_col: Optional name of resolution time column
            include_all: If True, include all temporal features

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()

        # Encode timestamp features
        timestamp_features = self.encode_timestamp(df[timestamp_col], include_all=include_all)

        for name, values in timestamp_features.items():
            df[name] = values

        # Encode time to resolution if provided
        if resolution_col is not None and resolution_col in df.columns:
            resolution_features = self.encode_time_to_resolution(
                df[timestamp_col], df[resolution_col]
            )
            for name, values in resolution_features.items():
                df[name] = values

        # Store feature names
        self.feature_names_ = list(timestamp_features.keys())
        if resolution_col is not None and resolution_col in df.columns:
            self.feature_names_.extend(list(resolution_features.keys()))

        return df

    def get_feature_vector(
        self,
        timestamp: Union[datetime, pd.Timestamp],
        resolution_time: Union[datetime, pd.Timestamp] = None,
        include_all: bool = True,
    ) -> np.ndarray:
        """
        Get feature vector for a single timestamp

        Args:
            timestamp: Current timestamp
            resolution_time: Optional resolution timestamp
            include_all: If True, include all temporal features

        Returns:
            Feature vector as numpy array
        """
        features = self.encode_timestamp(timestamp, include_all=include_all)

        if resolution_time is not None:
            resolution_features = self.encode_time_to_resolution(timestamp, resolution_time)
            features.update(resolution_features)

        # Concatenate all feature values
        feature_vector = np.concatenate([v.flatten() for v in features.values()])

        return feature_vector

    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order"""
        return self.feature_names_.copy()


class PositionalEncoder:
    """
    Transformer-style positional encoding for multiple temporal scales

    Implements: PE(pos, 2i) = sin(pos / 10000^(2i/d))
                PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    This generalizes cyclical encoding to multiple frequencies simultaneously.
    """

    def __init__(self, d_model: int = 16):
        """
        Initialize positional encoder

        Args:
            d_model: Dimension of positional encoding (must be even)
        """
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        self.d_model = d_model

        # Pre-compute division terms
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.div_term = div_term

    def encode(self, positions: Union[int, np.ndarray, pd.Series]) -> np.ndarray:
        """
        Encode positions using sinusoidal positional encoding

        Args:
            positions: Position value(s)

        Returns:
            Positional encoding array of shape (n_positions, d_model)
        """
        positions = np.asarray(positions).reshape(-1, 1)
        n_positions = positions.shape[0]

        # Initialize encoding array
        encoding = np.zeros((n_positions, self.d_model))

        # Apply sin to even indices
        encoding[:, 0::2] = np.sin(positions * self.div_term)

        # Apply cos to odd indices
        encoding[:, 1::2] = np.cos(positions * self.div_term)

        return encoding
