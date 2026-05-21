"""Pandas helper utilities with validation and performance-focused conversions."""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd


def coerce_trade_dataframe(
    trades: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    numeric_cols: Sequence[str] = ("volume", "price"),
    categorical_cols: Sequence[str] = ("ticker", "market_id"),
    sort: bool = True,
) -> pd.DataFrame:
    """
    Normalize trade data columns with explicit typing and optional sorting.

    Args:
        trades: Input DataFrame containing trade data.
        timestamp_col: Name of timestamp column to coerce to datetime.
        numeric_cols: Column names to coerce to numeric dtype.
        categorical_cols: Column names to convert to categorical dtype.
        sort: Whether to sort by the timestamp column.

    Returns:
        Normalized DataFrame with coerced dtypes.
    """
    if not isinstance(trades, pd.DataFrame):
        raise TypeError("trades must be a pandas DataFrame")

    if trades.empty:
        return trades.copy()

    assignments = {}
    if timestamp_col in trades.columns:
        assignments[timestamp_col] = pd.to_datetime(
            trades[timestamp_col], errors="coerce"
        )

    for col in numeric_cols:
        if col in trades.columns:
            assignments[col] = pd.to_numeric(trades[col], errors="coerce")

    df = trades.assign(**assignments) if assignments else trades.copy()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if sort and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    return df


def trades_from_records(
    records: Optional[Iterable[Mapping[str, object]]],
    *,
    timestamp_col: str = "timestamp",
    numeric_cols: Sequence[str] = ("volume", "price"),
    categorical_cols: Sequence[str] = ("ticker", "market_id"),
    sort: bool = True,
) -> pd.DataFrame:
    """
    Build a trade DataFrame from raw record dictionaries with validation.

    Args:
        records: Iterable of raw trade records.
        timestamp_col: Name of timestamp column to coerce to datetime.
        numeric_cols: Column names to coerce to numeric dtype.
        categorical_cols: Column names to convert to categorical dtype.
        sort: Whether to sort by the timestamp column.

    Returns:
        Normalized DataFrame with coerced dtypes.
    """
    if records is None:
        return pd.DataFrame()

    data = list(records)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return coerce_trade_dataframe(
        df,
        timestamp_col=timestamp_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        sort=sort,
    )
