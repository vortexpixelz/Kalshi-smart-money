import numpy as np
import pandas as pd
import pytest

from smart_money_detection.pipeline import SmartMoneyDetector


@pytest.fixture()
def synthetic_trades() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=50, freq="min")
    volumes = rng.integers(1, 25, size=len(timestamps)).astype(float)
    prices = rng.normal(loc=50, scale=5, size=len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "volume": volumes, "price": prices})


def test_fit_predict_with_synthetic_data(synthetic_trades: pd.DataFrame) -> None:
    detector = SmartMoneyDetector()
    detector.fit(synthetic_trades, price_col="price")

    scores = detector.score(synthetic_trades)
    predictions = detector.predict(synthetic_trades)

    assert scores.shape == (len(synthetic_trades),)
    assert predictions.shape == (len(synthetic_trades),)
    assert set(predictions).issubset({0, 1})


def test_predict_empty_after_fit_logs_info(
    synthetic_trades: pd.DataFrame, caplog
) -> None:
    detector = SmartMoneyDetector()
    detector.fit(synthetic_trades, price_col="price")
    empty = synthetic_trades.iloc[0:0]

    with caplog.at_level("INFO"):
        predictions = detector.predict(empty)

    assert predictions.size == 0
    assert "no trades provided" in caplog.text.lower()


def test_fit_missing_columns_raises(synthetic_trades: pd.DataFrame) -> None:
    detector = SmartMoneyDetector()
    with pytest.raises(ValueError):
        detector.fit(synthetic_trades.drop(columns=["volume"]))
    with pytest.raises(ValueError):
        detector.fit(synthetic_trades.drop(columns=["timestamp"]))


def test_add_feedback_handles_empty_labels(
    synthetic_trades: pd.DataFrame, caplog
) -> None:
    detector = SmartMoneyDetector()
    detector.fit(synthetic_trades, price_col="price")

    with caplog.at_level("INFO"):
        detector.add_feedback(sample_ids=[0, 1], labels=np.array([]))

    assert "no feedback labels" in caplog.text.lower()
