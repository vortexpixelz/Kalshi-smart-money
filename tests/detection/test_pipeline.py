from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_money_detection.config import Config
from smart_money_detection.pipeline import SmartMoneyDetector


@pytest.fixture(scope="module")
def deterministic_config() -> Config:
    cfg = Config()
    cfg.ensemble.weighting_method = "uniform"
    return cfg


@pytest.fixture(scope="module")
def sandbox_snapshot_trades() -> pd.DataFrame:
    snapshot_path = Path(__file__).resolve().parent.parent / "data" / "sandbox_trades_snapshot.json"
    df = pd.read_json(snapshot_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_fit_and_predict_with_snapshot_data(
    sandbox_snapshot_trades: pd.DataFrame, deterministic_config: Config
) -> None:
    detector = SmartMoneyDetector(config=deterministic_config)
    detector.fit(sandbox_snapshot_trades, price_col="price")

    scores = detector.score(sandbox_snapshot_trades)
    predictions = detector.predict(sandbox_snapshot_trades)

    assert scores.shape[0] == len(sandbox_snapshot_trades)
    assert predictions.shape[0] == len(sandbox_snapshot_trades)
    assert set(predictions).issubset({0, 1})


def test_regression_scores_with_snapshot_data(
    sandbox_snapshot_trades: pd.DataFrame, deterministic_config: Config
) -> None:
    detector = SmartMoneyDetector(config=deterministic_config)
    detector.fit(sandbox_snapshot_trades, price_col="price")

    scores = detector.score(sandbox_snapshot_trades)
    assert scores.shape[0] == len(sandbox_snapshot_trades)
    assert float(np.mean(scores)) == pytest.approx(0.240, rel=0.02)
    assert float(np.median(scores)) == pytest.approx(0.152, rel=0.02)


def test_manual_review_suggestions_respect_request(
    sandbox_snapshot_trades: pd.DataFrame, deterministic_config: Config
) -> None:
    detector = SmartMoneyDetector(config=deterministic_config)
    detector.fit(sandbox_snapshot_trades, price_col="price")

    query_indices, suggested = detector.suggest_manual_reviews(
        sandbox_snapshot_trades, n_queries=5
    )

    assert len(query_indices) == 5
    assert len(suggested) == 5
    assert set(suggested.columns).issuperset({"trade_id", "timestamp", "volume"})


def test_predict_requires_fit(sandbox_snapshot_trades: pd.DataFrame) -> None:
    detector = SmartMoneyDetector()
    with pytest.raises(RuntimeError):
        detector.predict(sandbox_snapshot_trades)


def test_fit_with_empty_dataframe_logs_warning(
    caplog, deterministic_config: Config
) -> None:
    detector = SmartMoneyDetector(config=deterministic_config)
    empty = pd.DataFrame(columns=["timestamp", "volume", "price"])

    with caplog.at_level("WARNING"):
        detector.fit(empty, price_col="price")

    assert "empty trade dataset" in caplog.text.lower()
    assert detector.is_fitted is False


def test_score_with_empty_dataframe_returns_empty_array(
    caplog, deterministic_config: Config
) -> None:
    detector = SmartMoneyDetector(config=deterministic_config)
    empty = pd.DataFrame(columns=["timestamp", "volume", "price"])

    with caplog.at_level("INFO"):
        detector.fit(pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "volume": [1.0]}))
        scores = detector.score(empty)

    assert isinstance(scores, np.ndarray)
    assert scores.size == 0
    assert "no trades provided" in caplog.text.lower()
