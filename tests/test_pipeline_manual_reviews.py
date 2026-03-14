import numpy as np
import pandas as pd
import pytest

from smart_money_detection.active_learning.query_strategies import QueryByCommittee
from smart_money_detection.pipeline import SmartMoneyDetector


class DummyDetector:
    def __init__(self, name, pred_value=0, score_value=0.5):
        self.name = name
        self.pred_value = pred_value
        self.score_value = score_value
        self.score_calls = 0

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.pred_value, dtype=int)

    def score(self, X):
        self.score_calls += 1
        return np.full(len(X), self.score_value, dtype=float)

    def predict_with_scores(self, X):
        scores = self.score(X)
        return self.predict(X), scores


class DummyEnsemble:
    def __init__(self):
        self.last_context = None
        self.weighting = self

    def combine_scores(self, scores):
        # Simple mean combine for testing
        return scores.mean(axis=1)

    def fit(self, X):
        return self

    def score(self, X, context=None):
        self.last_context = context
        return np.linspace(0, 1, len(X), endpoint=False)


class DummyQueryStrategy:
    def __init__(self):
        self.last_kwargs = None

    def select_queries(self, X, scores, n_queries=None, **kwargs):
        self.last_kwargs = kwargs
        n = n_queries or 1
        return np.arange(min(len(X), n))


def _build_detector():
    detector = SmartMoneyDetector()
    dummy_detectors = [
        DummyDetector("d1", pred_value=0, score_value=0.2),
        DummyDetector("d2", pred_value=1, score_value=0.8),
    ]

    detector.detectors = dummy_detectors
    detector.ensemble = DummyEnsemble()
    detector.ensemble.detectors = dummy_detectors
    detector.ensemble.n_detectors = len(dummy_detectors)
    detector.query_strategy = DummyQueryStrategy()
    detector.is_fitted = True

    return detector


def test_suggest_manual_reviews_uses_context(monkeypatch):
    detector = _build_detector()

    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    trades = pd.DataFrame(
        {
            "volume": np.arange(1, 6, dtype=float),
            "timestamp": timestamps,
        }
    )

    expected_context = np.arange(10, dtype=float).reshape(5, 2)

    def fake_context(ts):
        assert ts.equals(trades["timestamp"])
        return expected_context

    monkeypatch.setattr(detector, "_get_temporal_context", fake_context)

    indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col="volume",
        timestamp_col="timestamp",
        n_queries=2,
    )

    assert len(indices) == 2
    assert len(suggested) == 2
    assert detector.query_strategy.last_kwargs["context"] is expected_context
    assert detector.detectors[0].score_calls == 1
    assert detector.detectors[1].score_calls == 1


def test_suggest_manual_reviews_without_timestamp(monkeypatch):
    detector = _build_detector()

    trades = pd.DataFrame({"volume": np.arange(1, 6, dtype=float)})

    def fail_context(_):  # pragma: no cover - defensive guard
        raise AssertionError("_get_temporal_context should not be called")

    monkeypatch.setattr(detector, "_get_temporal_context", fail_context)

    indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col="volume",
        timestamp_col="timestamp",
        n_queries=3,
    )

    assert len(indices) == 3
    assert len(suggested) == 3
    assert detector.query_strategy.last_kwargs["context"] is None


def test_suggest_manual_reviews_missing_volume_column():
    detector = _build_detector()

    trades = pd.DataFrame({"other": np.arange(5)})

    with pytest.raises(ValueError, match="Required volume column 'volume' is missing"):
        detector.suggest_manual_reviews(trades)


def test_qbc_deterministic_ordering_on_ties():
    strategy = QueryByCommittee(batch_size=3)

    X = np.zeros((5, 1))
    scores = np.zeros(5)
    committee_predictions = np.ones((5, 2), dtype=int)

    indices = strategy.select_queries(
        X,
        scores,
        committee_predictions=committee_predictions,
        committee_scores=np.ones_like(committee_predictions, dtype=float),
    )

    assert indices.tolist() == [0, 1, 2]
