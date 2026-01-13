import numpy as np
import pandas as pd

from smart_money_detection.active_learning import QueryByCommittee
from smart_money_detection.pipeline import SmartMoneyDetector


class DummyDetector:
    def __init__(self, name, pred_value=0, score_value=0.5):
        self.name = name
        self.pred_value = pred_value
        self.score_value = score_value

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.pred_value, dtype=int)

    def score(self, X):
        return np.full(len(X), self.score_value, dtype=float)


class SequenceDetector:
    def __init__(self, name, predictions, scores):
        self.name = name
        self._predictions = np.asarray(predictions, dtype=int)
        self._scores = np.asarray(scores, dtype=float)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self._predictions[: len(X)]

    def score(self, X):
        return self._scores[: len(X)]


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
    detector.ensemble.detectors = dummy_detectors
    detector.ensemble.n_detectors = len(dummy_detectors)
    detector.ensemble.weighting.weights = np.ones(len(dummy_detectors)) / len(
        dummy_detectors
    )
    detector.query_strategy = DummyQueryStrategy()
    detector.is_fitted = True

    return detector


def _build_qbc_detector():
    detector = SmartMoneyDetector()
    dummy_detectors = [
        SequenceDetector("d1", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        SequenceDetector("d2", [0, 1, 1, 0], [0.2, 0.7, 0.6, 0.3]),
    ]

    detector.detectors = dummy_detectors
    detector.ensemble.detectors = dummy_detectors
    detector.ensemble.n_detectors = len(dummy_detectors)
    detector.ensemble.weighting.weights = np.ones(len(dummy_detectors)) / len(
        dummy_detectors
    )
    detector.query_strategy = QueryByCommittee(batch_size=2)
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

    def fake_context(trades_arg, timestamp_col, use_temporal_context):
        assert trades_arg is trades
        assert timestamp_col == "timestamp"
        assert use_temporal_context is True
        return expected_context

    monkeypatch.setattr(detector.data_service, "build_temporal_context", fake_context)

    indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col="volume",
        timestamp_col="timestamp",
        n_queries=2,
    )

    assert len(indices) == 2
    assert len(suggested) == 2
    assert detector.query_strategy.last_kwargs["committee_predictions"].shape == (5, 2)


def test_suggest_manual_reviews_without_timestamp(monkeypatch):
    detector = _build_detector()

    trades = pd.DataFrame({"volume": np.arange(1, 6, dtype=float)})

    def fail_context(*_):  # pragma: no cover - defensive guard
        raise AssertionError("_get_temporal_context should not be called")

    monkeypatch.setattr(detector.data_service, "build_temporal_context", fail_context)

    indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col="volume",
        timestamp_col="timestamp",
        n_queries=3,
    )

    assert len(indices) == 3
    assert len(suggested) == 3


def test_qbc_selection_with_temporal_context(monkeypatch):
    detector = _build_qbc_detector()
    trades = pd.DataFrame(
        {
            "volume": np.array([1.0, 2.0, 3.0, 4.0]),
            "timestamp": pd.date_range("2024-02-01", periods=4, freq="h"),
        }
    )

    expected_context = np.arange(8, dtype=float).reshape(4, 2)

    def fake_context(trades_arg, timestamp_col, use_temporal_context):
        assert trades_arg is trades
        assert timestamp_col == "timestamp"
        assert use_temporal_context is True
        return expected_context

    monkeypatch.setattr(detector.data_service, "build_temporal_context", fake_context)

    indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col="volume",
        timestamp_col="timestamp",
        n_queries=2,
    )

    assert indices.tolist() == [1, 3]
    assert suggested["volume"].tolist() == [2.0, 4.0]


def test_qbc_selection_without_temporal_context(monkeypatch):
    detector = _build_qbc_detector()
    trades = pd.DataFrame({"volume": np.array([1.0, 2.0, 3.0, 4.0])})

    def fail_context(*_):  # pragma: no cover - defensive guard
        raise AssertionError("build_temporal_context should not be called")

    monkeypatch.setattr(detector.data_service, "build_temporal_context", fail_context)

    indices, suggested = detector.suggest_manual_reviews(
        trades,
        volume_col="volume",
        timestamp_col="timestamp",
        n_queries=2,
    )

    assert indices.tolist() == [1, 3]
    assert suggested["volume"].tolist() == [2.0, 4.0]
