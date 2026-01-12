import pathlib
import sys

import numpy as np
import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_money_detection.detectors.base import BaseDetector
from smart_money_detection.ensemble.ensemble import AnomalyEnsemble
from smart_money_detection.ensemble.weighting import UniformWeighting


class DummyDetector(BaseDetector):
    def __init__(self, scores, name):
        super().__init__(name=name)
        self._scores = np.asarray(scores)
        self.is_fitted_ = True

    def fit(self, X, y=None):  # pragma: no cover - not used in tests
        self.is_fitted_ = True
        return self

    def predict(self, X):  # pragma: no cover - not used in tests
        return (self.score(X) > 0.5).astype(int)

    def score(self, X):
        length = len(X)
        return self._scores[:length]


def manual_normalized_scores(detectors, X):
    normalized = []
    for detector in detectors:
        scores = detector.score(X)
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        normalized.append(scores)
    return np.column_stack(normalized), normalized


def make_ensemble():
    detectors = [
        DummyDetector([0.2, 0.5, 0.9], name="detector_a"),
        DummyDetector([5.0, 5.0, 5.0], name="detector_b"),
        DummyDetector([10.0, 2.0, 6.0], name="detector_c"),
    ]
    ensemble = AnomalyEnsemble(detectors, weighting_method="uniform")
    return ensemble, detectors


def test_score_matches_manual_normalization():
    ensemble, detectors = make_ensemble()
    X = np.zeros((3, 1))

    manual_matrix, _ = manual_normalized_scores(detectors, X)
    manual_scores = ensemble.weighting.combine_scores(manual_matrix)
    ensemble_scores = ensemble.score(X)

    assert np.allclose(ensemble_scores, manual_scores)


class RecordingWeighting(UniformWeighting):
    def __init__(self, n_detectors):
        super().__init__(n_detectors)
        self.last_scores = None

    def update_weights(self, scores, y_true=None, context=None):
        self.last_scores = scores
        return super().update_weights(scores, y_true, context)


def test_update_uses_normalized_scores():
    ensemble, detectors = make_ensemble()
    ensemble.weighting = RecordingWeighting(len(detectors))
    X = np.zeros((3, 1))
    y_true = np.array([1, 0, 1])

    manual_matrix, _ = manual_normalized_scores(detectors, X)

    ensemble.update(X, y_true)

    assert ensemble.weighting.last_scores is not None
    assert np.allclose(ensemble.weighting.last_scores, manual_matrix.T)


def test_detector_contributions_use_normalized_scores():
    ensemble, detectors = make_ensemble()
    X = np.zeros((3, 1))

    _, normalized = manual_normalized_scores(detectors, X)
    weights = ensemble.get_weights()

    contributions = ensemble.get_detector_contributions(X)

    for idx, detector in enumerate(detectors):
        detector_contrib = contributions[detector.name]
        assert np.allclose(detector_contrib['scores'], normalized[idx])
        assert detector_contrib['weight'] == pytest.approx(weights[idx])
        assert np.allclose(
            detector_contrib['weighted_scores'], normalized[idx] * weights[idx]
        )
