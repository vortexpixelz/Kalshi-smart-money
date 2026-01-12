from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_money_detection.pipeline import SmartMoneyDetector
from smart_money_detection.detectors.base import BaseDetector
from smart_money_detection.ensemble import AnomalyEnsemble


class PerfectDetector(BaseDetector):
    def __init__(self):
        super().__init__(name="perfect")

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return (self.score(X) >= 0.5).astype(int)

    def score(self, X):
        self.check_is_fitted()
        X = self._validate_input(X)
        return X.flatten()


class InverseDetector(BaseDetector):
    def __init__(self):
        super().__init__(name="inverse")

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return (self.score(X) >= 0.5).astype(int)

    def score(self, X):
        self.check_is_fitted()
        X = self._validate_input(X)
        return 1 - X.flatten()


def test_optimize_weights_improves_f1_with_gradient():
    np.random.seed(42)
    n_samples = 20
    labels = np.array([0, 1] * (n_samples // 2))

    trades = pd.DataFrame(
        {
            "volume": labels.astype(float),
            "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="min"),
        }
    )

    detector = SmartMoneyDetector()
    custom_detectors = [PerfectDetector(), InverseDetector()]
    detector.detectors = custom_detectors
    detector.ensemble = AnomalyEnsemble(detectors=custom_detectors, weighting_method="uniform")

    detector.fit(trades, volume_col="volume", timestamp_col="timestamp")
    detector.score(trades, volume_col="volume", timestamp_col="timestamp")

    sample_ids = trades.index.tolist()
    detector.add_feedback(sample_ids, labels, trades=trades, volume_col="volume", update_weights=False)

    cached_scores = np.vstack([detector._detector_score_cache[sid] for sid in sample_ids])
    baseline_weights = detector.ensemble.get_weights()
    baseline_scores = cached_scores @ baseline_weights
    baseline_predictions = (baseline_scores >= 0.5).astype(int)
    baseline_f1 = f1_score(labels, baseline_predictions)

    optimized_weights, optimized_metric = detector.optimize_weights(method="gradient", n_iterations=50)

    assert optimized_weights[0] > optimized_weights[1]
    assert optimized_metric >= baseline_f1
    assert optimized_metric > 0.9
