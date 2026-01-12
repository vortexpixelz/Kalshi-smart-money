"""Services wrapping detector and ensemble coordination."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..detectors.base import DetectorProtocol
from ..ensemble.base import EnsembleProtocol


class DetectionService:
    """Coordinate detectors, ensembles, and feedback updates."""

    def __init__(self, ensemble: EnsembleProtocol):
        self.ensemble = ensemble

    @property
    def detectors(self) -> Sequence[DetectorProtocol]:
        detectors = getattr(self.ensemble, "detectors", ())
        return tuple(detectors)

    def fit(self, X: np.ndarray) -> None:
        """Fit detectors and ensemble on historical data."""
        self.ensemble.fit(X)

    def score(
        self, X: np.ndarray, context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.ensemble.score(X, context)

    def predict(
        self, X: np.ndarray, context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.ensemble.predict(X, context)

    def update(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.ensemble.update(X, y_true, context)

    def committee_outputs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        scores = []
        for detector in self.detectors:
            predictions.append(detector.predict(X))
            scores.append(detector.score(X))
        if not predictions:
            return np.empty((X.shape[0], 0)), np.empty((X.shape[0], 0))
        return np.column_stack(predictions), np.column_stack(scores)

    def get_weights(self) -> np.ndarray:
        return self.ensemble.get_weights()

    def get_detector_contributions(self, X: np.ndarray):
        return self.ensemble.get_detector_contributions(X)
