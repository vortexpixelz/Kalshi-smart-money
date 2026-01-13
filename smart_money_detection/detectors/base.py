"""Interfaces and base classes for anomaly detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd

InputData = Union[np.ndarray, pd.DataFrame]


@runtime_checkable
class DetectorProtocol(Protocol):
    """Structural protocol for detector implementations."""

    name: str
    is_fitted: bool

    def fit(self, X: InputData, y: Optional[np.ndarray] = None) -> "DetectorProtocol":
        ...

    def predict(self, X: InputData) -> np.ndarray:
        ...

    def score(self, X: InputData) -> np.ndarray:
        ...


class DetectorError(RuntimeError):
    """Raised when a detector cannot complete its requested operation."""


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self.n_samples_seen = 0
        self.is_fitted = False

    def fit(self, X: InputData, y: Optional[np.ndarray] = None) -> "BaseDetector":
        """Fit the detector on training data."""
        array = self._validate_input(X)
        self._fit(array, y)
        self.n_samples_seen = array.shape[0]
        self.is_fitted = True
        return self

    def score(self, X: InputData) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        self._check_is_fitted()
        array = self._validate_input(X)
        return self._score(array)

    @abstractmethod
    def _scores_to_predictions(
        self, scores: np.ndarray, X: Union[np.ndarray, pd.DataFrame, None] = None
    ) -> np.ndarray:
        """Convert anomaly scores into binary predictions."""

    def predict(self, X: InputData) -> np.ndarray:
        """Predict anomaly labels (0 = normal, 1 = anomaly)."""
        predictions, _ = self.predict_with_scores(X)
        return predictions

    def predict_with_scores(self, X: InputData) -> Tuple[np.ndarray, np.ndarray]:
        """Return both predictions and anomaly scores in a single call."""
        array = self._validate_input(X)
        scores = self.score(array)
        predictions = self._scores_to_predictions(scores, array)
        return predictions, scores

    def fit_predict(self, X: InputData) -> np.ndarray:
        """Fit the detector and predict on the same data."""
        self.fit(X)
        return self.predict(X)

    # ------------------------------------------------------------------
    # Internal API for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Subclass-specific fitting logic."""

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Subclass-specific scoring logic."""

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise DetectorError(
                f"{self.name} has not been fitted yet. Call fit() before predict() or score()."
            )

    def check_is_fitted(self) -> None:  # pragma: no cover - deprecated alias
        self._check_is_fitted()

    @staticmethod
    def _to_2d_array(X: InputData) -> np.ndarray:
        """Convert supported input types into a 2-D NumPy array."""
        if isinstance(X, pd.DataFrame):
            array = X.to_numpy(copy=False)
        elif isinstance(X, np.ndarray):
            array = X
        else:
            raise TypeError(f"Expected np.ndarray or pd.DataFrame, received {type(X)!r}")

        if array.ndim == 1:
            array = np.reshape(array, (-1, 1))

        if not np.all(np.isfinite(array)):
            raise DetectorError("Input contains non-finite values")

        return array.astype(float, copy=False)

    def _validate_input(self, X: InputData) -> np.ndarray:
        return self._to_2d_array(X)
