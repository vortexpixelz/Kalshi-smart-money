"""Interfaces and base classes for anomaly detectors."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd

InputData = Union[np.ndarray, pd.DataFrame]


class DetectorError(RuntimeError):
    """Raised when a detector cannot complete its requested operation."""


@dataclass
class DetectorState:
    name: str
    n_samples_seen: int = 0
    is_fitted: bool = False


@runtime_checkable
class DetectorProtocol(Protocol):
    """Structural protocol for detector implementations."""

    name: str
    is_fitted_: bool

    def fit(
        self, X: InputData, y: Optional[np.ndarray] = None
    ) -> "DetectorProtocol":
        ...

    def predict(self, X: InputData) -> np.ndarray:
        ...

    def score(self, X: InputData) -> np.ndarray:
        ...


class BaseDetector(ABC):
    """Base detector implementing shared fit/score/predict logic."""

    def __init__(self, name: str) -> None:
        self.state = DetectorState(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def name(self) -> str:  # pragma: no cover - trivial proxy
        return self.state.name

    @property
    def is_fitted_(self) -> bool:  # pragma: no cover - trivial proxy
        return self.state.is_fitted

    def fit(self, X: InputData, y: Optional[np.ndarray] = None) -> "BaseDetector":
        X_validated = self._validate_input(X)
        self._fit(X_validated, y)
        self.state.is_fitted = True
        self.state.n_samples_seen += X_validated.shape[0]
        return self

    def score(self, X: InputData) -> np.ndarray:
        self._check_is_fitted()
        X_validated = self._validate_input(X)
        return self._score(X_validated)

    def predict(self, X: InputData) -> np.ndarray:
        predictions, _ = self.predict_with_scores(X)
        return predictions

    def predict_with_scores(self, X: InputData) -> tuple[np.ndarray, np.ndarray]:
        X_validated = self._validate_input(X)
        scores = self.score(X_validated)
        predictions = self._scores_to_predictions(scores, X_validated)
        return predictions, scores

    def fit_predict(self, X: InputData) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def _validate_input(self, X: InputData) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            values = X.values
        else:
            values = np.asarray(X)

        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values.astype(float)

    def _check_is_fitted(self) -> None:
        if not self.state.is_fitted:
            raise DetectorError(
                f"{self.state.name} has not been fitted yet. Call fit() before predict() or score()."
            )

    # Backwards-compatible public alias
    def check_is_fitted(self) -> None:  # pragma: no cover - deprecated surface
        self._check_is_fitted()

    @abstractmethod
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Subclass-specific fitting logic."""

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Subclass-specific scoring logic."""

    @abstractmethod
    def _scores_to_predictions(
        self, scores: np.ndarray, X: Optional[InputData] = None
    ) -> np.ndarray:
        """Convert anomaly scores into binary predictions."""


__all__ = ["BaseDetector", "DetectorProtocol", "DetectorError", "InputData"]
