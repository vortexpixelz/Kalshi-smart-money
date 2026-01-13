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


@runtime_checkable
class DetectorProtocol(Protocol):
    """Structural protocol for detector implementations."""

    name: str
    is_fitted_: bool

    def fit(self, X: InputData, y: Optional[np.ndarray] = None) -> "DetectorProtocol":
        ...

    def predict(self, X: InputData) -> np.ndarray:
        ...

    def score(self, X: InputData) -> np.ndarray:
        ...


@dataclass
class DetectorState:
    """State container for detector metadata."""

    name: str
    n_samples_seen: int = 0
    is_fitted: bool = False


class BaseDetector(ABC):
    """Typed base class for anomaly detectors with shared validation and logging."""

    def __init__(self, name: str, *, logger: Optional[logging.Logger] = None) -> None:
        self.state = DetectorState(name=name)
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")

    @property
    def name(self) -> str:
        return self.state.name

    @property
    def is_fitted_(self) -> bool:
        return self.state.is_fitted

    def fit(self, X: InputData, y: Optional[np.ndarray] = None) -> "BaseDetector":
        """Fit the detector on training data."""
        try:
            array = self._to_2d_array(X)
            self._fit(array, y)
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to fit detector %s", self.name)
            raise DetectorError(f"Failed to fit detector {self.name}: {exc}") from exc

        self.state.n_samples_seen += array.shape[0]
        self.state.is_fitted = True
        self.logger.debug("Fitted detector %s on %d samples", self.name, array.shape[0])
        return self

    def score(self, X: InputData) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        self._check_is_fitted()
        try:
            scores = self._compute_scores(X)
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to score with detector %s", self.name)
            raise DetectorError(f"Failed to score with detector {self.name}: {exc}") from exc

        return scores

    def predict(self, X: InputData) -> np.ndarray:
        """Predict anomaly labels (0 = normal, 1 = anomaly)."""
        predictions, _ = self.predict_with_scores(X)
        return predictions

    def predict_with_scores(self, X: InputData) -> tuple[np.ndarray, np.ndarray]:
        """Return both predictions and anomaly scores in a single call."""
        self._check_is_fitted()
        try:
            array = self._to_2d_array(X)
            scores = np.asarray(self._score(array), dtype=float).reshape(-1)
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to score with detector %s", self.name)
            raise DetectorError(f"Failed to score with detector {self.name}: {exc}") from exc

        predictions = self._scores_to_predictions(scores, array)
        return predictions, scores

    def fit_predict(self, X: InputData) -> np.ndarray:
        """Fit the detector and predict on the same data."""
        self.fit(X)
        return self.predict(X)

    def _compute_scores(self, X: InputData) -> np.ndarray:
        array = self._to_2d_array(X)
        scores = self._score(array)
        return np.asarray(scores, dtype=float).reshape(-1)

        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values.astype(float)

    @abstractmethod
    def _scores_to_predictions(
        self,
        scores: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert anomaly scores into binary predictions."""

    def _check_is_fitted(self) -> None:
        if not self.state.is_fitted:
            raise DetectorError(
                f"{self.state.name} has not been fitted yet. Call fit() before predict() or score()."
            )

    def check_is_fitted(self) -> None:  # pragma: no cover - deprecated surface
        """Deprecated alias for `_check_is_fitted`."""
        self._check_is_fitted()

    @staticmethod
    def _to_2d_array(X: InputData) -> np.ndarray:
        """Convert supported input types into a 2-D NumPy array."""
        if isinstance(X, pd.DataFrame):
            array = X.to_numpy(copy=False)
        else:
            array = np.asarray(X)

    @abstractmethod
    def _scores_to_predictions(
        self, scores: np.ndarray, X: Optional[InputData] = None
    ) -> np.ndarray:
        """Convert anomaly scores into binary predictions."""

        if array.ndim != 2:
            raise DetectorError("Input must be a 2-D array")

        if not np.all(np.isfinite(array)):
            raise DetectorError("Input contains non-finite values")

__all__ = ["BaseDetector", "DetectorProtocol", "DetectorError", "InputData"]
