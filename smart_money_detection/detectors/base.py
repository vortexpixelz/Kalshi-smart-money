"""Shared utilities and base abstractions for anomaly detectors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

import numpy as np
import pandas as pd

InputData = Union[np.ndarray, pd.DataFrame]


class DetectorError(RuntimeError):
    """Raised when a detector cannot complete its requested operation."""


class SupportsPredict(Protocol):
    """Protocol used for type-checking detector predictions."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly labels for *X*."""


@dataclass
class DetectorState:
    """Lightweight container for detector metadata."""

    name: str
    n_samples_seen: int = 0
    is_fitted: bool = False


class BaseDetector(ABC):
    """Abstract base class that enforces consistent detector behaviour."""

    def __init__(self, name: Optional[str] = None, *, logger: Optional[logging.Logger] = None) -> None:
        self.state = DetectorState(name=name or self.__class__.__name__)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.state.name}")
        # Backwards-compatible attributes referenced in older code paths.
        self.is_fitted_: bool = False
        self.n_samples_seen_: int = 0

    @property
    def name(self) -> str:
        """Human-readable detector name."""

        return self.state.name

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------
    def fit(self, X: InputData, y: Optional[np.ndarray] = None) -> "BaseDetector":
        """Fit the detector on *X* with optional labels *y*."""

        array = self._to_2d_array(X)
        try:
            self._fit(array, y)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.logger.exception("%s failed during fit", self.state.name)
            raise DetectorError(f"{self.state.name} failed to fit") from exc

        self.state.is_fitted = True
        self.state.n_samples_seen = array.shape[0]
        self.is_fitted_ = True
        self.n_samples_seen_ = array.shape[0]
        return self

    def score(self, X: InputData) -> np.ndarray:
        """Return anomaly scores for *X*."""

        self._check_is_fitted()
        array = self._to_2d_array(X)
        try:
            scores = self._score(array)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.logger.exception("%s failed during score", self.state.name)
            raise DetectorError(f"{self.state.name} failed to score") from exc

        return np.asarray(scores, dtype=float)

    def predict(self, X: InputData) -> np.ndarray:
        """Return binary anomaly decisions for *X*."""

        scores = self.score(X)
        return (scores > 0.0).astype(int)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def fit_predict(self, X: InputData) -> np.ndarray:
        """Fit the detector on *X* and return predictions for the same data."""

        return self.fit(X).predict(X)

    def decision_function(self, X: InputData) -> np.ndarray:
        """Alias for :meth:`score` for scikit-learn style compatibility."""

        return self.score(X)

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
        if not self.state.is_fitted:
            raise DetectorError(
                f"{self.state.name} has not been fitted yet. Call fit() before predict() or score()."
            )

    # Backwards-compatible public alias
    def check_is_fitted(self) -> None:  # pragma: no cover - deprecated surface
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
