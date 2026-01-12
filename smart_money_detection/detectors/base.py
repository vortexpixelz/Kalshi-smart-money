"""Interfaces and base classes for anomaly detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class DetectorProtocol(Protocol):
    """Structural protocol for detector implementations."""

    name: str
    is_fitted_: bool

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> "DetectorProtocol":
        ...

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        ...

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        ...

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

import numpy as np
import pandas as pd

InputData = Union[np.ndarray, pd.DataFrame]

    @abstractmethod
    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> "BaseDetector":
        """
        Fit the detector on training data

class DetectorError(RuntimeError):
    """Raised when a detector cannot complete its requested operation."""


    @abstractmethod
    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores (higher = more anomalous)

    name: str
    n_samples_seen: int = 0
    is_fitted: bool = False


    @abstractmethod
    def _scores_to_predictions(
        self,
        scores: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> np.ndarray:
        """Convert anomaly scores into binary predictions."""

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict anomaly labels (0 = normal, 1 = anomaly)."""
        predictions, _ = self.predict_with_scores(X)
        return predictions

    def predict_with_scores(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return both predictions and anomaly scores in a single call."""

        X_validated = self._validate_input(X)
        scores = self.score(X_validated)
        predictions = self._scores_to_predictions(scores, X_validated)
        return predictions, scores

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the detector and predict on the same data

        Args:
            X: Training data

        Returns:
            Binary predictions array
        """
        self.fit(X)
        return self.predict(X)

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
