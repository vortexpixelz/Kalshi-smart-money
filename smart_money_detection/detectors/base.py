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


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors

    All detectors should inherit from this class and implement the fit, predict,
    and score methods.
    """

    def __init__(self, name: str = None):
        """
        Initialize base detector

        Args:
            name: Name of the detector
        """
        self.name = name or self.__class__.__name__
        self.is_fitted_ = False
        self.n_samples_seen_ = 0

    @abstractmethod
    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> "BaseDetector":
        """
        Fit the detector on training data

        Args:
            X: Training data
            y: Optional labels (for semi-supervised methods)

        Returns:
            self
        """
        pass

    @abstractmethod
    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores (higher = more anomalous)

        Args:
            X: Data to score

        Returns:
            Anomaly scores array
        """
        pass

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

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Alias for score method (scikit-learn compatibility)

        Args:
            X: Data to score

        Returns:
            Anomaly scores array
        """
        return self.score(X)

    def check_is_fitted(self):
        """Check if the detector has been fitted"""
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call fit() before predict() or score()."
            )

    def _validate_input(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Validate input data

        Args:
            X: Input data

        Returns:
            Validated input data
        """
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X
        else:
            raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(X)}")
