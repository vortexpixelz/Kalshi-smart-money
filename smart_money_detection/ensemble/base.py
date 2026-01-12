"""Base classes and protocols for ensemble strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..detectors.base import DetectorProtocol


@runtime_checkable
class EnsembleProtocol(Protocol):
    """Structural protocol describing an ensemble strategy."""

    detectors: Sequence[DetectorProtocol]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EnsembleProtocol":
        ...

    def score(self, X: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        ...

    def predict(self, X: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        ...

    def update(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...

    def get_weights(self) -> np.ndarray:
        ...

    def get_detector_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        ...


class BaseEnsemble(ABC):
    """
    Abstract base class for ensemble weighting methods

    All ensemble weighting strategies should inherit from this class and
    implement the update_weights method.
    """

    def __init__(self, n_detectors: int, name: str = None):
        """
        Initialize base ensemble

        Args:
            n_detectors: Number of base detectors in the ensemble
            name: Name of the ensemble method
        """
        self.n_detectors = n_detectors
        self.name = name or self.__class__.__name__

        # Initialize uniform weights
        self.weights = np.ones(n_detectors) / n_detectors
        self.n_updates = 0

    @abstractmethod
    def update_weights(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ):
        """
        Update ensemble weights based on detector scores and optional feedback

        Args:
            scores: Detector scores of shape (n_detectors, n_samples)
            y_true: Optional ground truth labels
            context: Optional context features

        Returns:
            Updated weights
        """
        pass

    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return self.weights.copy()

    def combine_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Combine detector scores using current weights

        Args:
            scores: Detector scores of shape (n_samples, n_detectors) or (n_detectors, n_samples)

        Returns:
            Combined scores of shape (n_samples,)
        """
        # Ensure scores are in shape (n_samples, n_detectors)
        if scores.shape[0] == self.n_detectors and scores.shape[1] != self.n_detectors:
            scores = scores.T

        # Weighted combination
        combined = np.dot(scores, self.weights)

        return combined

    def reset(self):
        """Reset weights to uniform"""
        self.weights = np.ones(self.n_detectors) / self.n_detectors
        self.n_updates = 0

    def get_state(self) -> Dict[str, Any]:
        """Get ensemble state for serialization"""
        return {
            'weights': self.weights.tolist(),
            'n_updates': self.n_updates,
            'n_detectors': self.n_detectors,
            'name': self.name,
        }

    def set_state(self, state: Dict[str, Any]):
        """Set ensemble state from serialization"""
        self.weights = np.array(state['weights'])
        self.n_updates = state['n_updates']
        self.n_detectors = state['n_detectors']
        self.name = state['name']
