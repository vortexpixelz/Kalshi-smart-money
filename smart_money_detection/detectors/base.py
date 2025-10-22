"""
Base class for anomaly detectors
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


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
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
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
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomaly)

        Args:
            X: Data to predict

        Returns:
            Binary predictions array
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
