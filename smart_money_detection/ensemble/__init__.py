"""
Ensemble weighting and combination methods
"""

from .base import BaseEnsemble, EnsembleProtocol
from .weighting import (
    UniformWeighting,
    MultiplicativeWeightsUpdate,
    ThompsonSamplingWeighting,
    UCBWeighting,
)
from .ensemble import AnomalyEnsemble, ProbabilityCalibrator

__all__ = [
    "BaseEnsemble",
    "EnsembleProtocol",
    "UniformWeighting",
    "MultiplicativeWeightsUpdate",
    "ThompsonSamplingWeighting",
    "UCBWeighting",
    "AnomalyEnsemble",
    "ProbabilityCalibrator",
]
