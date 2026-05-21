"""
Base anomaly detectors for smart money detection
"""

from .base import BaseDetector, DetectorProtocol
from .zscore import ZScoreDetector
from .iqr import IQRDetector
from .percentile import PercentileDetector
from .volume import RelativeVolumeDetector

__all__ = [
    "BaseDetector",
    "DetectorProtocol",
    "ZScoreDetector",
    "IQRDetector",
    "PercentileDetector",
    "RelativeVolumeDetector",
]
