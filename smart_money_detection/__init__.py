"""
Smart Money Detection - Ensemble anomaly detection for prediction markets

This package implements state-of-the-art ensemble methods for detecting informed
traders ("smart money") in prediction markets with minimal labeled data.
"""

__version__ = "0.1.0"
__author__ = "VirtualPixelz"

from .config import Config, load_config
from .pipeline import SmartMoneyDetector

__all__ = ["SmartMoneyDetector", "Config", "load_config"]
