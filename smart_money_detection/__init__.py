"""
Smart Money Detection - Ensemble anomaly detection for prediction markets

This package implements state-of-the-art ensemble methods for detecting informed
traders ("smart money") in prediction markets with minimal labeled data.
"""

__version__ = "0.1.0"
__author__ = "VirtualPixelz"

from .config import Config, load_config

__all__ = ["SmartMoneyDetector", "Config", "load_config"]


def __getattr__(name: str):
    if name == "SmartMoneyDetector":
        from .pipeline import SmartMoneyDetector

        return SmartMoneyDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
