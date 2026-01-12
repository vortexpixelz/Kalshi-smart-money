"""Service layer abstractions for the smart money detection pipeline."""

from .data_ingestion import DataIngestionService
from .detection import DetectionService

__all__ = ["DataIngestionService", "DetectionService"]
