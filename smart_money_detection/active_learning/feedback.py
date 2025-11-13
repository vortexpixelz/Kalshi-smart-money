"""
Feedback management for human-in-the-loop learning

Handles collection, storage, and integration of manual reviews into
the ensemble detection system.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json


class FeedbackManager:
    """
    Manages human feedback for active learning

    Features:
        - Store manual review results
        - Track review statistics and performance
        - Optimize decision thresholds for F1 score
        - Provide feedback for ensemble weight updates
    """

    def __init__(self, optimize_f1: bool = True):
        """
        Initialize feedback manager

        Args:
            optimize_f1: If True, optimize decision threshold for F1 score
        """
        self.optimize_f1 = optimize_f1

        # Storage
        self.feedback_data = []  # List of feedback dictionaries
        self.labeled_indices = set()  # Track which samples have been labeled

        # Statistics
        self.n_positive = 0
        self.n_negative = 0
        self.n_total = 0

        # Optimal threshold for F1
        self.optimal_threshold = 0.5

    def add_feedback(
        self,
        sample_id: Any,
        y_true: int,
        y_pred: Optional[int] = None,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add manual review feedback

        Args:
            sample_id: Unique identifier for the sample
            y_true: True label (0 = normal, 1 = anomaly)
            y_pred: Predicted label (optional)
            score: Anomaly score (optional)
            metadata: Additional metadata (trade info, context, etc.)
        """
        feedback = {
            'sample_id': sample_id,
            'y_true': y_true,
            'y_pred': y_pred,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
        }

        self.feedback_data.append(feedback)
        self.labeled_indices.add(sample_id)

        # Update statistics
        if y_true == 1:
            self.n_positive += 1
        else:
            self.n_negative += 1
        self.n_total += 1

        # Update optimal threshold if F1 optimization enabled
        if self.optimize_f1 and self.n_total >= 10:
            self._update_optimal_threshold()

    def add_batch_feedback(
        self,
        sample_ids: List[Any],
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add multiple feedback samples at once

        Args:
            sample_ids: List of sample identifiers
            y_true: True labels
            y_pred: Predicted labels (optional)
            scores: Anomaly scores (optional)
            metadata: List of metadata dictionaries (optional)
        """
        n_samples = len(sample_ids)

        if y_pred is None:
            y_pred = [None] * n_samples
        if scores is None:
            scores = [None] * n_samples
        if metadata is None:
            metadata = [None] * n_samples

        for i in range(n_samples):
            self.add_feedback(
                sample_ids[i],
                y_true[i],
                y_pred[i] if y_pred[i] is not None else None,
                scores[i] if scores[i] is not None else None,
                metadata[i],
            )

    def get_labeled_data(self) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """
        Get all labeled data

        Returns:
            Tuple of (sample_ids, labels, scores)
        """
        if len(self.feedback_data) == 0:
            return [], np.array([]), np.array([])

        sample_ids = [f['sample_id'] for f in self.feedback_data]
        labels = np.array([f['y_true'] for f in self.feedback_data])
        scores = np.array(
            [f['score'] if f['score'] is not None else np.nan for f in self.feedback_data],
            dtype=float,
        )

        return sample_ids, labels, scores

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feedback statistics

        Returns:
            Dictionary with statistics
        """
        if self.n_total == 0:
            return {
                'n_total': 0,
                'n_positive': 0,
                'n_negative': 0,
                'positive_rate': 0.0,
                'optimal_threshold': self.optimal_threshold,
            }

        # Basic stats
        stats = {
            'n_total': self.n_total,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative,
            'positive_rate': self.n_positive / self.n_total,
            'optimal_threshold': self.optimal_threshold,
        }

        # Performance metrics if predictions available
        if any(f['y_pred'] is not None for f in self.feedback_data):
            _, labels, scores = self.get_labeled_data()
            predictions = np.array(
                [f['y_pred'] if f['y_pred'] is not None else 0 for f in self.feedback_data]
            )

            # Compute metrics
            tp = ((predictions == 1) & (labels == 1)).sum()
            fp = ((predictions == 1) & (labels == 0)).sum()
            tn = ((predictions == 0) & (labels == 0)).sum()
            fn = ((predictions == 0) & (labels == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            stats.update(
                {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': (tp + tn) / self.n_total,
                    'tp': int(tp),
                    'fp': int(fp),
                    'tn': int(tn),
                    'fn': int(fn),
                }
            )

        return stats

    def _update_optimal_threshold(self):
        """
        Find optimal threshold for F1 score maximization

        Uses grid search over [0.3, 0.7] with 0.01 increments
        """
        _, labels, scores = self.get_labeled_data()

        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() < 5:  # Need minimum samples with scores
            return

        labels = labels[valid_mask]
        scores = scores[valid_mask]

        best_f1 = 0
        best_threshold = 0.5

        # Grid search
        for threshold in np.arange(0.3, 0.71, 0.01):
            predictions = (scores >= threshold).astype(int)

            tp = ((predictions == 1) & (labels == 1)).sum()
            fp = ((predictions == 1) & (labels == 0)).sum()
            fn = ((predictions == 0) & (labels == 1)).sum()

            if (tp + fp) > 0 and (tp + fn) > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        self.optimal_threshold = best_threshold

    def get_optimal_threshold(self) -> float:
        """Get optimal decision threshold for F1 score"""
        return self.optimal_threshold

    def export_feedback(self, filepath: str):
        """
        Export feedback data to JSON file

        Args:
            filepath: Path to save feedback data
        """
        with open(filepath, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

    def load_feedback(self, filepath: str):
        """
        Load feedback data from JSON file

        Args:
            filepath: Path to load feedback data from
        """
        with open(filepath, 'r') as f:
            self.feedback_data = json.load(f)

        # Rebuild statistics
        self.labeled_indices = {f['sample_id'] for f in self.feedback_data}
        self.n_positive = sum(1 for f in self.feedback_data if f['y_true'] == 1)
        self.n_negative = sum(1 for f in self.feedback_data if f['y_true'] == 0)
        self.n_total = len(self.feedback_data)

        # Update optimal threshold
        if self.optimize_f1 and self.n_total >= 10:
            self._update_optimal_threshold()

    def is_labeled(self, sample_id: Any) -> bool:
        """Check if sample has been labeled"""
        return sample_id in self.labeled_indices

    def get_unlabeled_mask(self, sample_ids: List[Any]) -> np.ndarray:
        """
        Get boolean mask of unlabeled samples

        Args:
            sample_ids: List of sample identifiers

        Returns:
            Boolean array (True = unlabeled)
        """
        return np.array([sid not in self.labeled_indices for sid in sample_ids])

    def get_recent_feedback(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get n most recent feedback samples

        Args:
            n: Number of recent samples to return

        Returns:
            List of feedback dictionaries
        """
        return self.feedback_data[-n:] if len(self.feedback_data) >= n else self.feedback_data

    def clear(self):
        """Clear all feedback data"""
        self.feedback_data = []
        self.labeled_indices = set()
        self.n_positive = 0
        self.n_negative = 0
        self.n_total = 0
        self.optimal_threshold = 0.5
