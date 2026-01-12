import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_money_detection.active_learning.feedback import FeedbackManager


def test_batch_feedback_matches_individual_updates():
    sample_ids = [f"sample-{i}" for i in range(5)]
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    scores = np.linspace(0.1, 0.9, len(sample_ids))
    metadata = [{"index": i} for i in range(len(sample_ids))]

    batch_manager = FeedbackManager(optimize_f1=False)
    single_manager = FeedbackManager(optimize_f1=False)

    batch_manager.add_batch_feedback(sample_ids, y_true, y_pred, scores, metadata)

    for sid, label, pred, score, meta in zip(sample_ids, y_true, y_pred, scores, metadata):
        single_manager.add_feedback(sid, int(label), int(pred), float(score), meta)

    assert batch_manager.n_total == single_manager.n_total
    assert batch_manager.n_positive == single_manager.n_positive
    assert batch_manager.n_negative == single_manager.n_negative
    assert batch_manager.labeled_indices == single_manager.labeled_indices

    assert len(batch_manager.feedback_data) == len(single_manager.feedback_data)
    for batch_record, single_record in zip(
        batch_manager.feedback_data, single_manager.feedback_data
    ):
        assert batch_record["sample_id"] == single_record["sample_id"]
        assert batch_record["y_true"] == single_record["y_true"]
        assert batch_record["y_pred"] == single_record["y_pred"]
        assert batch_record["score"] == single_record["score"]
        assert batch_record["metadata"] == single_record["metadata"]
        assert isinstance(batch_record["timestamp"], str)
