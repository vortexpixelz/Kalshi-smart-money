import numpy as np

from smart_money_detection.active_learning import QueryByCommittee


def test_qbc_deterministic_order_on_ties():
    qbc = QueryByCommittee(batch_size=3)
    committee_predictions = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
        ]
    )
    scores = np.zeros(committee_predictions.shape[0])

    indices = qbc.select_queries(
        np.zeros((committee_predictions.shape[0], 1)),
        scores,
        n_queries=3,
        committee_predictions=committee_predictions,
        committee_scores=None,
    )

    assert indices.tolist() == [0, 1, 2]
