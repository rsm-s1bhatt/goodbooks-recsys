import numpy as np
import pandas as pd
from recsys.data import holdout_split
from recsys.metrics import evaluate, top_k


def test_split_has_no_overlap_and_keeps_every_rating():
    r = pd.DataFrame({"user_id": np.repeat([1, 2], 10), "book_id": list(range(1, 11)) * 2, "rating": 5})
    tr, te = holdout_split(r)
    assert len(tr) + len(te) == len(r)
    assert tr.merge(te, on=["user_id", "book_id"]).empty
    assert (te.groupby("user_id").size() == 2).all()


def test_top_k_never_returns_seen_items():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    recs = top_k(scores, [np.array([0, 1])], k=2)
    assert set(recs[0]) == {2, 3}
    assert recs[0][0] == 3  # higher score first


def test_perfect_ranking_scores_one():
    m = evaluate(np.array([[1, 2]]), [{1, 2}], k=2)
    assert m["ndcg@2"] == 1.0 and m["precision@2"] == 1.0 and m["recall@2"] == 1.0


def test_miss_scores_zero():
    m = evaluate(np.array([[5, 6]]), [{1}], k=2)
    assert m["ndcg@2"] == 0.0 and m["hit_rate@2"] == 0.0
