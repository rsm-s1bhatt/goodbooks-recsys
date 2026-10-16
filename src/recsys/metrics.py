"""Top-K ranking metrics computed against each user's held-out liked books."""
import numpy as np


def top_k(scores, seen_rows, k=10):
    scores = scores.copy()
    for i, row in enumerate(seen_rows):
        scores[i, row] = -np.inf  # never recommend a book the user already rated in train
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def evaluate(recs, relevant, k=10):
    """recs: (n_users, k) item indices. relevant: list of sets of held-out liked items."""
    disc = 1 / np.log2(np.arange(2, k + 2))
    p, r, n, h = [], [], [], []
    for rec, rel in zip(recs, relevant):
        hits = np.array([i in rel for i in rec[:k]], dtype=float)
        p.append(hits.sum() / k)
        r.append(hits.sum() / len(rel))
        ideal = disc[: min(len(rel), k)].sum()
        n.append((hits * disc).sum() / ideal)
        h.append(float(hits.any()))
    return {f"precision@{k}": np.mean(p), f"recall@{k}": np.mean(r),
            f"ndcg@{k}": np.mean(n), f"hit_rate@{k}": np.mean(h)}
