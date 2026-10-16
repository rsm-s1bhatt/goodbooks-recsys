"""End-to-end experiment: tune on a validation split, then report test metrics and a cold-start test."""
import json, time
import numpy as np
from recsys.data import load, holdout_split, like_matrix, seen_matrix, LIKE
from recsys.models import Popularity, ItemKNN, ALS, TagContent, Hybrid
from recsys.metrics import top_k, evaluate

K, BATCH = 10, 2000


def run_eval(model, X_train, seen, test, users, n_items):
    rel = test[test.rating >= LIKE].groupby("user_id")["book_id"].apply(lambda s: set(s - 1))
    users = [u for u in users if (u + 1) in rel.index]
    all_recs, rels = [], []
    for s in range(0, len(users), BATCH):
        b = users[s:s + BATCH]
        sc = model.score(b, X_train)
        seen_rows = [seen.indices[seen.indptr[u]:seen.indptr[u + 1]] for u in b]
        all_recs.append(top_k(sc, seen_rows, K))
        rels += [rel[u + 1] for u in b]
    recs = np.vstack(all_recs)
    m = evaluate(recs, rels, K)
    m["catalog_coverage"] = len(np.unique(recs)) / n_items
    m["users_evaluated"] = len(users)
    return m, recs


def main():
    t0 = time.time()
    ratings, books, tags = load()
    nU, nI = ratings.user_id.max(), ratings.book_id.max()
    train, test = holdout_split(ratings)
    tr2, val = holdout_split(train, seed=7)          # validation split carved out of train only
    rng = np.random.default_rng(0)
    val_users = rng.choice(nU, 5000, replace=False).tolist()

    # --- tune ALS factors/reg and hybrid weight on validation ---
    Xv, Sv = like_matrix(tr2, nU, nI), seen_matrix(tr2, nU, nI)
    tuning = []
    best = None
    for f in (32, 64, 128):
        for reg in (0.01, 0.1):
            m, _ = run_eval(ALS(factors=f, reg=reg).fit(Xv), Xv, Sv, val, val_users, nI)
            tuning.append({"factors": f, "reg": reg, "val_ndcg@10": m["ndcg@10"]})
            if best is None or m["ndcg@10"] > best[2]:
                best = (f, reg, m["ndcg@10"])
    content_v = TagContent(tags, nI).fit(Xv)
    als_v = ALS(factors=best[0], reg=best[1]).fit(Xv)
    best_w = max((0.5, 0.7, 0.85, 1.0),
                 key=lambda w: run_eval(Hybrid(als_v, content_v, w), Xv, Sv, val, val_users, nI)[0]["ndcg@10"])

    # --- final fit on full train, evaluate on test ---
    X, S = like_matrix(train, nU, nI), seen_matrix(train, nU, nI)
    all_users = list(range(nU))
    als, content = ALS(factors=best[0], reg=best[1]).fit(X), TagContent(tags, nI).fit(X)
    models = {"Popularity": Popularity().fit(X), "Item-item kNN": ItemKNN(100).fit(X),
              "ALS matrix factorization": als, "Tag content (TF-IDF)": content,
              f"Hybrid (ALS {best_w:.2f} + content)": Hybrid(als, content, best_w)}
    results = {}
    pop_rank = np.argsort(np.argsort(-np.asarray(X.sum(0)).ravel()))  # 0 = most popular
    for name, mdl in models.items():
        m, recs = run_eval(mdl, X, S, test, all_users, nI)
        m["median_popularity_rank_of_recs"] = float(np.median(pop_rank[recs]))
        results[name] = m
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)

    # --- cold start: users who have liked only 3 books so far ---
    cs_users = rng.choice(nU, 5000, replace=False)
    liked_tr = train[train.rating >= LIKE]
    keep = liked_tr[liked_tr.user_id.isin(cs_users + 1)].sample(frac=1.0, random_state=1)
    keep = keep[keep.groupby("user_id").cumcount() < 3]
    rest = train[~train.user_id.isin(cs_users + 1)]
    cs_train = __import__("pandas").concat([rest, keep])
    Xc, Sc = like_matrix(cs_train, nU, nI), seen_matrix(cs_train, nU, nI)
    alsc, contc = ALS(factors=best[0], reg=best[1]).fit(Xc), TagContent(tags, nI).fit(Xc)
    cold = {}
    for name, mdl in {"Popularity": Popularity().fit(Xc), "ALS matrix factorization": alsc,
                      "Tag content (TF-IDF)": contc,
                      f"Hybrid (ALS {best_w:.2f} + content)": Hybrid(alsc, contc, best_w)}.items():
        # treat every non-kept train like + test like as the target for these users
        target = __import__("pandas").concat([train[train.user_id.isin(cs_users + 1)].merge(
            keep[["user_id", "book_id"]], how="left", indicator=True).query("_merge=='left_only'")
            .drop(columns="_merge"), test[test.user_id.isin(cs_users + 1)]])
        m, _ = run_eval(mdl, Xc, Sc, target, cs_users.tolist(), nI)
        cold[name] = m
        print("cold", name, round(m["ndcg@10"], 4), round(m["recall@10"], 4), flush=True)

    out = {"dataset": {"users": int(nU), "books": int(nI), "ratings": len(ratings),
                       "train": len(train), "test": len(test)},
           "als_tuning": tuning, "best_als": {"factors": best[0], "reg": best[1]}, "hybrid_weight": best_w,
           "test_results": results, "cold_start_3_likes": cold, "runtime_sec": round(time.time() - t0)}
    json.dump(out, open("results/metrics.json", "w"), indent=2, default=float)
    print("done", out["runtime_sec"], "s")


if __name__ == "__main__":
    main()
