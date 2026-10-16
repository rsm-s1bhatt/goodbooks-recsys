# Book Recommender: Collaborative Filtering vs. Content vs. Hybrid

Top-10 book recommendations for 53,424 real Goodreads users, built on the public
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset (5.98M ratings, 10,000 books).

The question: for each reader, which 10 books they haven't read yet are they most likely to love?
That is the same shape of problem as recommending articles to subscribers: sparse behavior,
a long tail of items, heavy popularity bias, and new users with almost no history.

## Setup

- **Data layer:** DuckDB SQL (`sql/prepare.sql`) dedupes ratings and builds a tag profile per book
  from the top 30 tags readers applied to it.
- **"Liked"** means a rating of 4 or 5.
- **Split:** a random 20% of each user's ratings is held out for testing. Model settings were
  tuned on a separate validation split carved out of the training data only, never on the test set.
- **Metrics:** precision, recall, NDCG and hit rate at 10, plus catalog coverage and how
  popular the recommended books are. Books a user already rated are never recommended.

## Models

| Model | Idea |
|---|---|
| Popularity | Recommend the most-liked books to everyone. The baseline to beat. |
| Item-item kNN | Books liked by the same readers are similar (cosine, top 100 neighbors). |
| ALS matrix factorization | Implicit-feedback ALS (Hu, Koren & Volinsky, 2008), 128 factors. |
| Tag content | TF-IDF over reader tags; a user's profile is the average of books they liked. |
| Hybrid | 0.7 × ALS + 0.3 × tag content, each z-scored per user. Weight chosen on validation. |

## Results (test set, all 53,341 users with a held-out liked book)

| Model | NDCG@10 | Precision@10 | Recall@10 | Hit rate@10 | Catalog coverage | Median popularity rank of recs |
|---|---|---|---|---|---|---|
| Popularity | 0.091 | 0.073 | 0.049 | 41.5% | 0.4% | 6 |
| Tag content | 0.085 | 0.075 | 0.053 | 44.3% | 52.8% | 1,503 |
| Item-item kNN | 0.271 | 0.226 | 0.154 | 82.1% | 40.3% | 29 |
| ALS | 0.276 | 0.250 | 0.173 | 87.8% | 41.4% | 243 |
| **Hybrid** | **0.282** | **0.252** | **0.175** | **88.1%** | **46.5%** | 296 |

**Cold start** (5,000 users cut down to only 3 liked books):

| Model | NDCG@10 | Recall@10 |
|---|---|---|
| Popularity | 0.285 | 0.038 |
| Tag content | 0.295 | 0.040 |
| ALS | 0.416 | 0.057 |
| **Hybrid** | **0.426** | **0.058** |

Cold-start numbers are scored against all of those users' remaining liked books, so they aren't
comparable to the main table, only to each other.

## What I took from it

- ALS roughly **3x the popularity baseline** on NDCG@10, and reaches 88% of users with at least one hit in the top 10.
- Item-kNN is nearly as accurate as ALS but keeps recommending the same hits (median popularity rank 29 vs. 243).
  ALS finds more of the long tail for the same accuracy.
- Tags alone are weak at ranking but cover over half the catalog. Blending a little of them into ALS
  raised accuracy slightly and **catalog coverage from 41% to 47%**.
- For users with only 3 likes, the hybrid beats popularity by **about 49%** on NDCG@10.

## Limitations

- Ratings have no timestamps, so the split is random rather than "predict the future from the past."
  A time-based split would give lower, more realistic numbers.
- The best ALS size (128 factors) was the largest one tried; a wider search might do better.
- Offline metrics only. A real system would need an A/B test on clicks or reads.

## Run it

```bash
pip install -r requirements.txt
./get_data.sh
PYTHONPATH=src python -m recsys.run     # about 6 minutes on 2 CPUs; writes results/metrics.json
PYTHONPATH=src pytest -q tests
```
