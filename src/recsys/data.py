"""Load goodbooks-10k through DuckDB and build a per-user random holdout split."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
LIKE = 4  # a rating of 4 or 5 counts as "liked"


def load(db_path=":memory:"):
    con = duckdb.connect(db_path)
    con.execute(f"SET file_search_path='{ROOT}'")
    con.execute((ROOT / "sql" / "prepare.sql").read_text())
    ratings = con.execute("SELECT * FROM ratings").df()
    books = con.execute("SELECT * FROM books").df()
    tags = con.execute("SELECT * FROM book_tag_text").df()
    return ratings, books, tags


def holdout_split(ratings: pd.DataFrame, test_frac=0.2, seed=42):
    """Hold out a random 20% of each user's ratings. Returns (train, test)."""
    rng = np.random.default_rng(seed)
    r = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    r["pos"] = r.groupby("user_id").cumcount()
    r["n"] = r.groupby("user_id")["book_id"].transform("size")
    is_test = r["pos"] < np.floor(r["n"] * test_frac)
    return r.loc[~is_test, ratings.columns].copy(), r.loc[is_test, ratings.columns].copy()


def like_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Binary user x item matrix of liked books (ids are 1-based in the data)."""
    liked = df[df.rating >= LIKE]
    return sp.csr_matrix(
        (np.ones(len(liked), dtype=np.float32), (liked.user_id - 1, liked.book_id - 1)),
        shape=(n_users, n_items),
    )


def seen_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Every rated book, used to avoid recommending what a user already read."""
    return sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df.user_id - 1, df.book_id - 1)),
        shape=(n_users, n_items),
    )
