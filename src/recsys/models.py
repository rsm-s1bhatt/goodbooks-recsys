"""Recommenders: popularity, item-item kNN, ALS matrix factorization, tag content, and a hybrid."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Popularity:
    def fit(self, X):
        self.scores = np.asarray(X.sum(axis=0)).ravel()
        return self

    def score(self, users, X):
        return np.tile(self.scores, (len(users), 1))


class ItemKNN:
    """Cosine similarity between books, keeping each book's top-k neighbours."""
    def __init__(self, k=100):
        self.k = k

    def fit(self, X):
        Xn = normalize(X.T.tocsr(), axis=1)
        S = (Xn @ Xn.T).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):
            s, e = S.indptr[i], S.indptr[i + 1]
            idx, v = S.indices[s:e], S.data[s:e]
            if len(v) > self.k:
                keep = np.argpartition(-v, self.k)[: self.k]
                idx, v = idx[keep], v[keep]
            rows += [i] * len(idx); cols += idx.tolist(); vals += v.tolist()
        self.S = sp.csr_matrix((vals, (rows, cols)), shape=S.shape)
        return self

    def score(self, users, X):
        return (X[users] @ self.S).toarray()


class ALS:
    """Implicit-feedback matrix factorization (Hu, Koren & Volinsky) via the `implicit` library."""
    def __init__(self, factors=64, reg=0.05, alpha=20.0, iters=15, seed=42):
        self.params = dict(factors=factors, regularization=reg, alpha=alpha, iterations=iters, random_state=seed)

    def fit(self, X):
        from implicit.als import AlternatingLeastSquares
        self.m = AlternatingLeastSquares(**self.params)
        self.m.fit(X.astype(np.float32), show_progress=False)
        return self

    def score(self, users, X):
        U = np.asarray(self.m.user_factors[users])
        return U @ np.asarray(self.m.item_factors).T


class TagContent:
    """TF-IDF over reader-applied tags; a user's profile is the mean of their liked books."""
    def __init__(self, tag_text, n_items):
        self.tag_text, self.n_items = tag_text, n_items

    def fit(self, X):
        docs = [""] * self.n_items
        for b, t in zip(self.tag_text.book_id, self.tag_text.tags):
            docs[b - 1] = t
        self.V = normalize(TfidfVectorizer(min_df=5, sublinear_tf=True).fit_transform(docs))
        return self

    def score(self, users, X):
        prof = normalize(X[users] @ self.V)
        return (prof @ self.V.T).toarray()


class Hybrid:
    """Weighted blend of z-scored ALS and tag-content scores."""
    def __init__(self, a, b, w=0.7):
        self.a, self.b, self.w = a, b, w

    def fit(self, X):
        return self

    @staticmethod
    def _z(s):
        return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)

    def score(self, users, X):
        return self.w * self._z(self.a.score(users, X)) + (1 - self.w) * self._z(self.b.score(users, X))
