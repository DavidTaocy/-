"""Hybrid: ALS + ItemCF + popularity fusion (v2, tuned for implicit feedback)."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _minmax_rows(scores: np.ndarray) -> np.ndarray:
    mn = scores.min(axis=1, keepdims=True)
    mx = scores.max(axis=1, keepdims=True)
    return (scores - mn) / (mx - mn + 1e-9)


def _weight_matrix(mat: csr_matrix) -> csr_matrix:
    from implicit.nearest_neighbours import bm25_weight, tfidf_weight

    try:
        return tfidf_weight(bm25_weight(mat))
    except Exception:
        try:
            return bm25_weight(mat)
        except Exception:
            return mat


class HybridRecommender:
    def __init__(
        self,
        factors: int = 256,
        regularization: float = 0.1,
        iterations: int = 80,
        confidence_alpha: float = 20.0,
        ease_l2: float = 250.0,
        use_ease: bool = False,
        pop_debias: float = 0.0,
        random_state: int = 42,
    ):
        self.factors = factors
        self.reg = regularization
        self.iter = iterations
        self.confidence_alpha = confidence_alpha
        self.ease_l2 = ease_l2
        self.use_ease = use_ease
        self.pop_debias = pop_debias
        self.random_state = random_state
        self.user_index: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_to_item: Dict[int, int] = {}
        self.user_items_matrix: Optional[csr_matrix] = None
        self.popularity: Dict[int, int] = {}
        self.popular_items: List[int] = []
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._item_similarity: Optional[csr_matrix] = None
        self._ease_B: Optional[np.ndarray] = None

    def fit(self, interactions: pd.DataFrame):
        # deterministic mappings
        users = np.unique(interactions["user_id"])
        items = np.unique(interactions["item_id"])
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {v: j for j, v in enumerate(items)}
        self.index_to_item = {j: v for v, j in self.item_index.items()}

        rows = interactions["user_id"].map(self.user_index).values
        cols = interactions["item_id"].map(self.item_index).values
        data = np.ones(len(interactions), dtype=np.float32)
        user_items = csr_matrix(
            (data, (rows, cols)), shape=(len(users), len(items))
        )
        self.user_items_matrix = user_items

        item_counts = interactions["item_id"].value_counts()
        self.popularity = {
            self.item_index[iid]: int(cnt)
            for iid, cnt in item_counts.items()
            if iid in self.item_index
        }
        self.popular_items = [
            self.index_to_item[idx]
            for idx in sorted(self.popularity, key=self.popularity.get, reverse=True)
        ]

        weighted_ui = _weight_matrix(user_items)
        if self.confidence_alpha > 0:
            weighted_ui = weighted_ui * self.confidence_alpha

        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import CosineRecommender

        als = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iter,
            random_state=self.random_state,
            use_gpu=False,
        )
        als.fit(weighted_ui)
        self._user_factors = np.asarray(als.user_factors)
        self._item_factors = np.asarray(als.item_factors)

        item_user = user_items.T.tocsr()
        weighted_iu = _weight_matrix(item_user)
        itemcf = CosineRecommender()
        itemcf.fit(weighted_iu)
        sim = itemcf.similarity
        n_items = len(self.item_index)
        if sim.shape[0] != n_items:
            from sklearn.metrics.pairwise import cosine_similarity

            sim = cosine_similarity(weighted_iu, dense_output=False)
        self._item_similarity = sim

        if self.use_ease:
            gram = (user_items.T @ user_items).astype(np.float64)
            if hasattr(gram, "toarray"):
                gram = gram.toarray()
            gram += self.ease_l2 * np.eye(n_items, dtype=np.float64)
            P = np.linalg.inv(gram)
            diag = np.diag(P).copy()
            diag[diag == 0] = 1e-9
            B = P / (-diag)
            np.fill_diagonal(B, 0.0)
            self._ease_B = B.astype(np.float32)

    def _seen_items(self, uidx: int) -> set:
        return set(self.user_items_matrix.getrow(uidx).nonzero()[1].tolist())

    def recommend_batch(
        self,
        user_ids: List[int],
        k: int = 10,
        w_als: float = 0.45,
        w_itemcf: float = 0.40,
        w_ease: float = 0.0,
        w_pop: float = 0.15,
        exclude_interacted: bool = True,
        batch_size: int = 512,
    ) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        unseen = [u for u in user_ids if u not in self.user_index]
        present = [u for u in user_ids if u in self.user_index]

        for u in unseen:
            result[u] = self.popular_items[:k]

        if not present:
            return result

        n_items = len(self.item_index)
        pops = np.array(
            [self.popularity.get(i, 0) for i in range(n_items)], dtype=np.float32
        )
        max_pop = float(pops.max()) if len(pops) > 0 else 1.0
        pop_norm = pops / max_pop
        log_pop = np.log1p(pops) / np.log1p(max_pop)

        if w_ease > 0 and self._ease_B is not None:
            weights = np.array([w_als, w_itemcf, w_ease, w_pop], dtype=np.float32)
        else:
            weights = np.array([w_als, w_itemcf, w_pop], dtype=np.float32)
            w_ease = 0.0
        weights = weights / weights.sum()

        for start in range(0, len(present), batch_size):
            batch_users = present[start : start + batch_size]
            uidxs = [self.user_index[u] for u in batch_users]
            user_emb = self._user_factors[uidxs]
            als_scores = user_emb.dot(self._item_factors.T)

            ui_batch = self.user_items_matrix[uidxs]
            itemcf_scores = ui_batch.dot(self._item_similarity)
            if hasattr(itemcf_scores, "toarray"):
                itemcf_scores = itemcf_scores.toarray()

            als_norm = _minmax_rows(als_scores)
            itemcf_norm = _minmax_rows(itemcf_scores)

            if w_ease > 0 and self._ease_B is not None:
                ease_scores = ui_batch.dot(self._ease_B)
                if hasattr(ease_scores, "toarray"):
                    ease_scores = ease_scores.toarray()
                ease_norm = _minmax_rows(ease_scores)
                combined = (
                    weights[0] * als_norm
                    + weights[1] * itemcf_norm
                    + weights[2] * ease_norm
                    + weights[3] * pop_norm
                )
            else:
                combined = (
                    weights[0] * als_norm
                    + weights[1] * itemcf_norm
                    + weights[2] * pop_norm
                )

            # optional mild popularity debias
            if self.pop_debias > 0:
                combined = combined - self.pop_debias * log_pop

            for row, (uid, uidx) in enumerate(zip(batch_users, uidxs)):
                scores = combined[row]
                seen = self._seen_items(uidx) if exclude_interacted else set()

                if len(scores) <= k:
                    top_local = np.argsort(-scores)
                else:
                    part = np.argpartition(-scores, k)[:k]
                    top_local = part[np.argsort(-scores[part])]

                recs: List[int] = []
                for local_idx in top_local:
                    item_idx = int(local_idx)
                    if item_idx in seen:
                        continue
                    recs.append(self.index_to_item[item_idx])
                    if len(recs) >= k:
                        break

                if len(recs) < k:
                    for pid in self.popular_items:
                        internal = self.item_index.get(pid)
                        if internal is not None and internal in seen:
                            continue
                        if pid in recs:
                            continue
                        recs.append(pid)
                        if len(recs) >= k:
                            break

                result[uid] = recs

        return result
