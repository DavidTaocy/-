"""EASE (Embarrassingly Shallow Autoencoder) for item recommendation."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class EASERecommender:
    def __init__(self, l2: float = 250.0):
        self.l2 = l2
        self.user_index: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_to_item: Dict[int, int] = {}
        self.user_items_matrix: Optional[csr_matrix] = None
        self.popular_items: List[int] = []
        self._B: Optional[np.ndarray] = None

    def fit(self, interactions: pd.DataFrame):
        # use deterministic ordering for indices
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
        self.popular_items = [
            iid for iid in item_counts.index if iid in self.item_index
        ]

        n_items = len(items)
        gram = (user_items.T @ user_items).astype(np.float64)
        if hasattr(gram, "toarray"):
            gram = gram.toarray()
        gram += self.l2 * np.eye(n_items, dtype=np.float64)
        P = np.linalg.inv(gram)
        diag = np.diag(P).copy()
        diag[diag == 0] = 1e-9
        B = P / (-diag)
        np.fill_diagonal(B, 0.0)
        self._B = B.astype(np.float32)

    def _seen_items(self, uidx: int) -> set:
        return set(self.user_items_matrix.getrow(uidx).nonzero()[1].tolist())

    def recommend_batch(
        self,
        user_ids: List[int],
        k: int = 10,
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

        for start in range(0, len(present), batch_size):
            batch_users = present[start : start + batch_size]
            uidxs = [self.user_index[u] for u in batch_users]
            ui_batch = self.user_items_matrix[uidxs]
            scores = ui_batch.dot(self._B)
            if hasattr(scores, "toarray"):
                scores = scores.toarray()

            for row, (uid, uidx) in enumerate(zip(batch_users, uidxs)):
                row_scores = scores[row]
                seen = self._seen_items(uidx) if exclude_interacted else set()

                if len(row_scores) <= k:
                    top_idx = np.argsort(-row_scores)
                else:
                    part = np.argpartition(-row_scores, k)[:k]
                    top_idx = part[np.argsort(-row_scores[part])]

                recs: List[int] = []
                for idx in top_idx:
                    if int(idx) in seen:
                        continue
                    recs.append(self.index_to_item[int(idx)])
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
