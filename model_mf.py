"""Matrix factorization using TruncatedSVD (works for implicit binary interactions).

This is a lightweight MF approach: build sparse user-item binary matrix,
apply TruncatedSVD to get latent embeddings and compute scores via dot product.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class MFRecommender:
    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        # maximum number of item candidates to score per user (reduce cost)
        self.candidate_size = 5000
        self.user_index = None
        self.item_index = None
        self.index_to_item = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions: pd.DataFrame):
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {v: j for j, v in enumerate(items)}
        self.index_to_item = {j: v for v, j in self.item_index.items()}

        rows = interactions["user_id"].map(self.user_index).values
        cols = interactions["item_id"].map(self.item_index).values
        data = np.ones(len(interactions), dtype=np.float32)
        mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        # store user->interacted item index sets for exclusion later
        self.user_interactions = {i: set(mat.getrow(i).nonzero()[1].tolist()) for i in range(mat.shape[0])}

        # TruncatedSVD on sparse matrix
        svd = TruncatedSVD(n_components=min(self.n_components, min(mat.shape) - 1), random_state=self.random_state)
        user_latent = svd.fit_transform(mat)
        item_latent = svd.components_.T  # (n_items, n_factors)

        self.user_factors = user_latent
        self.item_factors = item_latent
        # keep numpy versions for fast math
        self.item_factors_np = np.asarray(self.item_factors)
        self.user_factors_np = np.asarray(self.user_factors)
        # popularity list for fallback recommendations
        item_counts = interactions["item_id"].value_counts()
        self.popular_items = item_counts.index.tolist()
        # candidate item indices (top-N by popularity) to restrict scoring
        n_cand = min(self.candidate_size, len(self.item_index))
        popular_subset = item_counts.index[:n_cand].tolist()
        # map popular item ids to their internal indices
        self.candidate_item_indices = [self.item_index[iid] for iid in popular_subset if iid in self.item_index]

    def recommend(self, user_id: int, k: int = 10, exclude_interacted=True) -> List[int]:
        # Single-user wrapper that calls batch recommend for convenience
        res = self.recommend_batch([user_id], k=k, exclude_interacted=exclude_interacted)
        return res.get(user_id, [])

    def recommend_batch(self, user_ids: List[int], k: int = 10, exclude_interacted=True, batch_size: int = 512) -> Dict[int, List[int]]:
        """Recommend top-k for a list of user_ids efficiently in batches.

        Returns a dict user_id -> list[item_id]
        """
        # map input user_ids to indices that exist in training; unseen users will get popular fallback
        present = [u for u in user_ids if u in self.user_index]
        unseen = [u for u in user_ids if u not in self.user_index]

        result: Dict[int, List[int]] = {}
        # fallback for unseen users
        for u in unseen:
            result[u] = self.popular_items[:k]

        if len(present) == 0:
            return result

        # process in batches of training-indexed users
        idxs = [self.user_index[u] for u in present]
        # create reverse index mapping for quick lookup
        index_to_user = {v: k for k, v in self.user_index.items()}

        # prepare candidate item factors (restrict to popular candidates)
        if len(self.candidate_item_indices) == 0:
            cand_idx_arr = np.arange(self.item_factors_np.shape[0])
        else:
            cand_idx_arr = np.array(self.candidate_item_indices, dtype=int)
        item_factors_sub = self.item_factors_np[cand_idx_arr]  # (m, f)

        for i in range(0, len(idxs), batch_size):
            batch = idxs[i : i + batch_size]
            U = self.user_factors_np[batch]  # (b, f)
            # scores = U @ item_factors_sub.T -> (b, m)
            scores = U.dot(item_factors_sub.T)

            # get top-k indices per row using argpartition along axis=1
            if scores.shape[1] <= k:
                top_idx_matrix = np.argsort(-scores, axis=1)
            else:
                top_part = np.argpartition(-scores, kth=k, axis=1)[:, :k]
                # sort the top_part to get ordered top-k
                row_indices = np.arange(top_part.shape[0])[:, None]
                top_idx_matrix = top_part[row_indices, np.argsort(-scores[row_indices, top_part])]

            for row_local, uidx in enumerate(batch):
                row_top_idx = top_idx_matrix[row_local]
                if exclude_interacted:
                    seen = self.user_interactions.get(uidx, set())
                else:
                    seen = set()

                recs = []
                for rel_idx in row_top_idx:
                    orig_idx = int(cand_idx_arr[rel_idx])
                    if orig_idx in seen:
                        continue
                    recs.append(self.index_to_item[orig_idx])
                    if len(recs) >= k:
                        break

                # fill up with popular items if necessary
                if len(recs) < k:
                    for pid in self.popular_items:
                        if pid in recs:
                            continue
                        recs.append(pid)
                        if len(recs) >= k:
                            break

                result_key = index_to_user.get(uidx)
                if result_key is None:
                    continue
                result[result_key] = recs

        return result
