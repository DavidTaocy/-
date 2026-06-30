"""Wrapper for implicit ALS recommender."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ALSRecommender:
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15, random_state: int = 42):
        self.factors = factors
        self.reg = regularization
        self.iter = iterations
        self.random_state = random_state
        self.model = None
        self.user_index = {}
        self.item_index = {}
        self.index_to_user = {}
        self.index_to_item = {}
        self.user_items_matrix = None
        self.popularity = {}
        self.popular_items = []
        self._errored = False

    def fit(self, interactions: pd.DataFrame):
        # interactions: DataFrame with user_id, item_id
        # use sorted unique IDs for deterministic indexing
        users = np.unique(interactions["user_id"])
        items = np.unique(interactions["item_id"])
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {v: j for j, v in enumerate(items)}
        self.index_to_user = {i: u for u, i in self.user_index.items()}
        self.index_to_item = {j: v for v, j in self.item_index.items()}

        rows = interactions["item_id"].map(self.item_index).values
        cols = interactions["user_id"].map(self.user_index).values
        data = np.ones(len(interactions), dtype=np.float32)
        # implicit expects item-user matrix for training
        item_user = csr_matrix((data, (rows, cols)), shape=(len(items), len(users)))
        self.user_items_matrix = item_user.T.tocsr()

        # popularity counts for items (internal indices)
        item_counts = interactions["item_id"].value_counts()
        # map to internal indices
        self.popularity = {}
        for iid, cnt in item_counts.items():
            if iid in self.item_index:
                self.popularity[self.item_index[iid]] = int(cnt)
        # create popular items list (external ids)
        self.popular_items = [self.index_to_item[idx] for idx in sorted(self.popularity, key=self.popularity.get, reverse=True)]

        # lazy import to avoid hard dependency until requested
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(factors=self.factors, regularization=self.reg, iterations=self.iter, random_state=self.random_state)
        # train on item-user matrix
        # try applying BM25 weighting (if available in implicit) to improve ALS on implicit data
        try:
            from implicit.nearest_neighbours import bm25_weight
            weighted = bm25_weight(item_user)
            model.fit(weighted)
        except Exception:
            # fallback to raw matrix
            model.fit(item_user)
        self.model = model

    def recommend(self, user_id: int, N: int = 10, filter_already_liked: bool = True) -> List[int]:
        if user_id not in self.user_index:
            return []
        uidx = self.user_index[user_id]
        user_items = self.user_items_matrix
        # model.recommend expects user index and user_items (user x item)
        # Use learned factors directly instead of model.recommend to avoid internal checks.
        try:
            # note: in this implicit version, trained attributes are swapped
            # `model.user_factors` correspond to item vectors for our training matrix,
            # and `model.item_factors` correspond to user vectors.
            item_factors = getattr(self.model, 'user_factors', None)
            user_factors = getattr(self.model, 'item_factors', None)
            if item_factors is None or user_factors is None:
                return []

            uvec = user_factors[uidx]
            scores = item_factors.dot(uvec)

            # exclude already seen by masking scores (faster and ensures N results)
            if filter_already_liked:
                seen = set(self.user_items_matrix.getrow(uidx).nonzero()[1].tolist())
            else:
                seen = set()
            if len(seen) > 0:
                scores = scores.copy()
                scores[list(seen)] = -np.inf

            # get top-N indices
            if scores.size == 0:
                top_idx = np.array([], dtype=int)
            elif scores.size <= N:
                top_idx = np.argsort(-scores)
            else:
                top_part = np.argpartition(-scores, N)[:N]
                top_idx = top_part[np.argsort(-scores[top_part])]

            recs = []
            for idx in top_idx:
                idx_int = int(idx)
                # debug: ensure index exists in mapping
                if not hasattr(self, 'index_to_item'):
                    print('no index_to_item mapping')
                else:
                    try:
                        _ = self.index_to_item.get(idx_int, None)
                    except Exception:
                        print('index lookup issue for idx', idx)
                        print('item_factors shape', getattr(item_factors, 'shape', None))
                        print('index_to_item size', len(self.index_to_item))
                
                if idx_int in seen:
                    continue
                recs.append((idx_int, float(scores[idx_int])))
                if len(recs) >= N:
                    break
            out = []
            missing = []
            for i, _ in recs:
                try:
                    out.append(self.index_to_item[i])
                except KeyError:
                    missing.append(i)
            if missing and not self._errored:
                self._errored = True
                print("Missing item indices in index_to_item:", missing[:10])
                print("len(item_factors)", getattr(item_factors, 'shape', None))
                print("len(index_to_item)", len(self.index_to_item))
            return out
        except Exception as e:
            if not self._errored:
                self._errored = True
                import traceback

                print("ALS recommend error (factors):", e)
                traceback.print_exc()
            return []

    def score_candidates(self, user_id: int, candidate_indices: List[int]) -> Dict[int, float]:
        """Return ALS scores for given internal item indices for a user."""
        if user_id not in self.user_index:
            return {i: 0.0 for i in candidate_indices}
        uidx = self.user_index[user_id]
        try:
            # note: swapped orientation as above
            item_factors = getattr(self.model, 'user_factors', None)
            user_factors = getattr(self.model, 'item_factors', None)
            if item_factors is None or user_factors is None:
                return {i: 0.0 for i in candidate_indices}
            uvec = user_factors[uidx]
            scores = {}
            for idx in candidate_indices:
                scores[idx] = float(item_factors[int(idx)].dot(uvec))
            return scores
        except Exception:
            return {i: 0.0 for i in candidate_indices}

    def score_candidates_batch(self, user_ids: List[int], candidate_indices: List[int]) -> Dict[int, Dict[int, float]]:
        """Efficiently score candidates for a batch of users. Returns mapping user_id->(item_idx->score)."""
        out = {}
        if len(user_ids) == 0 or len(candidate_indices) == 0:
            return {u: {i: 0.0 for i in candidate_indices} for u in user_ids}
        try:
            # swapped orientation: model.user_factors are item vectors, model.item_factors are user vectors
            item_factors = getattr(self.model, 'user_factors', None)
            user_factors = getattr(self.model, 'item_factors', None)
            if item_factors is None or user_factors is None:
                return {u: {i: 0.0 for i in candidate_indices} for u in user_ids}

            # prepare matrices
            U_idxs = [self.user_index[u] for u in user_ids if u in self.user_index]
            if len(U_idxs) == 0:
                return {u: {i: 0.0 for i in candidate_indices} for u in user_ids}
            U = user_factors[U_idxs]  # (b, f)
            C = item_factors[candidate_indices]  # (m, f)
            scores_mat = U.dot(C.T)  # (b, m)

            # map back
            for row_idx, uid in enumerate(user_ids):
                if uid not in self.user_index:
                    out[uid] = {i: 0.0 for i in candidate_indices}
                    continue
                # find position in U_idxs
                try:
                    pos = U_idxs.index(self.user_index[uid])
                except ValueError:
                    out[uid] = {i: 0.0 for i in candidate_indices}
                    continue
                row_scores = scores_mat[pos]
                out[uid] = {int(candidate_indices[i]): float(row_scores[i]) for i in range(len(candidate_indices))}
            return out
        except Exception:
            return {u: {i: 0.0 for i in candidate_indices} for u in user_ids}

    def recommend_batch(self, user_ids: List[int], N: int = 10, filter_already_liked: bool = True, k: int = None, **kwargs) -> Dict[int, List[int]]:
        # accept both `N` and `k` (some callers use `k` named arg) and ignore extra kwargs
        if k is not None:
            N = k
        result: Dict[int, List[int]] = {}
        if len(user_ids) == 0:
            return result

        present = [u for u in user_ids if u in self.user_index]
        unseen = [u for u in user_ids if u not in self.user_index]
        for u in unseen:
            # fallback to popular items for unseen users
            result[u] = self.popular_items[:N]

        if not present:
            return result

        item_factors = getattr(self.model, 'user_factors', None)
        user_factors = getattr(self.model, 'item_factors', None)
        if item_factors is None or user_factors is None:
            for u in present:
                result[u] = []
            return result

        uidxs = [self.user_index[u] for u in present]
        U = user_factors[uidxs]  # (b, f)
        scores_mat = U.dot(item_factors.T)

        for row_idx, uid in enumerate(present):
            scores = scores_mat[row_idx]
            if filter_already_liked:
                seen = set(self.user_items_matrix.getrow(self.user_index[uid]).nonzero()[1].tolist())
                if len(seen) > 0:
                    scores = scores.copy()
                    scores[list(seen)] = -np.inf

            if scores.size == 0:
                top_local = np.array([], dtype=int)
            elif scores.size <= N:
                top_local = np.argsort(-scores)
            else:
                part = np.argpartition(-scores, N)[:N]
                top_local = part[np.argsort(-scores[part])]

            recs: List[int] = []
            for idx in top_local:
                recs.append(self.index_to_item[int(idx)])
                if len(recs) >= N:
                    break
            if len(recs) < N:
                for pid in self.popular_items:
                    if pid in recs:
                        continue
                    recs.append(pid)
                    if len(recs) >= N:
                        break
            result[uid] = recs

        return result
