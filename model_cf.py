"""Simple user-based collaborative filtering.

Given a user-item interaction dataframe (implicit), build a sparse user-item
matrix and compute cosine similarity between users. For a target user,
aggregate neighbors' items to score candidates.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class UserBasedCF:
    def __init__(self, topk_neighbors: int = 50):
        self.topk_neighbors = topk_neighbors
        self.user_index = None
        self.item_index = None
        self.user_item = None
        self.user_sim = None

    def fit(self, interactions: pd.DataFrame):
        # interactions: DataFrame with user_id, item_id
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {v: j for j, v in enumerate(items)}
        self.index_to_item = {j: v for v, j in self.item_index.items()}

        rows = interactions["user_id"].map(self.user_index).values
        cols = interactions["item_id"].map(self.item_index).values
        data = np.ones(len(interactions), dtype=np.float32)
        self.user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        # build nearest-neighbors index to query top-k similar users efficiently
        # we request topk_neighbors+1 because the nearest neighbor includes the user itself
        self.nn = NearestNeighbors(n_neighbors=min(self.topk_neighbors + 1, self.user_item.shape[0]),
                        metric="cosine",
                        algorithm="brute")
        self.nn.fit(self.user_item)

    def recommend(self, user_id: int, k: int = 10, exclude_interacted=True) -> List[int]:
        if user_id not in self.user_index:
            return []
        uidx = self.user_index[user_id]
        # query nearest neighbors (returns distances); convert to similarity
        n_req = min(self.topk_neighbors + 1, self.user_item.shape[0])
        distances, neigh_idx = self.nn.kneighbors(self.user_item[uidx], n_neighbors=n_req, return_distance=True)
        neigh_idx = neigh_idx[0]
        distances = distances[0]
        # convert cosine distance to similarity
        sim_vals = 1.0 - distances
        # drop self (first neighbor if identical)
        mask = neigh_idx != uidx
        neigh_idx = neigh_idx[mask][: self.topk_neighbors]
        neigh_weights = sim_vals[mask][: self.topk_neighbors]

        # weighted sum of neighbors' item vectors
        neigh_item_matrix = self.user_item[neigh_idx]
        scores = np.dot(neigh_weights, neigh_item_matrix.toarray()).ravel()

        if exclude_interacted:
            user_items = set(self.user_item.getrow(uidx).nonzero()[1].tolist())
        else:
            user_items = set()

        # select top-k item indices
        candidate_idx = np.argsort(-scores)
        out = []
        for idx in candidate_idx:
            if scores[idx] <= 0:
                break
            if idx in user_items:
                continue
            out.append(self.index_to_item[idx])
            if len(out) >= k:
                break
        return out
