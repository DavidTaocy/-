"""Local validation script: random per-user holdout and F1 evaluation.

Usage example:
python src/validate.py --max_train 20000 --factors 64 --top_k 10
"""
import argparse
import random
from collections import defaultdict
import numpy as np
import pandas as pd

from data_loader import load_data
from als_model import ALSRecommender
from hybrid_model import HybridRecommender


def train_val_split(interactions: pd.DataFrame, holdout_per_user: int = 1, seed: int = 42):
    """Random per-user holdout: for users with > holdout_per_user interactions, sample holdout_per_user as val."""
    random.seed(seed)
    users = interactions["user_id"].unique()
    train_rows = []
    val_rows = []
    grouped = interactions.groupby("user_id")
    for u, g in grouped:
        items = g["item_id"].tolist()
        if len(items) <= holdout_per_user:
            train_rows.extend(g.index.tolist())
            continue
        val_idx = random.sample(g.index.tolist(), holdout_per_user)
        for idx in g.index.tolist():
            if idx in val_idx:
                val_rows.append(idx)
            else:
                train_rows.append(idx)

    train_df = interactions.loc[train_rows].reset_index(drop=True)
    val_df = interactions.loc[val_rows].reset_index(drop=True)
    return train_df, val_df


def compute_f1(pred_map, val_map):
    TP = 0
    FP = 0
    FN = 0
    for u, true_items in val_map.items():
        preds = set(pred_map.get(u, []))
        TP += len(preds & true_items)
        FP += len(preds - true_items)
        FN += len(true_items - preds)
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return P, R, F1, TP, FP, FN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_train", type=int, default=20000)
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--candidate_size", type=int, default=5000, help="Number of popular items to consider as candidate set")
    p.add_argument("--per_user_cand", type=int, default=0, help="Number of per-user top ALS candidates to union with global candidates (0=disabled)")
    p.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0", help="(legacy) comma-separated fusion weights")
    # Hybrid-specific weights
    p.add_argument("--use_hybrid", action="store_true", help="Use HybridRecommender (ALS+ItemCF+optional EASE)")
    p.add_argument("--w_als", type=float, default=0.45)
    p.add_argument("--w_itemcf", type=float, default=0.40)
    p.add_argument("--w_ease", type=float, default=0.0)
    p.add_argument("--w_pop", type=float, default=0.15)
    p.add_argument("--use_ease", action='store_true')
    p.add_argument("--ease_l2", type=float, default=250.0)
    p.add_argument("--try_grid", action='store_true', help="Try a small grid of weight combinations to find best F1")
    return p.parse_args()


def main():
    args = parse_args()
    df, test_users, _ = load_data(args.data_dir)
    # sample training rows for speed
    if args.max_train and args.max_train > 0 and len(df) > args.max_train:
        df = df.sample(args.max_train, random_state=42).reset_index(drop=True)

    print("Splitting train/val...")
    train_df, val_df = train_val_split(df, holdout_per_user=1)
    print(f"Train: {len(train_df)} rows; Val: {len(val_df)} rows")

    # users to validate on
    val_group = val_df.groupby("user_id")
    val_map = {u: set(g["item_id"].tolist()) for u, g in val_group}

    # prefer hybrid recommender (ALS + ItemCF + optional EASE)
    if args.use_hybrid:
        print("Training Hybrid (ALS + ItemCF)...")
        hybrid = HybridRecommender(factors=args.factors, iterations=args.iterations, use_ease=args.use_ease, ease_l2=args.ease_l2)
        hybrid.fit(train_df)

        users = list(val_map.keys())
        print(f"Validation users: {len(users)}; present in Hybrid mapping: {sum(1 for u in users if u in hybrid.user_index)}")

        # generate weight grid or single combo
        combos = []
        if args.try_grid:
            als_opts = [0.3, 0.4, 0.5]
            itemcf_opts = [0.2, 0.35, 0.45]
            ease_opts = [0.0, 0.15] if args.use_ease else [0.0]
            for wa in als_opts:
                for wi in itemcf_opts:
                    for we in ease_opts:
                        wp = 1.0 - (wa + wi + we)
                        if wp < 0:
                            continue
                        combos.append((wa, wi, we, wp))
        else:
            combos = [(args.w_als, args.w_itemcf, args.w_ease, args.w_pop)]

        best = None
        results = {}
        for (wa, wi, we, wp) in combos:
            pred_map = hybrid.recommend_batch(users, k=args.top_k, w_als=wa, w_itemcf=wi, w_ease=we, w_pop=wp, exclude_interacted=True)
            P, R, F1, TP, FP, FN = compute_f1(pred_map, val_map)
            results[(wa, wi, we, wp)] = (P, R, F1, TP, FP, FN)
            print(f"w=(%0.2f,%0.2f,%0.2f,%0.2f) -> P:{P:.6f} R:{R:.6f} F1:{F1:.6f} TP:{TP} FP:{FP} FN:{FN}" % (wa, wi, we, wp))
            if best is None or F1 > best[0]:
                best = (F1, (wa, wi, we, wp))

        print("Best weights:", best)
    else:
        print("Training ALS (legacy path)...")
        als = ALSRecommender(factors=args.factors, iterations=args.iterations)
        als.fit(train_df)
        # fallback to original fusion behavior over alphas
        pop_items = sorted(als.popularity.items(), key=lambda x: -x[1])
        candidate_size = min(args.candidate_size, len(pop_items))
        candidate_indices = [iid for iid, _ in pop_items[:candidate_size]]
        pop_scores = {candidate_indices[i]: (als.popularity.get(candidate_indices[i], 0) / max(als.popularity.values())) for i in range(len(candidate_indices))} if len(candidate_indices)>0 else {}
        users = list(val_map.keys())
        alphas = [float(x) for x in args.alphas.split(",")]
        best = None
        for alpha in alphas:
            pred_map = {}
            # simple per-user scoring using precomputed candidates
            for u in users:
                if u not in als.user_index:
                    pred_map[u] = set()
                    continue
                # per-user top over candidates
                scores = als.score_candidates(u, candidate_indices)
                combined = {i: alpha * scores.get(i, 0.0) + (1.0 - alpha) * pop_scores.get(i, 0.0) for i in candidate_indices}
                if len(combined) == 0:
                    pred_map[u] = set()
                else:
                    items_arr = np.array(list(combined.keys()))
                    scores_arr = np.array([combined[int(x)] for x in items_arr])
                    k = args.top_k
                    if scores_arr.size <= k:
                        top_idx = np.argsort(-scores_arr)
                    else:
                        part = np.argpartition(-scores_arr, k)[:k]
                        top_idx = part[np.argsort(-scores_arr[part])]
                    top_items = [als.index_to_item[int(items_arr[j])] for j in top_idx if int(items_arr[j]) in als.index_to_item]
                    pred_map[u] = set(top_items)

            P, R, F1, TP, FP, FN = compute_f1(pred_map, val_map)
            print(f"alpha={alpha:.2f} -> P:{P:.6f} R:{R:.6f} F1:{F1:.6f} TP:{TP} FP:{FP} FN:{FN}")
            if best is None or F1 > best[0]:
                best = (F1, alpha)
        print("Best alpha:", best)


if __name__ == "__main__":
    main()
