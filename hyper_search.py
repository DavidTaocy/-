"""Random hyperparameter search for ALS fusion with popularity.

Usage: python src/hyper_search.py --samples 20 --max_train 20000
"""
import argparse
import random
import time
from collections import defaultdict
import json

import numpy as np
import pandas as pd

from data_loader import load_data
from als_model import ALSRecommender
from ease_model import EASERecommender


def train_val_split(interactions: pd.DataFrame, holdout_per_user: int = 1, seed: int = 42):
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


def evaluate_config(train_df, val_map, config, top_k=10):
    als = ALSRecommender(factors=config["factors"], iterations=config["iterations"]) 
    als.fit(train_df)

    ease = None
    if config.get("use_ease", False):
        ease = EASERecommender(l2=config.get("ease_l2", 250.0))
        ease.fit(train_df)

    # build popularity-sorted candidate internal indices
    pop_items = sorted(als.popularity.items(), key=lambda x: -x[1])
    candidate_size = min(config["candidate_size"], len(pop_items))
    candidate_indices = [iid for iid, _ in pop_items[:candidate_size]]

    # normalized pop scores
    pops = [als.popularity.get(i, 0) for i in candidate_indices]
    maxp = max(pops) if len(pops) > 0 else 1
    pop_scores = {candidate_indices[i]: pops[i] / maxp for i in range(len(candidate_indices))}

    users = list(val_map.keys())
    pred_map = {}

    # use batch scoring where possible
    batch = 256
    for i in range(0, len(users), batch):
        batch_users = users[i: i + batch]
        # ALS scores for batch
        scores_dict = als.score_candidates_batch(batch_users, candidate_indices)

        # EASE scores for batch (if enabled)
        ease_scores_dict = {}
        if ease is not None:
            # map candidate external ids to ease internal indices
            cand_ext = [als.index_to_item[idx] for idx in candidate_indices]
            ease_internal = [ease.item_index.get(ext, None) for ext in cand_ext]
            # get user internal indices for ease
            U_idxs = [ease.user_index.get(u, None) for u in batch_users]
            present_idxs = [x for x in U_idxs if x is not None]
            if len(present_idxs) > 0:
                ui_rows = ease.user_items_matrix[[idx for idx in U_idxs if idx is not None]]
                scores_all = ui_rows.dot(ease._B)
                if hasattr(scores_all, 'toarray'):
                    scores_all = scores_all.toarray()
            else:
                scores_all = None

            for u_idx, u in enumerate(batch_users):
                # start from popularity candidates
                combined = {}
                for idx in candidate_indices:
                    als_score = scores_dict.get(u, {}).get(idx, 0.0)
                    pop_score = pop_scores.get(idx, 0.0)
                    # include EASE score if available
                    ease_score = 0.0
                    if ease is not None and scores_all is not None:
                        # map candidate index to ease internal index
                        try:
                            ext = als.index_to_item[idx]
                            eidx = ease.item_index.get(ext, None)
                            if eidx is not None:
                                # find user row position
                                if ease.user_index.get(u, None) is not None:
                                    pos = [ease.user_index.get(x, None) for x in batch_users].index(ease.user_index.get(u))
                                    ease_score = float(scores_all[pos, eidx])
                        except Exception:
                            ease_score = 0.0

                    combined[idx] = config.get("alpha", 1.0) * als_score + config.get("beta", 0.0) * ease_score + (1.0 - config.get("alpha", 1.0) - config.get("beta", 0.0)) * pop_score

                # optionally add per-user top from full recommend
                if config.get("per_user_cand", 0) and u in als.user_index:
                    per = als.recommend(u, N=config["per_user_cand"], filter_already_liked=False)
                    # map to internal indices
                    for item in per:
                        if item in als.item_index:
                            idx = als.item_index[item]
                            if idx not in combined:
                                combined[idx] = config["alpha"] * 1.0 + (1.0 - config["alpha"]) * pop_scores.get(idx, 0.0)

                # select top-k
                if len(combined) == 0:
                    pred_map[u] = set()
                    continue
                items_arr = np.array(list(combined.keys()))
                scores_arr = np.array([combined[int(x)] for x in items_arr])
                if scores_arr.size <= top_k:
                    top_idx = np.argsort(-scores_arr)
                else:
                    part = np.argpartition(-scores_arr, top_k)[:top_k]
                    top_idx = part[np.argsort(-scores_arr[part])]
                preds = [als.index_to_item[int(items_arr[j])] for j in top_idx if int(items_arr[j]) in als.index_to_item]
                pred_map[u] = set(preds)

    P, R, F1, TP, FP, FN = compute_f1(pred_map, val_map)
    return P, R, F1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_train", type=int, default=20000)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    df, test_users, _ = load_data(args.data_dir)
    if args.max_train and args.max_train > 0 and len(df) > args.max_train:
        df = df.sample(args.max_train, random_state=args.seed).reset_index(drop=True)

    train_df, val_df = train_val_split(df, holdout_per_user=1, seed=args.seed)
    val_group = val_df.groupby("user_id")
    val_map = {u: set(g["item_id"].tolist()) for u, g in val_group}

    # search space
    space = {
        "factors": [32, 64, 96],
        "iterations": [5, 10, 15],
        "candidate_size": [500, 2000, 5000],
        "per_user_cand": [0, 20, 50, 200],
        "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
        "use_ease": [False, True],
        "beta": [0.0, 0.25, 0.5],
        "ease_l2": [50.0, 250.0, 1000.0],
    }

    best = None
    results = []
    start = time.time()
    # ensure outputs dir exists for best config write
    import os
    os.makedirs('outputs', exist_ok=True)

    for s in range(args.samples):
        cfg = {k: random.choice(v) for k, v in space.items()}
        cfg["seed"] = args.seed
        t0 = time.time()
        try:
            P, R, F1 = evaluate_config(train_df, val_map, cfg, top_k=args.top_k)
        except KeyboardInterrupt:
            print("Interrupted by user during evaluation; saving current best and exiting.")
            break
        except Exception as e:
            print("Error evaluating config:", e)
            continue
        dt = time.time() - t0
        results.append((cfg, P, R, F1, dt))
        print(f"sample {s+1}/{args.samples}: cfg={cfg} -> P={P:.6f} R={R:.6f} F1={F1:.6f} time={dt:.1f}s")
        if best is None or F1 > best[0]:
            best = (F1, cfg, P, R)
            # save intermediate best
            try:
                with open('outputs/best_config.json', 'w', encoding='utf-8') as bf:
                    json.dump({'f1': best[0], 'config': best[1], 'P': best[2], 'R': best[3]}, bf, ensure_ascii=False, indent=2)
            except Exception:
                pass

    total = time.time() - start
    print("\nSearch finished. Total time: %.1fs" % total)
    print("Best F1: %.6f, P: %.6f, R: %.6f" % (best[0], best[2], best[3]))
    print("Best config:", best[1])
    # write final best config
    try:
        with open('outputs/best_config.json', 'w', encoding='utf-8') as bf:
            json.dump({'f1': best[0], 'config': best[1], 'P': best[2], 'R': best[3]}, bf, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
