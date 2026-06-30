"""End-to-end optimization: recall, ranking, grid search, and submission export.

Usage examples:
  python -u src/full_optimize.py --max_train 20000 --grid_small
  python -u src/full_optimize.py --max_train 50000 --samples 30

This script strictly uses data in the `data/` folder and writes outputs to `outputs/`.
"""
import argparse
import json
import os
import time
from itertools import product

import numpy as np
import pandas as pd

from data_loader import load_data
from hybrid_model import HybridRecommender


def train_val_split(interactions: pd.DataFrame, holdout_per_user: int = 1, seed: int = 42):
    import random
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


def evaluate_hybrid_cfg(train_df, val_map, cfg, top_k=10):
    # train hybrid recommender
    hybrid = HybridRecommender(factors=cfg.get('factors', 64), iterations=cfg.get('iterations', 10), use_ease=cfg.get('use_ease', False), ease_l2=cfg.get('ease_l2', 250.0))
    hybrid.fit(train_df)

    users = list(val_map.keys())
    # get predictions using recommend_batch
    pred_map = hybrid.recommend_batch(users, k=top_k, w_als=cfg.get('w_als', 0.45), w_itemcf=cfg.get('w_itemcf', 0.40), w_ease=cfg.get('w_ease', 0.0), w_pop=cfg.get('w_pop', 0.15), exclude_interacted=True)

    P, R, F1, TP, FP, FN = compute_f1(pred_map, val_map)
    return P, R, F1


def grid_search(train_df, val_map, grid, fixed_cfg, top_k=10):
    best = None
    results = []
    for params in grid:
        cfg = dict(fixed_cfg)
        cfg.update(params)
        P, R, F1 = evaluate_hybrid_cfg(train_df, val_map, cfg, top_k=top_k)
        results.append((cfg, P, R, F1))
        if best is None or F1 > best[0]:
            best = (F1, cfg, P, R)
        print(f"cfg={params} -> P={P:.6f} R={R:.6f} F1={F1:.6f}")
    return best, results


def generate_submission_full(train_df, test_users, best_cfg, out_path, top_k=10, batch_size=512):
    hybrid = HybridRecommender(factors=best_cfg.get('factors', 64), iterations=best_cfg.get('iterations', 10), use_ease=best_cfg.get('use_ease', False), ease_l2=best_cfg.get('ease_l2', 250.0))
    hybrid.fit(train_df)

    users = test_users.tolist()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    import csv
    written = 0
    start = time.time()
    with open(out_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['user_id', 'item_id'])
        for i in range(0, len(users), batch_size):
            batch = users[i: i + batch_size]
            preds = hybrid.recommend_batch(batch, k=top_k, w_als=best_cfg.get('w_als', 0.45), w_itemcf=best_cfg.get('w_itemcf', 0.40), w_ease=best_cfg.get('w_ease', 0.0), w_pop=best_cfg.get('w_pop', 0.15), exclude_interacted=True)
            for u in batch:
                items = preds.get(u, [])[:top_k]
                for it in items:
                    writer.writerow([u, it])
                    written += 1
    elapsed = time.time() - start
    print(f'Wrote submission to {out_path}; rows: {written}; time: {elapsed:.1f}s')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--max_train', type=int, default=20000)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--grid_small', action='store_true')
    p.add_argument('--out', type=str, default='outputs/submission_full_opt.csv')
    return p.parse_args()


def main():
    args = parse_args()
    print('Loading data...')
    df, test_users, _ = load_data(args.data_dir)
    if args.max_train and args.max_train > 0 and len(df) > args.max_train:
        df = df.sample(args.max_train, random_state=42).reset_index(drop=True)
    print('Rows after sampling:', len(df))

    print('Splitting train/val...')
    train_df, val_df = train_val_split(df, holdout_per_user=1)
    val_group = val_df.groupby('user_id')
    val_map = {u: set(g['item_id'].tolist()) for u, g in val_group}

    fixed_cfg = {'factors': 64, 'iterations': 10, 'use_ease': False, 'ease_l2': 250.0}

    if args.grid_small:
        candidate_sizes = [500, 2000]
        per_user = [0, 20]
        w_als = [0.4, 0.6]
        w_itemcf = [0.3, 0.4]
        w_ease = [0.0]
        w_pop = [0.3, 0.2]
    else:
        candidate_sizes = [500, 2000, 5000]
        per_user = [0, 20, 50]
        w_als = [0.3, 0.4, 0.5]
        w_itemcf = [0.2, 0.35, 0.45]
        w_ease = [0.0, 0.15]
        w_pop = [0.15, 0.2]

    grid = []
    for cs, pu, wa, wi, we, wp in product(candidate_sizes, per_user, w_als, w_itemcf, w_ease, w_pop):
        # ensure weights sum <=1; we'll normalize in Hybrid
        grid.append({'candidate_size': cs, 'per_user_cand': pu, 'w_als': wa, 'w_itemcf': wi, 'w_ease': we, 'w_pop': wp})

    print('Grid size:', len(grid))
    best, results = grid_search(train_df, val_map, grid, fixed_cfg, top_k=args.top_k)
    print('Best:', best)

    # save best
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/best_full_opt.json', 'w', encoding='utf-8') as f:
        json.dump({'best': best[1], 'f1': float(best[0]), 'P': float(best[2]), 'R': float(best[3])}, f, ensure_ascii=False, indent=2)

    # generate submission on full train using best cfg
    print('Generating submission for test users...')
    generate_submission_full(df, test_users, best[1], args.out, top_k=args.top_k, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
