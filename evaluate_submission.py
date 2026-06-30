"""Evaluate a submission CSV against a per-user holdout from the full training set.

Usage:
  python src/evaluate_submission.py --sub outputs/submission_opt_hyper2.csv
"""
import argparse
import random
from typing import Dict

import numpy as np
import pandas as pd

from data_loader import load_data


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


def compute_f1(pred_map: Dict[int, set], val_map: Dict[int, set]):
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
    p.add_argument("--sub", type=str, required=True, help="Path to submission CSV")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--holdout", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading data...")
    df, test_users, _ = load_data(args.data_dir)
    print(f"Total interactions: {len(df)}")

    print("Creating train/val holdout from full training set...")
    train_df, val_df = train_val_split(df, holdout_per_user=args.holdout, seed=args.seed)
    print(f"Train rows: {len(train_df)}; Val rows: {len(val_df)}")

    val_group = val_df.groupby("user_id")
    val_map = {u: set(g["item_id"].tolist()) for u, g in val_group}

    print(f"Loading submission from {args.sub}...")
    sub = pd.read_csv(args.sub)
    sub_group = sub.groupby("user_id")
    pred_map = {u: set(g["item_id"].tolist()) for u, g in sub_group}

    # restrict evaluation to users present in val_map
    eval_users = list(val_map.keys())
    # ensure every eval user has an entry in pred_map
    for u in eval_users:
        if u not in pred_map:
            pred_map[u] = set()

    P, R, F1, TP, FP, FN = compute_f1(pred_map, val_map)
    print(f"Evaluation on holdout (users={len(eval_users)}): P={P:.6f} R={R:.6f} F1={F1:.6f} TP={TP} FP={FP} FN={FN}")


if __name__ == "__main__":
    main()
