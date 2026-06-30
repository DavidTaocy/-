"""Optimized pipeline: multi-strategy recall, rerank, grid search, submission export.

Usage examples:
  python src/optimize_pipeline.py --data_dir data --top_k 10 --candidate_size 2000
  python src/optimize_pipeline.py --grid_search --samples 20

This script strictly uses files inside the `data` folder and writes outputs
to `outputs/best_config.json` and `outputs/submission.csv`.
"""
from typing import Dict, List, Set, Tuple
import argparse
import json
import os
import time
import itertools
import random

import numpy as np
import pandas as pd

from data_loader import load_data
from als_model import ALSRecommender
from model_mf import MFRecommender
from model_cf import UserBasedCF


def train_val_split(interactions: pd.DataFrame, holdout_per_user: int = 1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    users = interactions['user_id'].unique()
    train_idx = []
    val_idx = []
    grouped = interactions.groupby('user_id')
    for u, g in grouped:
        idxs = g.index.tolist()
        if len(idxs) <= holdout_per_user:
            train_idx.extend(idxs)
            continue
        holdout = random.sample(idxs, holdout_per_user)
        for i in idxs:
            if i in holdout:
                val_idx.append(i)
            else:
                train_idx.append(i)
    return interactions.loc[train_idx].reset_index(drop=True), interactions.loc[val_idx].reset_index(drop=True)


def compute_f1(pred_map: Dict[int, Set[int]], val_map: Dict[int, Set[int]]) -> Tuple[float, float, float]:
    TP = FP = FN = 0
    for u, true_items in val_map.items():
        preds = set(pred_map.get(u, []))
        TP += len(preds & true_items)
        FP += len(preds - true_items)
        FN += len(true_items - preds)
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return P, R, F1


def build_features(interactions: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, float]]:
    # user read count and item popularity (raw and normalized)
    user_counts = interactions['user_id'].value_counts().to_dict()
    item_counts = interactions['item_id'].value_counts().to_dict()
    max_item = max(item_counts.values()) if len(item_counts) > 0 else 1
    item_pop_norm = {iid: cnt / max_item for iid, cnt in item_counts.items()}
    return user_counts, item_pop_norm


def generate_candidates(users: List[int], als: ALSRecommender, mf: MFRecommender, cf: UserBasedCF, candidate_size: int) -> Dict[int, List[int]]:
    # collect per-method candidate lists and merge
    cand_map: Dict[int, List[int]] = {u: [] for u in users}

    # ALS candidates (external ids)
    als_res = als.recommend_batch(users, N=candidate_size, filter_already_liked=False)
    for u, lst in als_res.items():
        cand_map[u].extend(lst)

    # MF candidates
    try:
        mf_res = mf.recommend_batch(users, k=candidate_size, exclude_interacted=False)
    except Exception:
        mf_res = {u: [] for u in users}
    for u, lst in mf_res.items():
        cand_map[u].extend(lst)

    # User-based CF candidates
    for u in users:
        try:
            cf_list = cf.recommend(u, k=min(200, candidate_size), exclude_interacted=False)
        except Exception:
            cf_list = []
        cand_map[u].extend(cf_list)

    # deduplicate preserving order
    for u in users:
        seen = set()
        out = []
        for it in cand_map[u]:
            if it not in seen:
                seen.add(it)
                out.append(it)
            if len(out) >= candidate_size:
                break
        cand_map[u] = out
    return cand_map


def score_and_rank(users: List[int], cand_map: Dict[int, List[int]], als: ALSRecommender, mf: MFRecommender, cf: UserBasedCF, item_pop_norm: Dict[int, float],
                   w_als: float, w_mf: float, w_cf: float, pop_penalty: float, top_k: int) -> Dict[int, List[int]]:
    pred_map: Dict[int, List[int]] = {}

    # prepare popularity array for internal scoring where possible
    for u in users:
        cands = cand_map.get(u, [])
        if len(cands) == 0:
            pred_map[u] = []
            continue

        # ALS requires internal indices
        candidate_internal = [als.item_index.get(it, None) for it in cands]
        # map None -> skip later
        als_scores = {}
        try:
            valid_idxs = [i for i in candidate_internal if i is not None]
            if len(valid_idxs) > 0 and u in als.user_index:
                sdict = als.score_candidates(u, valid_idxs)
                # map back to external ids
                for internal_idx, sc in sdict.items():
                    ext = als.index_to_item.get(internal_idx)
                    if ext is not None:
                        als_scores[ext] = sc
        except Exception:
            als_scores = {}

        # MF scores via dot if available
        mf_scores = {}
        try:
            if u in mf.user_index:
                uidx = mf.user_index[u]
                uvec = mf.user_factors_np[uidx]
                for it in cands:
                    j = mf.item_index.get(it, None)
                    if j is None:
                        continue
                    mf_scores[it] = float(uvec.dot(mf.item_factors_np[j]))
        except Exception:
            mf_scores = {}

        # CF scores: boolean 1 if present in top-k list from cf.recommend
        cf_top = set()
        try:
            cf_list = cf.recommend(u, k=200, exclude_interacted=False)
            cf_top = set(cf_list)
        except Exception:
            cf_top = set()

        combined: List[Tuple[int, float]] = []
        for it in cands:
            a = als_scores.get(it, 0.0)
            m = mf_scores.get(it, 0.0)
            c = 1.0 if it in cf_top else 0.0
            pop = item_pop_norm.get(it, 0.0)
            score = w_als * a + w_mf * m + w_cf * c - pop_penalty * pop
            combined.append((it, score))

        combined.sort(key=lambda x: -x[1])
        pred_map[u] = [it for it, _ in combined[:top_k]]

    return pred_map


def evaluate(train_df: pd.DataFrame, val_map: Dict[int, Set[int]], config: Dict) -> Tuple[float, float, float]:
    # fit models
    als = ALSRecommender(factors=config.get('als_factors', 64), regularization=config.get('als_reg', 0.01), iterations=config.get('als_iter', 15))
    als.fit(train_df)
    mf = MFRecommender(n_components=config.get('mf_components', 64))
    mf.fit(train_df)
    cf = UserBasedCF(topk_neighbors=config.get('cf_neighbors', 50))
    cf.fit(train_df)

    users = list(val_map.keys())
    cand_map = generate_candidates(users, als, mf, cf, candidate_size=config.get('candidate_size', 2000))
    _, item_pop_norm = build_features(train_df)
    pred_map = score_and_rank(users, cand_map, als, mf, cf, item_pop_norm,
                              w_als=config.get('w_als', 0.5), w_mf=config.get('w_mf', 0.2), w_cf=config.get('w_cf', 0.2),
                              pop_penalty=config.get('pop_penalty', 0.0), top_k=config.get('top_k', 10))

    P, R, F1 = compute_f1(pred_map, val_map)
    return P, R, F1


def grid_search(train_df: pd.DataFrame, val_map: Dict[int, Set[int]], space: Dict, max_evals: int = 50, seed: int = 42):
    random.seed(seed)
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))
    if max_evals and max_evals > 0 and len(combos) > max_evals:
        combos = random.sample(combos, max_evals)

    best = None
    results = []
    for comb in combos:
        cfg = {k: v for k, v in zip(keys, comb)}
        cfg['top_k'] = cfg.get('top_k', 10)
        t0 = time.time()
        try:
            P, R, F1 = evaluate(train_df, val_map, cfg)
        except Exception as e:
            print('error evaluating cfg', cfg, '->', e)
            continue
        dt = time.time() - t0
        results.append((cfg, P, R, F1, dt))
        print(f"eval cfg={cfg} -> P={P:.6f} R={R:.6f} F1={F1:.6f} time={dt:.1f}s")
        if best is None or F1 > best[0]:
            best = (F1, cfg, P, R)
            try:
                os.makedirs('outputs', exist_ok=True)
                with open('outputs/best_config.json', 'w', encoding='utf-8') as bf:
                    json.dump({'f1': best[0], 'config': best[1], 'P': best[2], 'R': best[3]}, bf, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return best, results


def train_full_and_export(train_df: pd.DataFrame, test_users: List[int], best_cfg: Dict, out_path: str = 'outputs/submission.csv'):
    # retrain models on full train and generate submission
    als = ALSRecommender(factors=best_cfg.get('als_factors', 64), regularization=best_cfg.get('als_reg', 0.01), iterations=best_cfg.get('als_iter', 15))
    als.fit(train_df)
    mf = MFRecommender(n_components=best_cfg.get('mf_components', 64))
    mf.fit(train_df)
    cf = UserBasedCF(topk_neighbors=best_cfg.get('cf_neighbors', 50))
    cf.fit(train_df)

    users = list(test_users)
    cand_map = generate_candidates(users, als, mf, cf, candidate_size=best_cfg.get('candidate_size', 2000))
    _, item_pop_norm = build_features(train_df)
    pred_map = score_and_rank(users, cand_map, als, mf, cf, item_pop_norm,
                              w_als=best_cfg.get('w_als', 0.5), w_mf=best_cfg.get('w_mf', 0.2), w_cf=best_cfg.get('w_cf', 0.2),
                              pop_penalty=best_cfg.get('pop_penalty', 0.0), top_k=best_cfg.get('top_k', 10))

    # write submission with two columns user_id,item_id per row
    rows = []
    for u in users:
        items = pred_map.get(u, [])
        for it in items:
            rows.append({'user_id': int(u), 'item_id': int(it)})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print('Wrote submission to', out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--candidate_size', type=int, default=2000)
    p.add_argument('--grid_search', action='store_true')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--max_train', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    df, test_users, _ = load_data(args.data_dir)
    if args.max_train and args.max_train > 0 and len(df) > args.max_train:
        df = df.sample(args.max_train, random_state=args.seed).reset_index(drop=True)

    train_df, val_df = train_val_split(df, holdout_per_user=1, seed=args.seed)
    val_map = {u: set(g['item_id'].tolist()) for u, g in val_df.groupby('user_id')}

    if args.grid_search:
        space = {
            'als_factors': [32, 64],
            'als_iter': [5, 10],
            'candidate_size': [500, 2000],
            'w_als': [0.4, 0.6],
            'w_mf': [0.1, 0.3],
            'w_cf': [0.0, 0.2],
            'pop_penalty': [0.0, 0.1],
            'top_k': [args.top_k],
        }
        best, results = grid_search(train_df, val_map, space, max_evals=args.samples, seed=args.seed)
        if best is None:
            print('No valid config found')
            return
        f1, best_cfg, P, R = best[0], best[1], best[2], best[3]
        print('Best F1', f1, 'cfg', best_cfg)
        # export submission using best_cfg
        train_full_and_export(df, test_users.tolist(), best_cfg)
    else:
        # quick single-run with default params
        cfg = {'als_factors': 64, 'als_iter': 15, 'candidate_size': args.candidate_size, 'w_als': 0.5, 'w_mf': 0.2, 'w_cf': 0.2, 'pop_penalty': 0.0, 'top_k': args.top_k}
        P, R, F1 = evaluate(train_df, val_map, cfg)
        print('Offline eval -> P:%.6f R:%.6f F1:%.6f' % (P, R, F1))
        train_full_and_export(df, test_users.tolist(), cfg)


if __name__ == '__main__':
    main()
