"""Generate submission using ALS + optional EASE fusion and per-user candidates.

Reads parameters from CLI or from `outputs/best_config.json` when available.
"""
import argparse
import csv
import json
import os
import time
from typing import List

import numpy as np

from data_loader import load_data
from als_model import ALSRecommender
from ease_model import EASERecommender


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out", type=str, default="outputs/submission_hybrid.csv")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--from_best", action='store_true', help="Load config from outputs/best_config.json")
    # fallback args
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--candidate_size", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--beta", type=float, default=0.0, help="EASE weight")
    p.add_argument("--per_user_cand", type=int, default=0)
    p.add_argument("--use_ease", action='store_true')
    p.add_argument("--ease_l2", type=float, default=250.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = None
    if args.from_best and os.path.exists('outputs/best_config.json'):
        try:
            with open('outputs/best_config.json', 'r', encoding='utf-8') as f:
                j = json.load(f)
                cfg = j.get('config', None)
        except Exception:
            cfg = None

    if cfg is None:
        cfg = {
            'factors': args.factors,
            'iterations': args.iterations,
            'candidate_size': args.candidate_size,
            'alpha': args.alpha,
            'beta': args.beta,
            'per_user_cand': args.per_user_cand,
            'use_ease': args.use_ease,
            'ease_l2': args.ease_l2,
        }

    print('Loading data...')
    train_df, test_users, _ = load_data(args.data_dir)
    print('Training ALS...')
    als = ALSRecommender(factors=cfg['factors'], iterations=cfg['iterations'])
    als.fit(train_df)

    ease = None
    if cfg.get('use_ease'):
        print('Training EASE...')
        ease = EASERecommender(l2=cfg.get('ease_l2', 250.0))
        ease.fit(train_df)
        print(f'EASE trained: users={len(ease.user_index)}, items={len(ease.item_index)}')

    pop_items = sorted(als.popularity.items(), key=lambda x: -x[1])
    candidate_size = min(cfg.get('candidate_size', 2000), len(pop_items))
    candidate_indices: List[int] = [iid for iid, _ in pop_items[:candidate_size]]

    pops = [als.popularity.get(i, 0) for i in candidate_indices]
    maxp = max(pops) if len(pops) > 0 else 1
    pop_arr = np.array([p / maxp for p in pops])
    pop_scores = {candidate_indices[i]: (pops[i] / maxp) for i in range(len(candidate_indices))}

    users = test_users.tolist()
    # Diagnostic: print candidate / index mapping info
    print('Diagnostic:')
    print('  total_candidates (requested):', cfg.get('candidate_size'))
    print('  candidate_indices (actual):', len(candidate_indices))
    # sample some candidate indices
    print('  sample_candidate_indices[:10]:', candidate_indices[:10])
    # ALS mappings
    print('  ALS mappings: internal_items=', len(als.item_index), 'index_to_item=', len(als.index_to_item))
    # EASE mappings (if present)
    if ease is not None:
        print('  EASE mappings: internal_items=', len(ease.item_index), 'user_count=', len(ease.user_index))
        try:
            print('  EASE _B shape:', getattr(ease, '_B').shape)
        except Exception:
            pass
        mapped = sum(1 for ext in [als.index_to_item[idx] for idx in candidate_indices] if ease.item_index.get(ext, None) is not None)
        print('  candidate -> ease mapped count:', mapped)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    written = 0
    start = time.time()

    print(f'Generating submission to {args.out} (top_k={args.top_k})')
    with open(args.out, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['user_id', 'item_id'])

        for i in range(0, len(users), args.batch_size):
            batch = users[i: i + args.batch_size]
            scores_batch = als.score_candidates_batch(batch, candidate_indices)

            # prepare EASE per-batch matrices if needed
            if ease is not None:
                # map candidate external ids to ease internal
                cand_ext = [als.index_to_item[idx] for idx in candidate_indices]
                ease_internal = [ease.item_index.get(ext, None) for ext in cand_ext]
                # prepare array and mask for fast indexing
                import numpy as _np
                ease_internal_arr = _np.array([e if e is not None else -1 for e in ease_internal], dtype=int)

                U_idxs = [ease.user_index.get(u, None) for u in batch]
                present_pos_list = [x for x in U_idxs if x is not None]
                if len(present_pos_list) > 0:
                    ui_rows = ease.user_items_matrix[present_pos_list]
                    # only multiply columns of B corresponding to candidate ease indices to save work
                    mask = ease_internal_arr != -1
                    if mask.any():
                        idxs = ease_internal_arr[mask]
                        # B_sub shape: (n_items, num_candidates_with_ease)
                        B_sub = ease._B[:, idxs]
                        # avoid extremely large matrix multiply — skip EASE for this batch if too big
                        est_cost = ui_rows.shape[0] * len(idxs)
                        if est_cost > 5_000_000:
                            print(f'Skipping EASE batch multiply (estimated cost {est_cost}), will treat EASE as disabled for this batch', flush=True)
                            ease_scores_all = None
                        else:
                            ease_scores_all = ui_rows.dot(B_sub)
                    else:
                        ease_scores_all = None

                    if ease_scores_all is not None and hasattr(ease_scores_all, 'toarray'):
                        ease_scores_all = ease_scores_all.toarray()
                    # map from user_internal_idx -> row position in ease_scores_all
                    present_pos_map = {uid: pos for pos, uid in enumerate(present_pos_list)}
                else:
                    ease_scores_all = None
                    present_pos_map = {}
            else:
                ease_scores_all = None

            # debug: print candidate/ease mapping stats for this batch
            if ease is not None:
                try:
                    mapped_cands = int((ease_internal_arr != -1).sum())
                except Exception:
                    mapped_cands = 0
                print(f'Batch {i//args.batch_size}: users={len(batch)}; candidate_size={len(candidate_indices)}; ease_mapped_candidates={mapped_cands}; ease_users_in_batch={len(present_pos_list)}', flush=True)
            else:
                print(f'Batch {i//args.batch_size}: users={len(batch)}; candidate_size={len(candidate_indices)}; ease_disabled', flush=True)

            for bi, u in enumerate(batch):
                # combine scores
                # vectorized combination per user using precomputed arrays/maps
                combined = {}
                # ALS scores for this user over candidate_indices
                user_scores_dict = scores_batch.get(u, {})
                # build numpy arrays for faster ops
                import numpy as _np
                als_scores_arr = _np.array([user_scores_dict.get(idx, 0.0) for idx in candidate_indices], dtype=float)
                pop_arr_local = _np.array([pop_scores.get(idx, 0.0) for idx in candidate_indices], dtype=float)

                ease_arr_for_user = _np.zeros(len(candidate_indices), dtype=float)
                if ease is not None and ease_scores_all is not None:
                    u_internal = ease.user_index.get(u, None)
                    if u_internal is not None and u_internal in present_pos_map:
                        pos = present_pos_map[u_internal]
                        # gather ease scores for available candidate ease indices
                        mask = ease_internal_arr != -1
                        if mask.any():
                            idxs = ease_internal_arr[mask]
                            ease_vals = _np.zeros(len(candidate_indices), dtype=float)
                            # ease_scores_all columns correspond to idxs in the same order
                            ease_vals[mask] = _np.asarray(ease_scores_all[pos, :len(idxs)])
                            ease_arr_for_user = ease_vals

                a = cfg.get('alpha', 1.0)
                b = cfg.get('beta', 0.0)
                combined_scores_arr = a * als_scores_arr + b * ease_arr_for_user + (1.0 - a - b) * pop_arr_local
                for pos_idx, idx in enumerate(candidate_indices):
                    combined[idx] = float(combined_scores_arr[pos_idx])

                # add per-user candidates (use batch recommend when available)
                if cfg.get('per_user_cand', 0) and u in als.user_index:
                    # recommend per-user via batch API to avoid per-user overhead
                    per_batch = als.recommend_batch([u], N=cfg.get('per_user_cand'), filter_already_liked=False)
                    per = per_batch.get(u, [])
                    for item in per:
                        if item in als.item_index:
                            idx = als.item_index[item]
                            if idx not in combined:
                                combined[idx] = cfg.get('alpha', 1.0) * 1.0 + (1.0 - cfg.get('alpha', 1.0)) * pop_scores.get(idx, 0.0)

                if len(combined) == 0:
                    top_items = []
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

                for iid in top_items[: args.top_k]:
                    writer.writerow([u, iid])
                    written += 1

                # progress logging every ~1000 users processed
                processed = min(i + args.batch_size, len(users))
                if processed % 1000 == 0 or processed == len(users):
                    elapsed_batch = time.time() - start
                    print(f'Processed {processed}/{len(users)} users; written {written}; elapsed {elapsed_batch:.1f}s', flush=True)

    elapsed = time.time() - start
    print(f'Wrote submission to {args.out}; rows: {written}; time: {elapsed:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
