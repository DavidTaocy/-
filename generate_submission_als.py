"""Train ALS on the full training set and generate a submission CSV.

Usage:
python src/generate_submission_als.py --data_dir data --out outputs/submission.csv \
    --top_k 10 --factors 64 --iterations 15 --candidate_size 5000 --alpha 0.25
"""
import argparse
import os
import csv
import time
from typing import List

import numpy as np
import pandas as pd

from data_loader import load_data
from als_model import ALSRecommender


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out", type=str, default="outputs/submission.csv")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--candidate_size", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.25, help="weight for ALS score; (1-alpha) used for popularity")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--diag_users", type=int, default=0, help="number of users to sample for detailed diagnostics")
    p.add_argument("--diag_out", type=str, default="outputs/diag_samples.txt", help="diagnostic output path")
    p.add_argument("--per_user_cand", type=int, default=0, help="number of per-user top ALS candidates to union with global candidates (0=disabled)")
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading data...")
    train_df, test_users, sample_sub = load_data(args.data_dir)

    print(f"Train rows: {len(train_df)}; Test users: {len(test_users)}")

    print("Training ALS...")
    als = ALSRecommender(factors=args.factors, iterations=args.iterations)
    als.fit(train_df)

    # build candidate list from popularity (internal indices)
    pop_items = sorted(als.popularity.items(), key=lambda x: -x[1])
    candidate_size = min(args.candidate_size, len(pop_items))
    candidate_indices: List[int] = [iid for iid, _ in pop_items[:candidate_size]]

    # normalized popularity scores aligned with candidate_indices
    pops = [als.popularity.get(i, 0) for i in candidate_indices]
    maxp = max(pops) if len(pops) > 0 else 1
    pop_arr = np.array([p / maxp for p in pops])
    # global pop_scores dict for arbitrary internal indices
    pop_scores = {candidate_indices[i]: (pops[i] / maxp) for i in range(len(candidate_indices))}
    global_max_pop = maxp

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    start_t = time.time()

    tmp_out = args.out + ".tmp"
    print(f"Generating submission to {args.out} (temp: {tmp_out})\n  top_k={args.top_k}, alpha={args.alpha}, factors={args.factors}, iterations={args.iterations}, candidate_size={candidate_size}, per_user_cand={args.per_user_cand}")
    with open(tmp_out, "w", newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id"])

        users = test_users.tolist()
        total_users = len(users)
        print('candidate pool size (len(pop_items)):', len(pop_items))
        print('using candidate_size:', candidate_size)
        print('number of test users:', total_users)
        flush_every = max(1, args.batch_size * 4)
        written_since_flush = 0
        batches = 0
        try:
            for i in range(0, len(users), args.batch_size):
                batch = users[i: i + args.batch_size]

                # prepare factors for per-user candidate if needed
                item_factors = getattr(als.model, 'user_factors', None)
                user_factors = getattr(als.model, 'item_factors', None)

                if args.per_user_cand and args.per_user_cand > 0 and item_factors is not None and user_factors is not None:
                    # compute per-user top-N across all items for users present
                    U_idxs = [als.user_index[u] for u in batch if u in als.user_index]
                    if len(U_idxs) > 0:
                        U = user_factors[U_idxs]
                        scores_all = U.dot(item_factors.T)
                    else:
                        scores_all = None

                # score candidates for batch (global candidate scoring)
                scores_batch = als.score_candidates_batch(batch, candidate_indices)

                for bi, u in enumerate(batch):
                    # if per-user mode and user in model, compute union scores
                    if args.per_user_cand and args.per_user_cand > 0 and u in als.user_index and scores_all is not None:
                        try:
                            pos = U_idxs.index(als.user_index[u])
                        except Exception:
                            pos = None

                        if pos is None:
                            # fallback to global candidates
                            scores_map = scores_batch.get(u, {})
                            sc = np.array([scores_map.get(idx, 0.0) for idx in candidate_indices], dtype=float)
                            combined = args.alpha * sc + (1.0 - args.alpha) * pop_arr
                            if len(combined) <= args.top_k:
                                top_idx = np.argsort(-combined)
                            else:
                                part = np.argpartition(-combined, args.top_k)[: args.top_k]
                                top_idx = part[np.argsort(-combined[part])]
                            top_items = [als.index_to_item[candidate_indices[int(j)]] for j in top_idx if candidate_indices[int(j)] in als.index_to_item]
                        else:
                            row_scores_all = scores_all[pos]
                            # per-user top internal indices
                            if args.per_user_cand >= len(row_scores_all):
                                per_top = np.argsort(-row_scores_all)
                            else:
                                part = np.argpartition(-row_scores_all, args.per_user_cand)[: args.per_user_cand]
                                per_top = part[np.argsort(-row_scores_all[part])]
                            per_top = [int(x) for x in per_top]

                            # union candidate list (preserve order: global popular first)
                            union_set = list(dict.fromkeys(candidate_indices + per_top))

                            # build combined scores for union
                            combs = []
                            for idx_internal in union_set:
                                als_score = float(row_scores_all[int(idx_internal)]) if int(idx_internal) < len(row_scores_all) else 0.0
                                pop_score = pop_scores.get(int(idx_internal), 0.0)
                                combs.append(args.alpha * als_score + (1.0 - args.alpha) * pop_score)
                            combs_arr = np.array(combs)
                            if len(combs_arr) <= args.top_k:
                                top_idx = np.argsort(-combs_arr)
                            else:
                                part = np.argpartition(-combs_arr, args.top_k)[: args.top_k]
                                top_idx = part[np.argsort(-combs_arr[part])]
                            top_items = [als.index_to_item.get(union_set[int(j)], None) for j in top_idx]
                            top_items = [t for t in top_items if t is not None]
                    else:
                        # global candidate scoring path
                        scores_map = scores_batch.get(u, {})
                        if len(scores_map) == 0:
                            top_items = [als.index_to_item[idx] for idx in candidate_indices[: args.top_k]]
                        else:
                            sc = np.array([scores_map.get(idx, 0.0) for idx in candidate_indices], dtype=float)
                            combined = args.alpha * sc + (1.0 - args.alpha) * pop_arr
                            if len(combined) <= args.top_k:
                                top_idx = np.argsort(-combined)
                            else:
                                part = np.argpartition(-combined, args.top_k)[: args.top_k]
                                top_idx = part[np.argsort(-combined[part])]
                            top_items = []
                            for j in top_idx:
                                idx_internal = candidate_indices[int(j)]
                                if idx_internal in als.index_to_item:
                                    top_items.append(als.index_to_item[idx_internal])

                    # write rows
                    for iid in top_items[: args.top_k]:
                        writer.writerow([u, iid])
                        written += 1
                        written_since_flush += 1

                    # diagnostics: sample a few users and dump their raw score lists
                    if args.diag_users and args.diag_users > 0:
                        if i < args.diag_users:
                            # prepare diagnostic lines
                            try:
                                with open(args.diag_out, 'a', encoding='utf-8') as df:
                                    df.write(f"USER {u}\n")
                                    smap = scores_batch.get(u, {})
                                    # list top 50 internal indices by score
                                    if len(smap) > 0:
                                        items_scores = sorted(smap.items(), key=lambda x: -x[1])[:50]
                                        for idx, sc in items_scores:
                                            ext = als.index_to_item.get(int(idx), 'MISSING')
                                            df.write(f"  internal:{idx} -> item:{ext} score:{sc:.6f}\n")
                                    else:
                                        df.write("  NO SCORES\n")
                            except Exception:
                                pass

                # periodic flush to make progress visible and safeguard partial output
                if written_since_flush >= flush_every:
                    try:
                        fout.flush()
                    except Exception:
                        pass
                    written_since_flush = 0

                batches += 1
                if batches % 10 == 0:
                    percent = min(100.0, (i + len(batch)) / total_users * 100.0)
                    print(f"Progress: batch {batches}, users processed: {i + len(batch)}/{total_users} ({percent:.1f}%), rows written: {written}")
        except KeyboardInterrupt:
            print("Interrupted by user; flushing and closing temporary file.")
            try:
                fout.flush()
            except Exception:
                pass
            raise
        except Exception as e:
            print("Error while generating submission:", e)
            import traceback

            traceback.print_exc()
            try:
                fout.flush()
            except Exception:
                pass
            raise

    # move temp file to final output atomically
    try:
        os.replace(tmp_out, args.out)
    except Exception:
        # fallback to rename
        try:
            os.rename(tmp_out, args.out)
        except Exception:
            print("Warning: couldn't move temp file to final destination; temp file left at", tmp_out)
    elapsed = time.time() - start_t
    print(f"Wrote submission to {args.out}; rows: {written}; time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
