"""Main runner: choose model, generate top-K recommendations per test user, output submission CSV.

Usage example:
python src/main.py --method mf --top_k 10 --data_dir data --out outputs/submission.csv
"""
import argparse
import os
import csv
import pandas as pd
import time

from data_loader import load_data
from model_cf import UserBasedCF
from model_mf import MFRecommender


def build_and_run(method: str, top_k: int, data_dir: str, out_path: str):
    train_df, test_users, sample_sub = load_data(data_dir)
    # allow sampling for quick demos
    max_train = getattr(build_and_run, "_max_train", None)
    max_test = getattr(build_and_run, "_max_test", None)
    if max_train is not None and max_train > 0:
        train_df = train_df.sample(min(len(train_df), max_train), random_state=42).reset_index(drop=True)
    if max_test is not None and max_test > 0:
        test_users = test_users.head(max_test)

    # ensure output dir exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if method == "cf":
        model = UserBasedCF()
    elif method == "mf":
        model = MFRecommender(n_components=64)
    elif method == "mf_opt":
        model = MFRecommender(n_components=64)
    elif method == "cf_opt":
        model = UserBasedCF()
    else:
        raise ValueError("method must be 'cf', 'cf_opt', 'mf' or 'mf_opt'")

    print("Fitting model...")
    model.fit(train_df)

    print("Generating recommendations (streaming write)...")
    test_list = test_users.tolist()
    batch_size = getattr(build_and_run, "_batch_size", 512)

    # stream results to CSV to avoid huge memory use
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    start_t = time.time()
    with open(out_path, "w", newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id"])

        if method == "mf_opt":
            # process test users in batches and use recommend_batch
            for i in range(0, len(test_list), batch_size):
                batch_users = test_list[i: i + batch_size]
                rec_map = model.recommend_batch(batch_users, k=top_k)
                for uid in batch_users:
                    recs = rec_map.get(uid, [])
                    for iid in recs:
                        writer.writerow([uid, iid])
                        written += 1
        else:
            for uid in test_list:
                recs = model.recommend(uid, k=top_k)
                for iid in recs:
                    writer.writerow([uid, iid])
                    written += 1

    elapsed = time.time() - start_t
    print(f"Wrote submission to {out_path}; rows: {written}; time: {elapsed:.1f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["cf", "cf_opt", "mf", "mf_opt"], default="mf")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out", type=str, default="outputs/submission.csv")
    p.add_argument("--max_train", type=int, default=0, help="Max train rows to sample (0 = use all)")
    p.add_argument("--max_test", type=int, default=0, help="Max test users to predict (0 = use all)")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for predictions when using optimized mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # attach sampling and batch parameters to function for optional quick demos
    setattr(build_and_run, "_max_train", args.max_train)
    setattr(build_and_run, "_max_test", args.max_test)
    setattr(build_and_run, "_batch_size", args.batch_size)
    build_and_run(args.method, args.top_k, args.data_dir, args.out)
