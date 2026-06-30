"""Data loading and preprocessing utilities.

This module detects typical file names in the provided `data_dir` and
returns pandas DataFrames for train interactions, test users, and sample submission.

Assumptions:
- Train file contains at least columns: user_id, item_id
- Test file contains at least column: user_id
"""
from typing import Tuple
import os
import pandas as pd


def _find_file(data_dir: str, candidates):
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_data(data_dir: str = "data", min_user_count: int = 1, min_item_count: int = 1) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load train, test users and sample submission if present.

    Returns:
        train_df: DataFrame with columns at least `user_id`, `item_id`
        test_users: Series of user_id values to predict for
        sample_submission: DataFrame sample (may be empty if not found)
    """
    train_candidates = ["train.csv", "train_dataset.csv", "train_dataset.csv"]
    test_candidates = ["test.csv", "test_dataset.csv"]
    submission_candidates = ["submission.csv", "sample_submission.csv"]

    train_path = _find_file(data_dir, train_candidates)
    test_path = _find_file(data_dir, test_candidates)
    sub_path = _find_file(data_dir, submission_candidates)

    if train_path is None:
        raise FileNotFoundError(f"Train file not found in {data_dir}; looked for {train_candidates}")

    train_df = pd.read_csv(train_path, low_memory=False)

    if not {"user_id", "item_id"}.issubset(set(train_df.columns)):
        # try lowercase variations
        cols = {c.lower(): c for c in train_df.columns}
        if "user_id" in cols and "item_id" in cols:
            train_df = train_df.rename(columns={cols["user_id"]: "user_id", cols["item_id"]: "item_id"})
        else:
            raise ValueError("train file must contain user_id and item_id columns")

    # normalize id types and drop invalid rows
    train_df["user_id"] = pd.to_numeric(train_df["user_id"], errors="coerce")
    train_df["item_id"] = pd.to_numeric(train_df["item_id"], errors="coerce")
    train_df = train_df.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    train_df["user_id"] = train_df["user_id"].astype(int)
    train_df["item_id"] = train_df["item_id"].astype(int)
    # remove duplicate interactions
    train_df = train_df.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

    # optional filtering for very rare users/items
    if min_item_count > 1:
        item_counts = train_df["item_id"].value_counts()
        keep_items = item_counts[item_counts >= min_item_count].index
        train_df = train_df[train_df["item_id"].isin(keep_items)].reset_index(drop=True)
    if min_user_count > 1:
        user_counts = train_df["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_count].index
        train_df = train_df[train_df["user_id"].isin(keep_users)].reset_index(drop=True)

    if test_path is None:
        test_users = pd.Series(dtype=int)
    else:
        test_df = pd.read_csv(test_path, low_memory=False)
        if "user_id" not in test_df.columns:
            cols = {c.lower(): c for c in test_df.columns}
            if "user_id" in cols:
                test_df = test_df.rename(columns={cols["user_id"]: "user_id"})
            else:
                raise ValueError("test file must contain user_id column")
        test_df["user_id"] = pd.to_numeric(test_df["user_id"], errors="coerce")
        test_df = test_df.dropna(subset=["user_id"]).reset_index(drop=True)
        test_df["user_id"] = test_df["user_id"].astype(int)
        test_users = test_df["user_id"].drop_duplicates().reset_index(drop=True)

    if sub_path is None:
        sample_submission = pd.DataFrame()
    else:
        sample_submission = pd.read_csv(sub_path)

    return train_df, test_users, sample_submission


if __name__ == "__main__":
    # quick smoke test
    import sys
    dd = sys.argv[1] if len(sys.argv) > 1 else "data"
    t, u, s = load_data(dd)
    print("Loaded:", t.shape, "train rows; test users:", len(u))
