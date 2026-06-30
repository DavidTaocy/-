"""Validate submission.csv format and create a zip package for submission."""
import os
import sys
import pandas as pd
import zipfile


def validate_and_zip(sub_path: str, zip_path: str):
    if not os.path.exists(sub_path):
        print(f"ERROR: submission file not found: {sub_path}")
        sys.exit(2)

    df = pd.read_csv(sub_path)

    # Basic checks
    expected_cols = ["user_id", "item_id"]
    cols_ok = list(df.columns) == expected_cols
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print("ERROR: missing columns:", missing)
        sys.exit(3)

    # check for extra columns
    extra = [c for c in df.columns if c not in expected_cols]
    if extra:
        print("ERROR: extra columns present:", extra)
        sys.exit(4)

    # check for nulls
    nulls = df[expected_cols].isnull().any()
    if nulls.any():
        print("ERROR: Null values found in columns:", nulls[nulls].index.tolist())
        sys.exit(5)

    # try coerce to integer
    try:
        df["user_id"] = df["user_id"].astype(int)
        df["item_id"] = df["item_id"].astype(int)
    except Exception as e:
        print("ERROR: user_id/item_id must be integers:", e)
        sys.exit(6)

    # report basic stats
    n_rows = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    print(f"OK: rows={n_rows}, unique_users={n_users}, unique_items={n_items}")

    # create zip
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(sub_path, arcname=os.path.basename(sub_path))

    print(f"WROTE: {zip_path}")


if __name__ == "__main__":
    sub = os.path.join("outputs", "submission.csv")
    z = os.path.join("outputs", "submission.zip")
    if len(sys.argv) >= 2:
        sub = sys.argv[1]
    if len(sys.argv) >= 3:
        z = sys.argv[2]
    validate_and_zip(sub, z)
