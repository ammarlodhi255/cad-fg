#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def _counts(df: pd.DataFrame):
    n = int(df.shape[0])
    # be tolerant to non-int labels
    if "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce")
        pos = int((lab == 1).sum())
        neg = int((lab == 0).sum())
    else:
        pos = neg = 0
    return n, pos, neg


def _patient_ct(df: pd.DataFrame) -> int:
    # Robust patient counting: prefer numeric IDs; fallback to string IDs
    if "patient" not in df.columns:
        return 0
    try:
        ser = pd.to_numeric(df["patient"], errors="coerce")
        n = int(ser.dropna().nunique())
        if n > 0:
            return n
        return int(df["patient"].astype(str).str.strip().nunique())
    except Exception:
        return int(df["patient"].astype(str).str.strip().nunique())


def main():
    ap = argparse.ArgumentParser(description="Print patient and class counts for a CSV (e.g., pristine test)")
    ap.add_argument("--csv", default="outputs/index/test_pristine.csv", help="Path to pristine test CSV")
    ap.add_argument("--name", default="pristine-test", help="Label used in the header print")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"[error] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    n, pos, neg = _counts(df)
    pats = _patient_ct(df)

    print(f"[data] {args.name} sizes:")
    print(f"  test: patients={pats}  n={n}  pos={pos}  neg={neg}")


if __name__ == "__main__":
    main()

