#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedShuffleSplit


def build_patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not {"patient", "label"}.issubset(df.columns):
        raise KeyError("Input CSV must contain 'patient' and 'label' columns")
    # Patient-level label: 1 if any frame/video positive, else 0
    pl = (
        df.groupby("patient")["label"].max().reset_index().rename(columns={"label": "patient_label"})
    )
    return pl


def save_fold_csvs(all_df: pd.DataFrame, train_pids: np.ndarray, val_pids: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tr = all_df[all_df["patient"].isin(train_pids)].reset_index(drop=True)
    va = all_df[all_df["patient"].isin(val_pids)].reset_index(drop=True)
    tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    va.to_csv(os.path.join(out_dir, "val.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Create k-fold patient-grouped CV CSVs. Supports full dataset or train+val only.")
    ap.add_argument("--all_csv", default="outputs/index/all_keyframes.csv", help="Full index CSV (entire dataset)")
    ap.add_argument("--train_csv", default="outputs/index/train.csv", help="Train split CSV")
    ap.add_argument("--val_csv", default="outputs/index/val.csv", help="Val split CSV")
    ap.add_argument("--exclude_csv", default="outputs/index/test.csv", help="Optional CSV whose patients are excluded in auto mode")
    ap.add_argument("--source", default="all", choices=["all","trainval","auto"], help="Data source for folds: all (entire set), trainval (only train+val), auto (prefer train+val else all minus exclude)")
    ap.add_argument("--out_root", default="outputs/folds/patient_stratified_5fold", help="Output root dir")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # Select source per user choice
    excluded_patients = []
    if args.source == "trainval":
        if not (os.path.exists(args.train_csv) and os.path.exists(args.val_csv)):
            raise FileNotFoundError("--source trainval requires both --train_csv and --val_csv to exist.")
        df_tr = pd.read_csv(args.train_csv)
        df_va = pd.read_csv(args.val_csv)
        df = pd.concat([df_tr, df_va], axis=0).drop_duplicates().reset_index(drop=True)
        src_note = f"train+val ({args.train_csv} + {args.val_csv})"
    elif args.source == "auto":
        if os.path.exists(args.train_csv) and os.path.exists(args.val_csv):
            df_tr = pd.read_csv(args.train_csv)
            df_va = pd.read_csv(args.val_csv)
            df = pd.concat([df_tr, df_va], axis=0).drop_duplicates().reset_index(drop=True)
            src_note = f"train+val ({args.train_csv} + {args.val_csv})"
        else:
            df = pd.read_csv(args.all_csv)
            src_note = args.all_csv
            if args.exclude_csv and os.path.exists(args.exclude_csv):
                try:
                    ex = pd.read_csv(args.exclude_csv)
                    if {"patient"}.issubset(ex.columns):
                        ex_pats = sorted(ex["patient"].dropna().astype(int).unique().tolist())
                        excluded_patients = ex_pats
                        df = df[~df["patient"].isin(ex_pats)].reset_index(drop=True)
                    else:
                        print(f"[warn] exclude_csv has no 'patient' column: {args.exclude_csv}")
                except Exception as e:
                    print(f"[warn] failed to read exclude_csv {args.exclude_csv}: {e}")
    else:  # args.source == 'all'
        df = pd.read_csv(args.all_csv)
        src_note = args.all_csv + " (entire dataset)"

    if df.empty:
        raise RuntimeError("No data available to split after applying inputs/exclusions.")

    # 1) Create pristine test (10% by default) at patient level with stratification if possible
    pl_all = build_patient_labels(df)
    pats_all = pl_all["patient"].values
    y_all = pl_all["patient_label"].values.astype(int)

    test_frac = 0.10
    sss_ok = len(np.unique(y_all)) >= 2
    if sss_ok:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=args.seed)
        test_idx, rest_idx = next(sss.split(pats_all, y_all))
        split_mode_test = "StratifiedShuffleSplit"
    else:
        # fallback: uniform group split when single class present
        rng = np.random.RandomState(args.seed)
        idx = np.arange(len(pats_all))
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(idx))))
        test_idx, rest_idx = idx[:n_test], idx[n_test:]
        split_mode_test = "RandomGroupSplit (no stratification: single-class patients)"

    test_pids = pats_all[test_idx]
    rest_pids = pats_all[rest_idx]

    test_df = df[df["patient"].isin(test_pids)].reset_index(drop=True)
    os.makedirs("outputs/index", exist_ok=True)
    test_out = os.path.join("outputs", "index", "test_pristine.csv")
    test_df.to_csv(test_out, index=False)

    # 2) Build outer CV folds on the remaining patients
    df_rest = df[df["patient"].isin(rest_pids)].reset_index(drop=True)
    # persist combined train+val pool for optional final training
    rest_out = os.path.join("outputs", "index", "trainval_pool.csv")
    df_rest.to_csv(rest_out, index=False)
    pl = build_patient_labels(df_rest)
    patients = pl["patient"].values
    y = pl["patient_label"].values.astype(int)

    uniq = np.unique(y)
    use_strat = len(uniq) >= 2
    if use_strat:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(patients, y)
        split_mode = "StratifiedKFold"
    else:
        splitter = GroupKFold(n_splits=args.n_splits)
        split_iter = splitter.split(np.zeros(len(patients)), groups=patients)
        split_mode = "GroupKFold (no stratification: single-class patients)"

    folds_manifest = []
    for fold, (tr_idx, va_idx) in enumerate(split_iter):
        tr_pids = patients[tr_idx]
        va_pids = patients[va_idx]

        out_dir = os.path.join(args.out_root, f"fold_{fold}")
        save_fold_csvs(df_rest, tr_pids, va_pids, out_dir)

        # simple stats for sanity
        tr_y = y[tr_idx]
        va_y = y[va_idx]
        info = {
            "fold": fold,
            "n_train_patients": int(len(tr_pids)),
            "n_val_patients": int(len(va_pids)),
            "train_pos": int(tr_y.sum()),
            "train_neg": int(len(tr_y) - tr_y.sum()),
            "val_pos": int(va_y.sum()),
            "val_neg": int(len(va_y) - va_y.sum()),
        }
        folds_manifest.append(info)

    # meta manifest
    meta = {
        "source": src_note,
        "excluded_patients": excluded_patients,
        "n_unique_patients_all": int(len(pats_all)),
        "test_splitter": split_mode_test,
        "n_test_patients": int(len(test_pids)),
        "n_rest_patients": int(len(rest_pids)),
        "cv_splitter": split_mode,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "test_csv": test_out,
        "trainval_pool_csv": rest_out,
    }
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump({"meta": meta, "folds": folds_manifest}, f, indent=2)

    print(f"[done] wrote {args.n_splits} folds under {args.out_root}")
    print(f"[info] source={src_note}; splitter={split_mode}; excluded_n={len(excluded_patients)}")
    for m in folds_manifest:
        print(
            f"fold {m['fold']}: train_pat={m['n_train_patients']} (pos={m['train_pos']},neg={m['train_neg']}), "
            f"val_pat={m['n_val_patients']} (pos={m['val_pos']},neg={m['val_neg']})"
        )


if __name__ == "__main__":
    main()
