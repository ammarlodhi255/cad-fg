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
    pl = df.groupby("patient")["label"].max().reset_index().rename(columns={"label": "patient_label"})
    return pl


def _cap_val_frames(df_val: pd.DataFrame, max_pos: int, max_neg: int, seed: int) -> pd.DataFrame:
    """Downsample frame-level rows in val to satisfy label caps.

    - Keeps at most max_pos rows with label==1 and at most max_neg rows with label==0.
    - If counts are already under caps, returns df_val unchanged.
    - Sampling is uniform over frames and reproducible via seed.
    """
    if df_val.empty or ("label" not in df_val.columns):
        return df_val
    rs = np.random.RandomState(seed)
    pos = df_val[df_val["label"] == 1]
    neg = df_val[df_val["label"] == 0]

    take_pos = min(len(pos), int(max_pos) if max_pos is not None else len(pos))
    take_neg = min(len(neg), int(max_neg) if max_neg is not None else len(neg))

    if take_pos < len(pos):
        pos = pos.sample(n=take_pos, random_state=rs, replace=False)
    if take_neg < len(neg):
        neg = neg.sample(n=take_neg, random_state=rs, replace=False)

    capped = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=rs).reset_index(drop=True)
    return capped


def save_fold_csvs(all_df: pd.DataFrame, train_pids: np.ndarray, val_pids: np.ndarray, out_dir: str,
                   max_pos: int, max_neg: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    tr = all_df[all_df["patient"].isin(train_pids)].reset_index(drop=True)
    va_full = all_df[all_df["patient"].isin(val_pids)].reset_index(drop=True)
    va = _cap_val_frames(va_full, max_pos=max_pos, max_neg=max_neg, seed=seed)

    tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    va.to_csv(os.path.join(out_dir, "val.csv"), index=False)

    # Return quick stats for manifest
    def _count_labels(df: pd.DataFrame):
        if df.empty or ("label" not in df.columns):
            return 0, 0, 0
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        return pos, neg, int(pos + neg)

    va_pos, va_neg, va_total = _count_labels(va)
    va_full_pos, va_full_neg, va_full_total = _count_labels(va_full)
    return {
        "val_frames_pos": va_pos,
        "val_frames_neg": va_neg,
        "val_frames_total": va_total,
        "val_frames_pos_before": va_full_pos,
        "val_frames_neg_before": va_full_neg,
        "val_frames_total_before": va_full_total,
        "val_patients_after": int(len(pd.unique(va["patient"])) if (not va.empty and "patient" in va.columns) else 0),
    }


def main():
    ap = argparse.ArgumentParser(description="Create k-fold patient-grouped CV CSVs with capped outer val frame counts.")
    ap.add_argument("--all_csv", default="outputs/index/all_keyframes.csv", help="Full index CSV (entire dataset)")
    ap.add_argument("--train_csv", default="outputs/index/train.csv", help="Train split CSV")
    ap.add_argument("--val_csv", default="outputs/index/val.csv", help="Val split CSV")
    ap.add_argument("--exclude_csv", default="outputs/index/test.csv", help="Optional CSV whose patients are excluded in auto mode")
    ap.add_argument("--source", default="all", choices=["all","trainval","auto"], help="Data source for folds: all (entire set), trainval (only train+val), auto (prefer train+val else all minus exclude)")
    ap.add_argument("--out_root", default="outputs/folds/patient_stratified_5fold_capped_val", help="Output root dir")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_max_pos", type=int, default=400, help="Maximum number of positive frames in outer val per fold")
    ap.add_argument("--val_max_neg", type=int, default=260, help="Maximum number of negative frames in outer val per fold")
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

    # 1) Create pristine test (10% by default) at patient level
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
        cap_stats = save_fold_csvs(
            df_rest,
            tr_pids,
            va_pids,
            out_dir,
            max_pos=args.val_max_pos,
            max_neg=args.val_max_neg,
            seed=args.seed + fold,
        )

        # patient-level stats for sanity (before frame capping)
        tr_y = y[tr_idx]
        va_y = y[va_idx]
        info = {
            "fold": fold,
            "n_train_patients": int(len(tr_pids)),
            "n_val_patients_before": int(len(va_pids)),
            "train_pos_patients": int(tr_y.sum()),
            "train_neg_patients": int(len(tr_y) - tr_y.sum()),
            "val_pos_patients_before": int(va_y.sum()),
            "val_neg_patients_before": int(len(va_y) - va_y.sum()),
            **cap_stats,
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
        "val_caps": {"max_pos": int(args.val_max_pos), "max_neg": int(args.val_max_neg)},
    }
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump({"meta": meta, "folds": folds_manifest}, f, indent=2)

    print(f"[done] wrote {args.n_splits} folds under {args.out_root}")
    print(f"[info] source={src_note}; splitter={split_mode}; excluded_n={len(excluded_patients)}; val_caps(pos<={args.val_max_pos}, neg<={args.val_max_neg})")
    for m in folds_manifest:
        print(
            f"fold {m['fold']}: train_pat={m['n_train_patients']} (pos_pat={m['train_pos_patients']},neg_pat={m['train_neg_patients']}), "
            f"val_pat_before={m['n_val_patients_before']} (pos_pat={m['val_pos_patients_before']},neg_pat={m['val_neg_patients_before']}), "
            f"val_frames={m['val_frames_total']} (pos={m['val_frames_pos']},neg={m['val_frames_neg']})"
        )


if __name__ == "__main__":
    main()

