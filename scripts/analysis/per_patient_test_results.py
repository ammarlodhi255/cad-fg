#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def per_patient(pred_dir):
    f = pd.read_csv(os.path.join(pred_dir, "frame_predictions.csv"))
    rows = {}
    for p, g in f.groupby("patient"):
        y, s = g["y_true"].astype(int).to_numpy(), g["p_lesion"].to_numpy()
        pred = (s >= 0.5).astype(int)
        both = len(np.unique(y)) == 2
        rows[int(p)] = {
            "n_frames": int(len(y)), "n_lesion": int(y.sum()),
            "f1": 100 * f1_score(y, pred, zero_division=0),
            "sensitivity": 100 * float((pred[y == 1] == 1).mean()) if (y == 1).any() else None,
            "specificity": 100 * float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
            "auroc": 100 * roc_auc_score(y, s) if both else None}
    return rows


def counts(df):
    pat = df.groupby("patient")["label"].max()
    vid = df.groupby(["patient", "video"])["label"].max()
    return {"patients": int(len(pat)), "lesion_patients": int(pat.sum()),
            "lesion_free_patients": int((pat == 0).sum()),
            "frames": int(len(df)), "lesion_free_frames": int((df["label"] == 0).sum()),
            "videos": int(len(vid)), "lesion_free_videos": int((vid == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xttn_dir", default="outputs/final_test/xttn")
    ap.add_argument("--imageonly_dir", default="outputs/final_test/imageonly")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--pool_csv", default="outputs/index/trainval_pool.csv")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models = {"YOLOv5l-Xttn": args.xttn_dir, "image-only YOLOv5l": args.imageonly_dir}
    res = {"per_patient_test": {m: per_patient(d) for m, d in models.items()}, "folds": {}}
    for i in range(5):
        res["folds"][f"fold_{i}"] = counts(pd.read_csv(os.path.join(args.folds_root, f"fold_{i}", "val.csv")))
    res["folds"]["test"] = counts(pd.read_csv(args.test_csv))
    res["all_patients"] = counts(pd.concat([pd.read_csv(args.pool_csv), pd.read_csv(args.test_csv)]))

    for m, rows in res["per_patient_test"].items():
        print(m)
        for p, r in rows.items():
            fmt = lambda v: "  --  " if v is None else f"{v:6.2f}"
            print(f"  patient {p:2d}: n={r['n_frames']:3d} lesion={r['n_lesion']:3d}  F1 {fmt(r['f1'])}  "
                  f"sens {fmt(r['sensitivity'])}  spec {fmt(r['specificity'])}  AUROC {fmt(r['auroc'])}")
    print("\nPatients and lesion-free frames/videos per set")
    for k, c in res["folds"].items():
        print(f"  {k:7s} {c}")
    print(f"  all     {res['all_patients']}")
    with open(os.path.join(args.out_dir, "per_patient_test_results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
