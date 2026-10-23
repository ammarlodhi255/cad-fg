#!/usr/bin/env python3
import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


LINE = re.compile(r"\s*p(\d+)_v(\d+)_(\d+)")


def cadica_keyframes(cadica_root):
    keys = set()
    pattern = os.path.join(cadica_root, "selectedVideos", "p*", "**", "p*_v*_selectedFrames.txt")
    for f in glob.glob(pattern, recursive=True):
        with open(f) as fh:
            for line in fh:
                m = LINE.match(line)
                if m:
                    keys.add((int(m[1]), int(m[2]), int(m[3])))
    return keys


def flag(df, keys):
    # True for index rows that are not CADICA keyframes
    return np.array([k not in keys for k in zip(df["patient"], df["video"], df["frame"])])


def test_metrics(df):
    y, p = df["y_true"].to_numpy(), df["p_lesion"].to_numpy()
    yhat = (p >= 0.5).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
    return {"n": int(len(df)), "f1": round(100 * 2 * prec * sens / (prec + sens), 2),
            "sensitivity": round(100 * sens, 2), "specificity": round(100 * spec, 2),
            "bal_acc": round(50 * (sens + spec), 2), "auroc": round(100 * roc_auc_score(y, p), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadica_root", default="data/CADICA")
    ap.add_argument("--index_dir", default="outputs/index")
    ap.add_argument("--xttn_dir", default="outputs/final_test/xttn")
    ap.add_argument("--imageonly_dir", default="outputs/final_test/imageonly")
    ap.add_argument("--xttn_retrained_dir", default="outputs/final_test/xttn_retrained")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keys = cadica_keyframes(args.cadica_root)
    out = {"cadica_keyframes": len(keys), "splits": {}, "test_metrics": {}}
    for name in ["all_keyframes", "trainval_pool", "test_pristine"]:
        df = pd.read_csv(os.path.join(args.index_dir, f"{name}.csv"))
        extra = flag(df, keys)
        out["splits"][name] = {
            "frames": int(len(df)), "extra_frames": int(extra.sum()),
            "extra_positive": int(df.loc[extra, "label"].sum()),
            "extra_negative": int((df.loc[extra, "label"] == 0).sum()),
            "keyframes_positive": int(df.loc[~extra, "label"].sum()),
            "keyframes_negative": int((df.loc[~extra, "label"] == 0).sum())}
    runs = {"xttn_published": args.xttn_dir, "imageonly": args.imageonly_dir, "xttn_retrained": args.xttn_retrained_dir}
    for name, d in runs.items():
        df = pd.read_csv(os.path.join(d, "frame_predictions.csv"))
        full, kf = test_metrics(df), test_metrics(df[~flag(df, keys)])
        out["test_metrics"][name] = {"all_frames": full, "keyframes_only": kf,
                                     "max_abs_change_pts": round(max(abs(full[k] - kf[k]) for k in
                                                                     ["f1", "sensitivity", "specificity", "bal_acc", "auroc"]), 2)}
    with open(os.path.join(args.out_dir, "keyframe_index_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
