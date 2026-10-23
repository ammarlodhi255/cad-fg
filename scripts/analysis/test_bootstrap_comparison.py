#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


PAIRS = [("imageonly", "xttn_published"), ("imageonly", "xttn_retrained"),
         ("xttn_retrained", "xttn_published")]
THR = 0.5
KEYS = ["patient", "video", "frame", "y_true"]


def load(runs):
    frames = None
    for name, d in runs.items():
        df = pd.read_csv(os.path.join(d, "frame_predictions.csv"))
        if frames is None:
            frames = df[KEYS].copy()
        elif not (df[KEYS].values == frames[KEYS].values).all():
            raise SystemExit(f"{d}: test rows differ from the other runs")
        frames[name] = df["p_lesion"].values
    return frames


def frame_metrics(y, p):
    yhat = (p >= THR).astype(int)
    return {"auroc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, yhat), "f1": f1_score(y, yhat)}


def pct(x):
    return round(100 * float(x), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xttn_dir", default="outputs/final_test/xttn")
    ap.add_argument("--imageonly_dir", default="outputs/final_test/imageonly")
    ap.add_argument("--xttn_retrained_dir", default="outputs/final_test/xttn_retrained")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = {"xttn_published": args.xttn_dir, "imageonly": args.imageonly_dir, "xttn_retrained": args.xttn_retrained_dir}

    d = load(runs)
    reported = {}
    for name, run in runs.items():
        with open(os.path.join(run, "metrics.json")) as f:
            m = json.load(f)
        keep = ["f1", "sensitivity", "specificity", "bal_acc", "auc", "precision"]
        reported[name] = {"frame": {k: pct(m["frame"][k]) for k in keep},
                          "video": {k: pct(m["video"][k]) for k in keep}}

    # paired bootstrap resampling whole test videos (frames of a video are correlated)
    groups = list(d.groupby(["patient", "video"]).indices.values())
    rng = np.random.default_rng(args.seed)
    single = {n: [] for n in runs}
    diffs = {f"{a}_minus_{b}": [] for a, b in PAIRS}
    for _ in range(args.n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y = d["y_true"].values[idx]
        if len(np.unique(y)) < 2:
            continue
        m = {n: frame_metrics(y, d[n].values[idx]) for n in runs}
        for n in runs:
            single[n].append([m[n][k] for k in ("auroc", "bal_acc", "f1")])
        for a, b in PAIRS:
            diffs[f"{a}_minus_{b}"].append([m[a][k] - m[b][k] for k in ("auroc", "bal_acc", "f1")])

    def summarise(arr, point=None):
        arr = 100 * np.asarray(arr)
        out = {}
        for j, k in enumerate(("auroc", "bal_acc", "f1")):
            s = {"ci95": [round(float(np.percentile(arr[:, j], 2.5)), 2),
                          round(float(np.percentile(arr[:, j], 97.5)), 2)]}
            if point is not None:
                s["delta"] = round(point[k], 2)
                s["p_two_sided"] = round(float(2 * min((arr[:, j] <= 0).mean(), (arr[:, j] >= 0).mean())), 3)
            out[k] = s
        return out

    full = {n: frame_metrics(d["y_true"].values, d[n].values) for n in runs}
    boot = {"n_boot": args.n_boot, "resampling_unit": f"video ({len(groups)})",
            "per_model": {n: summarise(v) for n, v in single.items()},
            "paired_differences": {
                f"{a}_minus_{b}": summarise(diffs[f"{a}_minus_{b}"],
                                            {k: 100 * (full[a][k] - full[b][k]) for k in full[a]})
                for a, b in PAIRS}}

    per_patient = {}
    for p, g in d.groupby("patient"):
        per_patient[int(p)] = {"frames": int(len(g)), "positive_frames": int(g["y_true"].sum()),
                               "videos": int(g["video"].nunique())}
        for n in runs:
            per_patient[int(p)][n] = {"mean_prob": round(float(g[n].mean()), 3),
                                      "share_called_positive": round(float((g[n] >= THR).mean()), 3)}

    out = {"test_set": {"frames": int(len(d)), "positive_frames": int(d["y_true"].sum()),
                        "videos": len(groups), "patients": int(d["patient"].nunique())},
           "reported_metrics_pct": reported, "bootstrap": boot, "per_patient": per_patient}
    with open(os.path.join(args.out_dir, "test_bootstrap_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"reported": reported, "paired": boot["paired_differences"]}, indent=1))


if __name__ == "__main__":
    main()
