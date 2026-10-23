#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ORDER = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]
GROUPS = {"<20%": ["p0_20"], "20-50%": ["p20_50"], "50-70%": ["p50_70"], "70-90%": ["p70_90"],
          ">=90%": ["p90_98", "p99", "p100"]}
ENDPOINTS = {">=50%": ORDER[2:], ">=70%": ORDER[3:]}


def frame_band(img_path):
    # most severe stenosis band among the CADICA boxes of the frame (last field of each groundtruth line)
    gt = img_path.replace("/input/", "/groundtruth/").replace(".png", ".txt")
    if not os.path.isfile(gt):
        return None
    bands = [line.split()[-1] for line in open(gt) if line.strip()]
    return max(bands, key=ORDER.index) if bands else None


def boot_auc_ci(y, s, boots):
    vals = []
    for idx in boots:
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, s[idx]))
    return [100 * float(np.percentile(vals, 2.5)), 100 * float(np.percentile(vals, 97.5))], len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xttn_dir", default="outputs/final_test/xttn")
    ap.add_argument("--imageonly_dir", default="outputs/final_test/imageonly")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models = {"YOLOv5l-Xttn": args.xttn_dir, "image-only YOLOv5l": args.imageonly_dir}
    preds = {m: pd.read_csv(os.path.join(d, "frame_predictions.csv")) for m, d in models.items()}
    base = preds["YOLOv5l-Xttn"][["patient", "video", "frame", "img_path", "y_true"]].copy()
    base["band"] = base["img_path"].map(frame_band)
    for m, p in preds.items():
        assert (p["img_path"].values == base["img_path"].values).all(), "prediction files not aligned"
        base[m] = p["p_lesion"].values
    neg = base["y_true"] == 0
    assert base.loc[neg, "band"].isna().all(), "a lesion-free frame has a box"

    vids = base.groupby(["patient", "video"]).indices
    keys = list(vids)
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([vids[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(args.n_boot)]

    res = {"n_lesion_free": int(neg.sum()),
           "positives_without_box": int(((base["y_true"] == 1) & base["band"].isna()).sum()),
           "groups": {}, "endpoints": {}}
    sets = {**{g: base["band"].isin(b) for g, b in GROUPS.items()},
            **{e: base["band"].isin(b) for e, b in ENDPOINTS.items()}}
    y_all = base["y_true"].to_numpy()
    for name, mask in sets.items():
        keep = (mask | neg).to_numpy()
        entry = {"n_frames": int(mask.sum()), "patients": sorted(base.loc[mask, "patient"].unique().tolist()),
                 "n_videos": int(base.loc[mask].groupby(["patient", "video"]).ngroups)}
        # resample videos, then keep this band's positives and all lesion-free frames
        sub_boots = [b[keep[b]] for b in boots]
        for m in models:
            s_all = base[m].to_numpy()
            ci, n_ok = boot_auc_ci(y_all, s_all, sub_boots)
            entry[m] = {"sensitivity": 100 * float((base.loc[mask, m] >= 0.5).mean()),
                        "auroc": 100 * roc_auc_score(y_all[keep], s_all[keep]),
                        "auroc_ci95": ci, "n_valid_resamples": n_ok}
        (res["groups"] if name in GROUPS else res["endpoints"])[name] = entry

    counts = {}
    for gt in glob.glob(os.path.join("data", "CADICA", "selectedVideos", "p*", "v*", "groundtruth", "*.txt")):
        for line in open(gt):
            if line.strip():
                b = line.split()[-1]
                counts[b] = counts.get(b, 0) + 1
    res["dataset_boxes_by_band"] = {b: counts.get(b, 0) for b in ORDER}

    for kind in ("groups", "endpoints"):
        print(f"\n{kind}")
        for name, e in res[kind].items():
            line = f"  {name:7s} n={e['n_frames']:3d} videos={e['n_videos']:2d} patients={e['patients']}"
            for m in models:
                lo, hi = e[m]["auroc_ci95"]
                line += f" | {m}: sens {e[m]['sensitivity']:6.2f} AUROC {e[m]['auroc']:6.2f} [{lo:.2f}, {hi:.2f}]"
            print(line)
    print("\ndataset boxes by band:", res["dataset_boxes_by_band"])
    with open(os.path.join(args.out_dir, "severity_stratified_eval.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
