#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd


def eta_squared(p, groups):
    # share of the variance of the predicted probability explained by patient identity
    within = ((p - p.groupby(groups).transform("mean")) ** 2).sum()
    total = ((p - p.mean()) ** 2).sum()
    return float(1 - within / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xttn_dir", default="outputs/final_test/xttn")
    ap.add_argument("--imageonly_dir", default="outputs/final_test/imageonly")
    ap.add_argument("--xttn_retrained_dir", default="outputs/final_test/xttn_retrained")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {"xttn_published": args.xttn_dir, "imageonly": args.imageonly_dir, "xttn_retrained": args.xttn_retrained_dir}
    out = {}
    for name, d in runs.items():
        df = pd.read_csv(os.path.join(d, "frame_predictions.csv"))
        out[name] = {"n_frames": int(len(df)), "n_patients": int(df["patient"].nunique()),
                     "patient_variance_share": round(eta_squared(df["p_lesion"], df["patient"]), 4)}
    with open(os.path.join(args.out_dir, "patient_variance.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
