#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd


BINS = 10


def calibration(y, p):
    # expected calibration error over 10 equal-width probability bins, and the reliability curve
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, BINS - 1)
    ece, curve = 0.0, []
    for b in range(BINS):
        m = idx == b
        if not m.any():
            continue
        conf, freq = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(freq - conf)
        curve.append({"bin": [round(edges[b], 1), round(edges[b + 1], 1)], "n": int(m.sum()),
                      "mean_predicted": round(conf, 4), "observed_fraction": round(freq, 4)})
    return 100 * float(ece), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xttn_dir", default="outputs/final_test/xttn")
    ap.add_argument("--imageonly_dir", default="outputs/final_test/imageonly")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out = {}
    for name, d in {"YOLOv5l-Xttn": args.xttn_dir, "image-only YOLOv5l": args.imageonly_dir}.items():
        f = pd.read_csv(os.path.join(d, "frame_predictions.csv"))
        y, p = f["y_true"].astype(int).to_numpy(), f["p_lesion"].to_numpy()
        ece, curve = calibration(y, p)
        out[name] = {"frames": int(len(y)), "prevalence": round(100 * float(y.mean()), 2),
                     "mean_predicted": round(100 * float(p.mean()), 2), "ece": round(ece, 2), "reliability": curve}
        print(f"{name:20s} ECE {ece:5.2f}%  mean predicted {100 * p.mean():5.2f}%  prevalence {100 * y.mean():5.2f}%")
    with open(os.path.join(args.out_dir, "calibration_ece.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
