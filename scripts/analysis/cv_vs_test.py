#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd


METRICS = [
    ("AUROC", "auc", "auc"),
    ("Balanced accuracy", "bal_acc", "bal_acc"),
    ("Specificity", "specificity", "specificity"),
    ("Sensitivity", "recall", "recall"),
    ("Precision", "precision", "precision"),
    ("F1", "f1", "f1"),
]


def fold_summary(folds, level):
    out = {}
    for label, col, _ in METRICS:
        v = folds[f"val_{level}_{col}"].astype(float)
        out[label] = {"mean": v.mean(), "min": v.min(), "max": v.max(),
                      "per_fold": [round(x, 4) for x in v.tolist()]}
    return out


def prevalence(index_csv):
    lab = pd.read_csv(index_csv)["label"]
    p = float(lab.mean())
    return {"frames": int(len(lab)), "positive": int(lab.sum()), "prevalence": p,
            "always_positive_f1": 2 * p / (1 + p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xttn_folds", default="outputs/checkpoints/new/yolo_mid_bal/yolov5l_cls_xattn_mid_early/cv_val_folds.csv")
    ap.add_argument("--imageonly_folds", default="outputs/checkpoints/new/image_only_models_bal/yolov5l-cls_imageonly/cv_val_folds.csv")
    ap.add_argument("--xttn_test", default="outputs/checkpoints/final/yolov5l_cls_xattn_mid_early/final_test_metrics.json")
    ap.add_argument("--pool_csv", default="outputs/index/trainval_pool.csv")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    xttn = pd.read_csv(args.xttn_folds).sort_values("fold")
    img = pd.read_csv(args.imageonly_folds).sort_values("fold")
    with open(args.xttn_test) as f:
        test = json.load(f)

    results = {"cv_pool": prevalence(args.pool_csv), "test_set": prevalence(args.test_csv)}
    for level in ("frame", "video"):
        cv_x, cv_i = fold_summary(xttn, level), fold_summary(img, level)
        results[level] = {"xttn_cv": cv_x, "imageonly_cv": cv_i,
                          "xttn_test": {lab: test[level][key] for lab, _, key in METRICS}}
        print(f"\n{level.upper()} LEVEL (%)")
        print(f"{'Metric':18s} {'Xttn CV mean [fold min-max]':>30s} {'Xttn test':>10s} {'Image-only CV mean [min-max]':>30s}")
        for label, _, key in METRICS:
            a, i = cv_x[label], cv_i[label]
            print(f"{label:18s} {100*a['mean']:8.2f} [{100*a['min']:6.2f}-{100*a['max']:6.2f}]"
                  f"{'':6s} {100*test[level][key]:10.2f} "
                  f"{100*i['mean']:12.2f} [{100*i['min']:6.2f}-{100*i['max']:6.2f}]")

    print("\nPositive prevalence (frames) and always-positive F1")
    for name in ("cv_pool", "test_set"):
        r = results[name]
        print(f"  {name:9s} {r['positive']:5d}/{r['frames']:5d} = {100*r['prevalence']:5.2f}% "
              f"-> always-positive F1 {100*r['always_positive_f1']:5.2f}%")

    out_path = os.path.join(args.out_dir, "cv_vs_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
