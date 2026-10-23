#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paired_cv_tests import corrected_paired_test, test_train_ratio


SEEDS = [1337, 1, 2]
MODELS = {"image-only YOLOv5l": ("yolov5l-cls_imageonly", "image_only_models_bal/yolov5l-cls_imageonly"),
          "YOLOv5l-Xttn": ("yolov5l_cls_xattn_mid_early", "yolo_mid_bal/yolov5l_cls_xattn_mid_early")}
METRICS = [("Frame AUROC (primary)", "val_frame_auc"), ("Frame balanced accuracy", "val_frame_bal_acc"),
           ("Frame specificity", "val_frame_specificity"), ("Frame sensitivity", "val_frame_recall"),
           ("Frame F1", "val_frame_f1"), ("Video AUROC", "val_video_auc"),
           ("Video top-k3 accuracy", "val_video_topk3_acc")]


def load(path):
    df = pd.read_csv(os.path.join(path, "cv_val_folds.csv")).sort_values("fold").reset_index(drop=True)
    assert list(df["fold"]) == [0, 1, 2, 3, 4], f"unexpected folds in {path}"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_dir", default="outputs/seed_repeats")
    ap.add_argument("--runs_root", default="outputs/checkpoints/new")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ratio = test_train_ratio(args.folds_root)
    runs = {s: {m: load(os.path.join(args.seeds_dir, f"seed_{s}", d)) for m, (d, _) in MODELS.items()} for s in SEEDS}
    pub = {m: load(os.path.join(args.runs_root, p)) for m, (_, p) in MODELS.items()}
    out = {"n_test_over_n_train": ratio, "per_seed": {}, "seed_variance": {}, "paired": {}, "rerun_vs_published": {}}

    print("Mean over folds per seed (%), seed SD vs mean fold SD")
    for label, col in METRICS:
        print(f"  {label}")
        for m in MODELS:
            means = [100 * runs[s][m][col].mean() for s in SEEDS]
            fold_sd = np.mean([100 * runs[s][m][col].std(ddof=1) for s in SEEDS])
            out["per_seed"].setdefault(label, {})[m] = dict(zip(map(str, SEEDS), means))
            out["seed_variance"].setdefault(label, {})[m] = {"mean_over_seeds": float(np.mean(means)),
                                                              "sd_across_seeds": float(np.std(means, ddof=1)),
                                                              "mean_sd_across_folds": float(fold_sd)}
            print(f"    {m:20s} " + "  ".join(f"s{s}: {v:6.2f}" for s, v in zip(SEEDS, means))
                  + f" | seed SD {np.std(means, ddof=1):5.2f}  fold SD {fold_sd:5.2f}")

    print("\nPaired Xttn - image-only, Nadeau-Bengio corrected (per seed, and seed-averaged folds)")
    for label, col in METRICS:
        print(f"  {label}")
        res = {}
        for s in SEEDS:
            res[str(s)] = corrected_paired_test(runs[s]["YOLOv5l-Xttn"][col], runs[s]["image-only YOLOv5l"][col], ratio)
        avg = {m: np.mean([runs[s][m][col].to_numpy() for s in SEEDS], axis=0) for m in MODELS}
        res["seed_average"] = corrected_paired_test(avg["YOLOv5l-Xttn"], avg["image-only YOLOv5l"], ratio)
        out["paired"][label] = res
        for k, r in res.items():
            print(f"    {k:12s} diff {100 * r['mean_diff']:+6.2f} [{100 * r['ci_low']:+6.2f}, {100 * r['ci_high']:+6.2f}]  p={r['p']:.3f}")

    print("\nSeed-1337 rerun vs published run (fold values, %)")
    for m in MODELS:
        for label, col in METRICS[:3] + METRICS[6:]:
            a, b = 100 * runs[1337][m][col].to_numpy(), 100 * pub[m][col].to_numpy()
            out["rerun_vs_published"].setdefault(m, {})[label] = {
                "rerun": a.round(2).tolist(), "published": b.round(2).tolist(),
                "mean_rerun": float(a.mean()), "mean_published": float(b.mean()),
                "max_abs_fold_diff": float(np.abs(a - b).max())}
            print(f"  {m:20s} {label:24s} rerun {a.mean():6.2f} published {b.mean():6.2f}  max |fold diff| {np.abs(a - b).max():5.2f}")
        be = [runs[1337][m]["best_epoch"].tolist(), pub[m]["best_epoch"].tolist()]
        out["rerun_vs_published"][m]["best_epoch"] = {"rerun": be[0], "published": be[1]}

    with open(os.path.join(args.out_dir, "seed_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved seed_analysis.json in {args.out_dir}")


if __name__ == "__main__":
    main()
