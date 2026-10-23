#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats


MODELS = {
    "YOLOv5l (image-only)": ("image_only_models_bal/yolov5l-cls_imageonly", (67.71, 96.53, 68.02, 79.65)),
    "VGG19 (image-only)":   ("image_only_models_bal/vgg19_imageonly",       (64.23, 77.55, 71.06, 74.04)),
    "YOLOv5l (late)":       ("yolo_earlylate_bal/yolov5l-cls_late",         (64.49, 77.23, 71.28, 72.68)),
    "YOLOv5l-Xttn":         ("yolo_mid_bal/yolov5l_cls_xattn_mid_early",    (70.51, 86.97, 72.44, 78.92)),
}
CHECK_COLS = ["val_frame_acc", "val_frame_recall", "val_frame_precision", "val_frame_f1"]
PRIMARY = ("Frame AUROC", "val_frame_auc")
SECONDARY = [
    ("Frame balanced accuracy", "val_frame_bal_acc"),
    ("Frame F1", "val_frame_f1"),
    ("Video AUROC", "val_video_auc"),
    ("Video top-k3 accuracy", "val_video_topk3_acc"),
]
COMPARISONS = [
    ("H1", "YOLOv5l-Xttn", "YOLOv5l (image-only)", "metadata fusion"),
    ("H2", "YOLOv5l (image-only)", "VGG19 (image-only)", "YOLO versus CNN backbones"),
    ("H3", "YOLOv5l-Xttn", "YOLOv5l (late)", "fusion strategy"),
]


def test_train_ratio(folds_root, k=5):
    ratios = []
    for i in range(k):
        n_tr = len(pd.read_csv(os.path.join(folds_root, f"fold_{i}", "train.csv")))
        n_va = len(pd.read_csv(os.path.join(folds_root, f"fold_{i}", "val.csv")))
        ratios.append(n_va / n_tr)
    return float(np.mean(ratios))


def corrected_ci(values, ratio):
    # Nadeau-Bengio corrected 95% CI of the mean over folds
    v = np.asarray(values, dtype=float)
    k = len(v)
    se = np.sqrt((1 / k + ratio) * v.var(ddof=1))
    half = stats.t.ppf(0.975, k - 1) * se
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def corrected_paired_test(a, b, ratio):
    # Nadeau-Bengio corrected resampled t-test on the fold-wise differences a - b
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    se = np.sqrt((1 / k + ratio) * d.var(ddof=1))
    t_stat = d.mean() / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t_stat), k - 1)
    half = stats.t.ppf(0.975, k - 1) * se
    return {"mean_diff": float(d.mean()), "ci_low": float(d.mean() - half),
            "ci_high": float(d.mean() + half), "t": float(t_stat), "p": float(p),
            "fold_diffs": [round(x, 4) for x in d.tolist()]}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def load_folds(path):
    df = pd.read_csv(os.path.join(path, "cv_val_folds.csv")).sort_values("fold").reset_index(drop=True)
    assert list(df["fold"]) == [0, 1, 2, 3, 4], f"unexpected folds in {path}"
    return df


def pct(x):
    return f"{100 * x:6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="outputs/checkpoints/new")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ratio = test_train_ratio(args.folds_root)
    k = 5
    print(f"Nadeau-Bengio: n_test/n_train = {ratio:.3f}; variance factor {1 / k + ratio:.3f} vs naive {1 / k:.3f}\n")

    data = {}
    print("Reproduction check (frame-level means, %)")
    for name, (rel, expected) in MODELS.items():
        df = load_folds(os.path.join(args.runs_root, rel))
        got = [100 * df[c].mean() for c in CHECK_COLS]
        ok = all(abs(g - e) < 0.01 for g, e in zip(got, expected))
        print(f"  {name:22s} {'OK ' if ok else 'MISMATCH'}  " + "  ".join(f"{g:.2f}" for g in got))
        if not ok:
            raise SystemExit(f"{name} does not reproduce the reported values; stopping.")
        data[name] = df

    results = {"n_test_over_n_train": ratio, "models": {}, "comparisons": {}}
    print("\nMean [Nadeau-Bengio 95% CI], %")
    for label, col in [PRIMARY] + SECONDARY:
        print(f"  {label}")
        for name, df in data.items():
            m, lo, hi = corrected_ci(df[col], ratio)
            results["models"].setdefault(name, {})[label] = {"mean": m, "ci_low": lo, "ci_high": hi}
            print(f"    {name:22s} {pct(m)} [{pct(lo)}, {pct(hi)}]")

    print(f"\nPrimary comparisons on {PRIMARY[0]} (Holm-corrected)")
    prim = [corrected_paired_test(data[a][PRIMARY[1]], data[b][PRIMARY[1]], ratio) for _, a, b, _ in COMPARISONS]
    adj = holm([r["p"] for r in prim])
    for (hid, a, b, claim), r, p_adj in zip(COMPARISONS, prim, adj):
        r["p_holm"] = p_adj
        results["comparisons"][hid] = {"A": a, "B": b, "claim": claim, "primary": r, "secondary": {}}
        print(f"  {hid} {a} vs {b} ({claim})")
        print(f"     diff {pct(r['mean_diff'])} [{pct(r['ci_low'])}, {pct(r['ci_high'])}]  "
              f"t={r['t']:.2f}  p={r['p']:.3f}  p_Holm={p_adj:.3f}")
        for label, col in SECONDARY:
            s = corrected_paired_test(data[a][col], data[b][col], ratio)
            results["comparisons"][hid]["secondary"][label] = s
            print(f"     (exploratory) {label:24s} diff {pct(s['mean_diff'])} "
                  f"[{pct(s['ci_low'])}, {pct(s['ci_high'])}]  p={s['p']:.3f}")

    with open(os.path.join(args.out_dir, "paired_cv_tests.json"), "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_root, "*", "*", "cv_val_folds.csv"))):
        df = pd.read_csv(path)
        if len(df) != 5:
            continue
        row = {"run": os.path.relpath(os.path.dirname(path), args.runs_root).replace("\\", "/")}
        for label, col in [PRIMARY] + SECONDARY:
            if col in df:
                m, lo, hi = corrected_ci(df[col], ratio)
                row[f"{label} mean"], row[f"{label} ci_low"], row[f"{label} ci_high"] = m, lo, hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "cv_ci_all_models.csv"), index=False)
    print(f"\nSaved paired_cv_tests.json and cv_ci_all_models.csv ({len(rows)} models) in {args.out_dir}")


if __name__ == "__main__":
    main()
