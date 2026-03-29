#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train_final_det.py

Train final YOLOv26x detection models on ALL available training data
(train + inner_val + outer_val from fold_1, which spans the entire non-test
dataset) for three split strategies:
  - random           → test: data/det_random_folds/test
  - video-stratified → test: data/det_video_stratified_folds/test
  - patient-stratified → test: data/cadica_yolo_det/test

No validation set is used during training (val=False).
After training, each model is evaluated on its respective held-out test set.

Usage:
    python scripts/03_train_final_det.py [options]

Options:
    --model       YOLO model name (default: yolo26x)
    --epochs      Number of epochs (default: 100)
    --imgsz       Image size (default: 640)
    --batch       Batch size (default: 16)
    --output-dir  Output directory (default: outputs/final_det_models)
    --skip-train  Skip training, only run evaluation on existing weights
    --splits      Comma-separated list of splits to run (default: random,video,patient)
"""

import os
import json
import time
import argparse

import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Palette  (one colour per split, in declaration order)
# ---------------------------------------------------------------------------
SPLIT_COLORS = {
    "random":  "#B2DF8A",
    "video":   "#2078B3",
    "patient": "#F99B9B",
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train final YOLO detection models on full (non-test) data"
)
parser.add_argument("--model", type=str, default="yolo26x",
                    help="YOLO model name (default: yolo26x)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--output-dir", type=str, default="outputs/final_det_models")
parser.add_argument("--skip-train", action="store_true",
                    help="Skip training; evaluate existing weights only")
parser.add_argument(
    "--splits",
    type=str,
    default="random,video,patient",
    help="Comma-separated list of splits to run: random,video,patient",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve relative to the repo root (one level above scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.abspath(os.path.join(REPO_ROOT, args.output_dir))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# fmt: off
SPLITS_CONFIG = {
    "random": {
        "folds_dir":  os.path.join(REPO_ROOT, "data", "det_random_folds"),
        "test_images": os.path.join(REPO_ROOT, "data", "det_random_folds", "test", "images"),
    },
    "video": {
        "folds_dir":  os.path.join(REPO_ROOT, "data", "det_video_stratified_folds"),
        "test_images": os.path.join(REPO_ROOT, "data", "det_video_stratified_folds", "test", "images"),
    },
    "patient": {
        "folds_dir":  os.path.join(REPO_ROOT, "data", "det_patient_stratified_folds"),
        "test_images": os.path.join(REPO_ROOT, "data", "cadica_yolo_det", "test", "images"),
    },
}
# fmt: on

REQUESTED_SPLITS = [s.strip() for s in args.splits.split(",")]
CLASS_NAMES = {0: "lesion"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_train_image_dirs(folds_dir: str) -> list[str]:
    """
    Return the three image directories from fold_1 that together cover
    the entire non-test dataset: train + inner_val + outer_val.

    In a nested 5-fold setup:
        fold_1/train + fold_1/inner_val + fold_1/outer_val == all non-test images
    because outer_val_1 is 1/5 of non-test data and (train_1 + inner_val_1)
    is the remaining 4/5.
    """
    fold1 = os.path.join(folds_dir, "fold_1")
    dirs = [
        os.path.join(fold1, "train",     "images"),
        os.path.join(fold1, "inner_val", "images"),
        os.path.join(fold1, "outer_val", "images"),
    ]
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected directory not found: {d}")
    return dirs


def count_images(dirs: list[str]) -> int:
    total = 0
    for d in dirs:
        total += len([
            f for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
    return total


def write_train_yaml(train_dirs: list[str], yaml_path: str) -> str:
    """
    Write a YOLO data YAML for training on multiple image directories.
    val is set to the first train dir as a placeholder; it is never used
    because training runs with val=False.
    """
    data = {
        "train": train_dirs,
        "val":   train_dirs[0],  # placeholder — overridden by val=False
        "names": CLASS_NAMES,
        "nc":    len(CLASS_NAMES),
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


def write_test_yaml(test_images_dir: str, yaml_path: str) -> str:
    """Write a YOLO data YAML for evaluation on a single test images directory."""
    abs_dir = os.path.abspath(test_images_dir)
    data = {
        "train": abs_dir,   # required by Ultralytics even when only validating
        "val":   abs_dir,
        "names": CLASS_NAMES,
        "nc":    len(CLASS_NAMES),
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


def save_combined_curves(split_csvs: dict[str, str], out_path: str) -> None:
    """
    Plot training loss curves for all splits on a single figure.

    split_csvs: {split_name: path_to_results.csv}
    One subplot per loss type (box_loss, cls_loss, dfl_loss); each subplot
    contains one line per split, coloured by SPLIT_COLORS.
    """
    # Load all available CSVs
    data: dict[str, pd.DataFrame] = {}
    for split, csv_path in split_csvs.items():
        if not os.path.exists(csv_path):
            print(f"[Curves] results.csv not found for '{split}' ({csv_path}), skipping.")
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        data[split] = df

    if not data:
        print("[Curves] No results.csv files found — skipping combined curve plot.")
        return

    # Determine loss columns from the first available CSV
    sample_df = next(iter(data.values()))
    loss_cols = [c for c in sample_df.columns if c.startswith("train/") and "loss" in c]

    if not loss_cols:
        print("[Curves] No train/*_loss columns found — skipping combined curve plot.")
        return

    n_panels = len(loss_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, col in zip(axes, loss_cols):
        for split, df in data.items():
            if col not in df.columns:
                continue
            epoch = df["epoch"] if "epoch" in df.columns else range(len(df))
            color = SPLIT_COLORS.get(split, "#888888")
            label = split
            ax.plot(epoch, df[col], color=color, linewidth=1.8, label=label)

        title = col.replace("train/", "").replace("_", " ")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Training loss curves — random / video / patient", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Curves] Saved → {out_path}")


def find_weights(split_output_dir: str) -> str | None:
    """
    Look for model weights. Since we train with val=False, prefer last.pt
    (best.pt selection requires validation metrics). Falls back to best.pt
    if last.pt is absent.
    """
    candidates = [
        os.path.join(split_output_dir, "runs", "train", "weights", "last.pt"),
        os.path.join(split_output_dir, "runs", "train", "weights", "best.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Final Detection Model Training — {args.model}")
print(f"{'='*65}")
print(f"  Mode:       {'EVALUATION ONLY' if args.skip_train else 'TRAIN + EVAL'}")
print(f"  Splits:     {REQUESTED_SPLITS}")
print(f"  Epochs:     {args.epochs}")
print(f"  Img size:   {args.imgsz}")
print(f"  Batch:      {args.batch}")
print(f"  Output dir: {OUTPUT_DIR}")
print(f"{'='*65}\n")

all_results = []
split_results_csvs: dict[str, str] = {}  # collected for combined curve plot

for split_name in REQUESTED_SPLITS:
    if split_name not in SPLITS_CONFIG:
        print(f"WARNING: Unknown split '{split_name}', skipping.")
        continue

    cfg = SPLITS_CONFIG[split_name]
    folds_dir   = cfg["folds_dir"]
    test_images = cfg["test_images"]

    print(f"\n{'='*65}")
    print(f"  SPLIT: {split_name.upper()}")
    print(f"{'='*65}")

    split_out = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_out, exist_ok=True)

    train_time_hours = None

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    existing_weights = find_weights(split_out)
    skip_this = args.skip_train or (existing_weights is not None)

    if skip_this and existing_weights is not None:
        print(f"[Train] Weights already exist, skipping training for '{split_name}'.")
    elif skip_this:
        print(f"WARNING: --skip-train set but no weights found for '{split_name}'.")

    if not skip_this:
        train_dirs = all_train_image_dirs(folds_dir)
        n_imgs = count_images(train_dirs)

        print(f"\n[Train] {n_imgs} images across:")
        for d in train_dirs:
            print(f"    {os.path.relpath(d, REPO_ROOT)}")

        train_yaml = os.path.join(split_out, "train_data.yaml")
        write_train_yaml(train_dirs, train_yaml)

        model = YOLO(f"{args.model}.pt")

        t0 = time.time()
        model.train(
            data=train_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer="auto",
            val=False,          # no in-training validation
            name="train",
            project=os.path.join(split_out, "runs"),
            exist_ok=True,
        )
        train_time_hours = (time.time() - t0) / 3600
        print(f"[Train] Finished in {train_time_hours:.2f} hours")

    # Record results.csv path for combined curve plot (whether just trained or pre-existing)
    split_results_csvs[split_name] = os.path.join(split_out, "runs", "train", "results.csv")

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    weights = find_weights(split_out)
    if weights is None:
        print(f"WARNING: No weights found under {split_out} — skipping evaluation.")
        continue

    print(f"\n[Eval]  Weights : {os.path.relpath(weights, REPO_ROOT)}")
    print(f"[Eval]  Test set: {os.path.relpath(test_images, REPO_ROOT)}")

    test_yaml = os.path.join(split_out, "test_data.yaml")
    write_test_yaml(test_images, test_yaml)

    eval_model = YOLO(weights)
    metrics = eval_model.val(
        data=test_yaml,
        project=os.path.join(split_out, "test_results"),
        name="eval",
        exist_ok=True,
        save=False,
    )

    precision = float(metrics.box.p)
    recall    = float(metrics.box.r)
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    result = {
        "split":            split_name,
        "model":            args.model,
        "weights":          os.path.relpath(weights, REPO_ROOT),
        "test_images":      os.path.relpath(test_images, REPO_ROOT),
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1_score":         round(f1, 4),
        "mAP50":            round(float(metrics.box.map50), 4),
        "mAP50-95":         round(float(metrics.box.map),   4),
        "train_time_hours": round(train_time_hours, 3) if train_time_hours else None,
    }
    all_results.append(result)

    # Per-split JSON
    with open(os.path.join(split_out, "test_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Results] {split_name}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  mAP50     : {result['mAP50']:.4f}")
    print(f"  mAP50-95  : {result['mAP50-95']:.4f}")

# ---------------------------------------------------------------------------
# Combined training curves
# ---------------------------------------------------------------------------
if split_results_csvs:
    curves_out = os.path.join(OUTPUT_DIR, "training_curves.png")
    save_combined_curves(split_results_csvs, curves_out)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if all_results:
    df = pd.DataFrame(all_results)
    summary_csv = os.path.join(OUTPUT_DIR, "final_model_comparison.csv")
    df.to_csv(summary_csv, index=False)

    print(f"\n{'='*65}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*65}")
    print(df[["split", "precision", "recall", "f1_score",
              "mAP50", "mAP50-95"]].to_string(index=False))
    print(f"\nFull results saved to: {summary_csv}")

print(f"\n{'='*65}")
print("  ALL DONE")
print(f"{'='*65}\n")
