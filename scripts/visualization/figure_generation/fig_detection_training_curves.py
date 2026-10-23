# fig_detection_training_curves.py
import os
import glob
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

SPLITS = [("random_strat", "Random Split"), ("video_strat", "Video-level Split"), ("patient_strat", "Patient-level Split")]
TRAIN_COL, VAL_COL = "train/box_loss", "val/box_loss"
COLORS = {"train": "#1F77B4", "val": "#FF7F0E"}


def load_split(logs, split, model):
    files = sorted(glob.glob(os.path.join(logs, split, model, "runs", "fold_*", "results.csv")))
    if len(files) != 5:
        raise SystemExit(f"{split}: expected 5 fold logs, found {len(files)}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    n = min(len(d) for d in dfs)
    epochs = dfs[0]["epoch"].to_numpy()[:n]
    train = np.stack([d[TRAIN_COL].to_numpy(dtype=float)[:n] for d in dfs])
    val = np.stack([d[VAL_COL].to_numpy(dtype=float)[:n] for d in dfs])
    train[~np.isfinite(train)] = np.nan
    val[~np.isfinite(val)] = np.nan
    return epochs, train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="outputs/detection_cv_results")
    ap.add_argument("--model", default="yolo26x")
    ap.add_argument("--out_dir", default=os.path.join(script_dir, "output_figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                         "font.family": "DejaVu Sans", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (split, title) in zip(axes, SPLITS):
        epochs, train, val = load_split(args.logs, split, args.model)
        # mean over the five folds (line) and range across folds (shaded); non-finite epochs are skipped
        for arr, key, label in ((train, "train", "Training"), (val, "val", "Validation")):
            ax.fill_between(epochs, np.nanmin(arr, axis=0), np.nanmax(arr, axis=0),
                            color=COLORS[key], alpha=0.18, linewidth=0)
            ax.plot(epochs, np.nanmean(arr, axis=0), color=COLORS[key], linewidth=2.0, label=label)
        ax.set_title(title, fontfamily="serif", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, linewidth=0.5, alpha=0.4)
    axes[0].set_ylabel("Box loss")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"detection_training_curves.{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
