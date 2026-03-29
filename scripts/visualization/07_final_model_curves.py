#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_final_model_curves.py

Run inference with the final YOLOv5-L cross-attention classifier on the
pristine test set and save ROC, PR, and Sensitivity-Specificity curves
at both frame level and video level.

Color palette and plot style match 02_train_kfold.py evaluation plots.

Usage (from repo root):
    python scripts/visualization/07_final_model_curves.py
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.dataio.dataset import load_metadata_any, build_meta_vectors, build_text_embed_map
from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionCLSCrossAttn
from src.trainloop.metrics import metrics_from_probs

from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette — identical to 02_train_kfold.py evaluation block
# ---------------------------------------------------------------------------
PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A", "#CAB2D6", "#8DD3C7"]
#           [0]pale    [1]blue    [2]green   [3]orange  [4]pink    [5]purple  [6]teal

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CKPT    = "outputs/checkpoints/final/yolov5l_cls_xattn_mid_early/final_full_train.pt"
DEFAULT_TEST    = "outputs/index/test_pristine.csv"
DEFAULT_OUTDIR  = "outputs/checkpoints/final/yolov5l_cls_xattn_mid_early/final_full/figs"
DEFAULT_YOLOv5  = "yolov5l-cls.pt"
THR             = 0.5

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt",     default=DEFAULT_CKPT)
ap.add_argument("--test-csv", default=DEFAULT_TEST)
ap.add_argument("--outdir",   default=DEFAULT_OUTDIR)
ap.add_argument("--thr",      type=float, default=THR)
args = ap.parse_args()

ckpt_path = os.path.join(REPO_ROOT, args.ckpt)
test_csv  = os.path.join(REPO_ROOT, args.test_csv)
figs_dir  = os.path.join(REPO_ROOT, args.outdir)
os.makedirs(figs_dir, exist_ok=True)
thr = args.thr

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {ckpt_path}")
ck        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
tab_dim   = int(ck["tab_dim"])
cfg       = ck["cfg"]
backbone  = ck["backbone"]
print(f"  backbone={backbone}  tab_dim={tab_dim}")

# ---------------------------------------------------------------------------
# Ensure yolov5l-cls.pt is present
# ---------------------------------------------------------------------------
yolov5_weights = os.path.join(REPO_ROOT, DEFAULT_YOLOv5)
if not os.path.isfile(yolov5_weights):
    url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt"
    print(f"Downloading {DEFAULT_YOLOv5} ...")
    if hasattr(torch, "hub") and hasattr(torch.hub, "download_url_to_file"):
        torch.hub.download_url_to_file(url, yolov5_weights, progress=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, yolov5_weights)

# ---------------------------------------------------------------------------
# Build metadata vectors
# ---------------------------------------------------------------------------
print("Building metadata vectors ...")
meta_df  = load_metadata_any(os.path.join(REPO_ROOT, cfg.get("metadata_path", "data/CADICA/metadata.xlsx")))
sel      = cfg.get("metadata_features", {})
sel_nc   = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
text_cols = sel.get("text") or []

text_map = None
if text_cols:
    text_map = build_text_embed_map(
        meta_df,
        patient_col="patient",
        text_cols=text_cols,
        model_name=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_dir=os.path.join(REPO_ROOT, cfg.get("text_cache_dir", "outputs/text_embeds")),
    )

meta_map = build_meta_vectors(meta_df, selected_features=sel_nc, text_embed_map=text_map)
assert len(next(iter(meta_map.values()))) == tab_dim
print(f"  dim={tab_dim}  patients={len(meta_map)}")

# ---------------------------------------------------------------------------
# Instantiate model
# ---------------------------------------------------------------------------
print("Instantiating model ...")
device     = "cuda" if torch.cuda.is_available() else "cpu"
model_file = backbone.split("_xattn")[0].replace("_cls", "-cls")
os.chdir(REPO_ROOT)

model = YOLOv5MidFusionCLSCrossAttn(
    model_name=model_file, tab_dim=tab_dim, pretrained=False, out_dim=2
).to(device)
model.load_state_dict(ck["model"])
model.eval()
print(f"  Model on {device}.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
print("Running inference on test set ...")
test_df = pd.read_csv(test_csv)

y_true_all, y_prob_all, y_path_all = [], [], []

with torch.no_grad():
    for _, row in test_df.iterrows():
        img_path = os.path.join(REPO_ROOT, row["img_path"])
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        img_in  = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_AREA)
        img_rgb = np.stack([img_in, img_in, img_in], axis=-1).astype(np.float32) / 255.0  # HxWx3
        mean    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std                         # HxWx3
        x = torch.tensor(img_norm.transpose(2, 0, 1)[None]).to(device)  # 1x3x224x224

        pid    = int(row["patient"])
        clin_np = np.asarray(meta_map.get(pid, np.zeros(tab_dim)), dtype=np.float32)
        clin   = torch.tensor(clin_np)[None].to(device)

        _, logits = model(x, clin)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

        y_true_all.append(int(row["label"]))
        y_prob_all.append(prob)
        y_path_all.append(row["img_path"])

y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all)
y_path_all = np.array(y_path_all)
print(f"  Inference done. {len(y_true_all)} frames.")

# Video-level: max-pool probabilities per video
video_df = pd.DataFrame({
    "img_path": y_path_all,
    "y_true":   y_true_all,
    "y_prob":   y_prob_all,
})
video_df = video_df.merge(
    test_df[["img_path", "patient", "video"]], on="img_path", how="left"
)
vid_agg  = video_df.groupby(["patient", "video"]).agg(
    y_prob=("y_prob", "max"),
    y_true=("y_true", "max"),
).reset_index()
y_true_vid = vid_agg["y_true"].values
y_prob_vid = vid_agg["y_prob"].values
print(f"  Video-level: {len(y_true_vid)} videos.")

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_prob, thr, domain_label):
    from sklearn.metrics import confusion_matrix
    yhat = (y_prob >= thr).astype(int)
    cm   = confusion_matrix(y_true, yhat, labels=[0, 1])
    plt.figure(figsize=(4.5, 4.0), dpi=150)
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="#111", fontsize=12)
    plt.title(f"Confusion Matrix ({domain_label}, thr={thr:.2f})")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(os.path.join(figs_dir, f"confusion_{domain_label}.{ext}"))
    plt.close()
    print(f"  [saved] confusion_{domain_label}")


def save_curves(y_true, y_prob, thr, domain_label):
    md = metrics_from_probs(y_true, y_prob, thr=thr)

    # ---- ROC ----
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = float(md.get("auc", float("nan")))
        plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(fpr, tpr, color=PALETTE[1], lw=2, label=f"ROC (AUC={auc_val:.3f})")
        plt.scatter([md.get("fpr", 0.0)], [md.get("sensitivity", 0.0)],
                    color=PALETTE[3], s=24, zorder=5, label=f"thr={thr:.2f}")
        plt.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
        plt.xlabel("1 - Specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title(f"ROC ({domain_label})")
        plt.grid(alpha=0.25)
        plt.legend(loc="lower right")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(figs_dir, f"roc_{domain_label}.{ext}"))
        plt.close()
        print(f"  [saved] roc_{domain_label}")
    except Exception as e:
        print(f"  [warn] ROC ({domain_label}): {e}")

    # ---- PR ----
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap_val = float(md.get("ap", float("nan")))
        plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(rec, prec, color=PALETTE[3], lw=2, label=f"PR (AP={ap_val:.3f})")
        plt.scatter([md.get("sensitivity", 0.0)], [md.get("ppv", 0.0)],
                    color=PALETTE[1], s=24, zorder=5, label=f"thr={thr:.2f}")
        plt.xlabel("Recall (Sensitivity)")
        plt.ylabel("Precision (PPV)")
        plt.title(f"PR ({domain_label})")
        plt.grid(alpha=0.25)
        plt.legend(loc="lower left")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(figs_dir, f"pr_{domain_label}.{ext}"))
        plt.close()
        print(f"  [saved] pr_{domain_label}")
    except Exception as e:
        print(f"  [warn] PR ({domain_label}): {e}")

    # ---- Sensitivity-Specificity ----
    try:
        uniq = np.unique(y_prob)
        cand = np.unique(np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
        sens_list, spec_list = [], []
        for t in cand:
            m = metrics_from_probs(y_true, y_prob, thr=float(t))
            sens_list.append(float(m.get("sensitivity", 0.0)))
            spec_list.append(float(m.get("specificity", 0.0)))
        plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(spec_list, sens_list, color=PALETTE[2], lw=2)
        plt.scatter([md.get("specificity", 0.0)], [md.get("sensitivity", 0.0)],
                    color=PALETTE[4], s=24, zorder=5, label=f"thr={thr:.2f}")
        plt.xlabel("Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"Sensitivity vs Specificity ({domain_label})")
        plt.grid(alpha=0.25)
        plt.legend(loc="lower left")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(figs_dir, f"sens_spec_{domain_label}.{ext}"))
        plt.close()
        print(f"  [saved] sens_spec_{domain_label}")
    except Exception as e:
        print(f"  [warn] Sens-Spec ({domain_label}): {e}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print(f"\nGenerating curves (thr={thr}) ...")
save_curves(y_true_all, y_prob_all, thr, "frame")
save_curves(y_true_vid, y_prob_vid, thr, "video")
save_confusion_matrix(y_true_all, y_prob_all, thr, "frame")
save_confusion_matrix(y_true_vid, y_prob_vid, thr, "video")

print(f"\nAll curves saved to: {figs_dir}")

# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
import json

def build_metrics_table(y_true, y_prob, thr, label):
    m = metrics_from_probs(y_true, y_prob, thr=thr)
    return {
        "level":               label,
        "n_samples":           int(len(y_true)),
        "n_positives":         int(y_true.sum()),
        "n_negatives":         int((1 - y_true).sum()),
        # ---- accuracy ----
        "accuracy_overall":    round(float(m["acc"]),         4),
        "accuracy_positives":  round(float(m["sensitivity"]), 4),   # TP / (TP+FN)
        "accuracy_negatives":  round(float(m["specificity"]), 4),   # TN / (TN+FP)
        # ---- sensitivity / specificity ----
        "sensitivity":         round(float(m["sensitivity"]), 4),
        "sensitivity_positives": round(float(m["sensitivity"]), 4), # same: correct rate on GT-positive
        "sensitivity_negatives": round(float(m["specificity"]), 4), # correct rate on GT-negative
        # ---- other ----
        "specificity":         round(float(m["specificity"]), 4),
        "balanced_accuracy":   round(float(m["bal_acc"]),     4),
        "precision_ppv":       round(float(m["ppv"]),         4),
        "npv":                 round(float(m["npv"]),         4),
        "f1":                  round(float(m["f1"]),          4),
        "auc":                 round(float(m["auc"]),         4),
        "ap":                  round(float(m["ap"]),          4),
        "kappa":               round(float(m["kappa"]),       4),
        "threshold":           thr,
    }

rows = [
    build_metrics_table(y_true_all, y_prob_all, thr, "frame"),
    build_metrics_table(y_true_vid, y_prob_vid, thr, "video"),
]

# Print table
COL_W = 26
print(f"\n{'─'*65}")
print(f"  Test-set evaluation  (threshold={thr})")
print(f"{'─'*65}")
header_metrics = [
    ("accuracy_overall",     "Accuracy (overall)"),
    ("accuracy_positives",   "Accuracy — positives"),
    ("accuracy_negatives",   "Accuracy — negatives"),
    ("sensitivity",          "Sensitivity"),
    ("sensitivity_positives","Sensitivity — positives"),
    ("sensitivity_negatives","Sensitivity — negatives"),
    ("specificity",          "Specificity"),
    ("balanced_accuracy",    "Balanced accuracy"),
    ("precision_ppv",        "Precision / PPV"),
    ("npv",                  "NPV"),
    ("f1",                   "F1"),
    ("auc",                  "AUC"),
    ("ap",                   "AP"),
    ("kappa",                "Kappa"),
]
print(f"{'Metric':<{COL_W}}  {'Frame':>8}  {'Video':>8}")
print(f"{'─'*COL_W}  {'─'*8}  {'─'*8}")
for key, label in header_metrics:
    frame_val = rows[0][key]
    video_val = rows[1][key]
    print(f"{label:<{COL_W}}  {frame_val:>8.4f}  {video_val:>8.4f}")
print(f"{'─'*65}")
print(f"  n_samples   frame={rows[0]['n_samples']}  video={rows[1]['n_samples']}")
print(f"  n_positives frame={rows[0]['n_positives']}  video={rows[1]['n_positives']}")
print(f"  n_negatives frame={rows[0]['n_negatives']}  video={rows[1]['n_negatives']}")
print(f"{'─'*65}\n")

# Save JSON
metrics_path = os.path.join(figs_dir, "test_metrics_full.json")
with open(metrics_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Metrics saved to: {metrics_path}")

# ---------------------------------------------------------------------------
# Save true-positive filenames
# ---------------------------------------------------------------------------
y_pred_all = (y_prob_all >= thr).astype(int)
tp_mask    = (y_true_all == 1) & (y_pred_all == 1)
tp_paths   = y_path_all[tp_mask]

tp_txt_path = os.path.join(figs_dir, "true_positives.txt")
with open(tp_txt_path, "w") as f:
    for p in tp_paths:
        f.write(p + "\n")
print(f"True positives ({len(tp_paths)}/{int(y_true_all.sum())} lesion frames): {tp_txt_path}")
