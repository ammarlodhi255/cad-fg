#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_gradcam_heatmaps_bbox.py

Generate GradCAM heatmaps and ground-truth bounding box overlays for the
YOLOv5-L cross-attention mid-fusion classifier, restricted to:
  - test_pristine.csv rows with label == 1  (lesion frames only)
  - frames that have a matching YOLO label file in cadica_yolo_det/test/labels/

Output:
  classification_heatmaps_with_bbox/
  ├── heatmaps/   ← GradCAM blended onto original image
  └── bbox/       ← original image with GT bounding box drawn

Usage (from repo root):
    python scripts/visualization/06_gradcam_heatmaps_bbox.py [--ckpt ...] [--outdir ...]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd

# ---- repo root on path ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.visualization.gradcam   import GradCAM
from scripts.visualization.overlays  import overlay_heatmap
from src.dataio.dataset import (
    load_metadata_any,
    build_meta_vectors,
    build_text_embed_map,
)
from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionCLSCrossAttn

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CKPT   = "outputs/checkpoints/final/yolov5l_cls_xattn_mid_early/final_full_train.pt"
DEFAULT_OUTDIR = "classification_heatmaps_with_bbox"
DEFAULT_TEST   = "outputs/index/test_pristine.csv"
DEFAULT_LABELS = "data/CADICA/selectedVideos"  # groundtruth lives alongside images
DEFAULT_YOLOv5 = "yolov5l-cls.pt"    # downloaded automatically if absent

# Bounding-box drawing style
BBOX_COLOR     = (0, 255, 0)   # green, BGR
BBOX_THICKNESS = 2

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt",      default=DEFAULT_CKPT)
ap.add_argument("--outdir",    default=DEFAULT_OUTDIR)
ap.add_argument("--test-csv",  default=DEFAULT_TEST)
args = ap.parse_args()

ckpt_path   = os.path.join(REPO_ROOT, args.ckpt)
outdir      = os.path.join(REPO_ROOT, args.outdir)
heatmap_dir = os.path.join(outdir, "heatmaps")
bbox_dir    = os.path.join(outdir, "bbox")
sbs_dir     = os.path.join(outdir, "side_by_side")
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(bbox_dir,    exist_ok=True)
os.makedirs(sbs_dir,     exist_ok=True)

test_csv = os.path.join(REPO_ROOT, args.test_csv)

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {ckpt_path}")
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

backbone_name = ck["backbone"]          # "yolov5l_cls_xattn_mid"
tab_dim       = int(ck["tab_dim"])      # 406
cfg           = ck["cfg"]

print(f"  backbone : {backbone_name}")
print(f"  tab_dim  : {tab_dim}")

# ---------------------------------------------------------------------------
# Ensure yolov5l-cls.pt is present (needed to instantiate the skeleton)
# ---------------------------------------------------------------------------
yolov5_weights = os.path.join(REPO_ROOT, DEFAULT_YOLOv5)
if not os.path.isfile(yolov5_weights):
    url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt"
    print(f"Downloading {DEFAULT_YOLOv5} from {url} ...")
    if hasattr(torch, "hub") and hasattr(torch.hub, "download_url_to_file"):
        torch.hub.download_url_to_file(url, yolov5_weights, progress=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, yolov5_weights)
    print("  Done.")

# ---------------------------------------------------------------------------
# Build metadata vectors (exactly as in training)
# ---------------------------------------------------------------------------
print("Building metadata vectors ...")
meta_df = load_metadata_any(
    os.path.join(REPO_ROOT, cfg.get("metadata_path", "data/CADICA/metadata.xlsx"))
)

sel    = cfg.get("metadata_features", {})
sel_nc = {
    "numeric":     sel.get("numeric"),
    "categorical": sel.get("categorical"),
}
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

# sanity-check dimension
_sample_vec = next(iter(meta_map.values()))
assert len(_sample_vec) == tab_dim, (
    f"Metadata vector length {len(_sample_vec)} != checkpoint tab_dim {tab_dim}"
)
print(f"  Metadata ready. dim={tab_dim}, patients={len(meta_map)}")

# ---------------------------------------------------------------------------
# Instantiate model and load weights
# ---------------------------------------------------------------------------
print("Instantiating model ...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name for _YOLOv5MidBase: "yolov5l_cls_xattn_mid" → "yolov5l-cls"
model_file = backbone_name.split("_xattn")[0].replace("_cls", "-cls")  # yolov5l-cls
# _YOLOv5MidBase looks for the .pt file in the CWD, so run from REPO_ROOT
os.chdir(REPO_ROOT)

model = YOLOv5MidFusionCLSCrossAttn(
    model_name=model_file,
    tab_dim=tab_dim,
    pretrained=False,
    out_dim=2,
).to(device)
model.load_state_dict(ck["model"])
model.eval()
print(f"  Model loaded on {device}.")

# ---------------------------------------------------------------------------
# Find GradCAM target layer: last Conv2d before model.prepool
# ---------------------------------------------------------------------------
def find_gradcam_target(model: nn.Module) -> str:
    """
    Walk all named modules of `model`. Return the name of the last nn.Conv2d
    whose module appears before the AdaptiveAvgPool2d (model.prepool) in the
    module iteration order.
    """
    prepool = model.prepool
    all_named = list(model.named_modules())
    prepool_idx = None
    for i, (_, mod) in enumerate(all_named):
        if mod is prepool:
            prepool_idx = i
            break
    if prepool_idx is None:
        raise RuntimeError("Could not find model.prepool in named_modules()")
    last_conv_name = None
    for name, mod in all_named[:prepool_idx]:
        if isinstance(mod, nn.Conv2d):
            last_conv_name = name
    if last_conv_name is None:
        raise RuntimeError("No Conv2d found before prepool")
    return last_conv_name

target_layer = find_gradcam_target(model)
print(f"  GradCAM target layer: {target_layer}")
cam = GradCAM(model, target_layer_name=target_layer)

# ---------------------------------------------------------------------------
# Load and filter test set
# ---------------------------------------------------------------------------
test_df = pd.read_csv(test_csv)
lesion_df = test_df[test_df["label"] == 1].copy()
print(f"\nTest frames total    : {len(test_df)}")
print(f"Lesion frames (label=1): {len(lesion_df)}")

# Keep only rows that have a matching bbox label file.
# Groundtruth lives next to input/:  .../pX/vY/groundtruth/<stem>.txt
def label_path_for(img_path: str) -> str:
    stem     = os.path.splitext(os.path.basename(img_path))[0]
    input_dir = os.path.dirname(os.path.join(REPO_ROOT, img_path))   # .../pX/vY/input
    gt_dir    = os.path.join(os.path.dirname(input_dir), "groundtruth")
    return os.path.join(gt_dir, stem + ".txt")

lesion_df["label_file"] = lesion_df["img_path"].apply(label_path_for)
lesion_df = lesion_df[lesion_df["label_file"].apply(os.path.exists)].reset_index(drop=True)
print(f"Frames with bbox file : {len(lesion_df)}")

if len(lesion_df) == 0:
    print("No frames passed both filters — nothing to do.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Helper: parse YOLO label file → list of (x1,y1,x2,y2) in pixel coords
# ---------------------------------------------------------------------------
def parse_gt_labels(label_file: str):
    """
    Parse CADICA groundtruth format: x_topleft y_topleft width height label
    Coordinates are in pixels (integers).
    """
    boxes = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x1, y1, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            boxes.append((x1, y1, x1 + bw, y1 + bh))
    return boxes

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print(f"\nGenerating heatmaps and bbox overlays ...")
skipped = 0

for i, row in lesion_df.iterrows():
    img_path_rel = row["img_path"]
    img_path     = os.path.join(REPO_ROOT, img_path_rel)

    if not os.path.isfile(img_path):
        print(f"  [SKIP] image not found: {img_path_rel}")
        skipped += 1
        continue

    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_name = f"{stem}.png"

    # ---- load original grayscale image ----
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"  [SKIP] cv2 failed to read: {img_path_rel}")
        skipped += 1
        continue
    H, W = img_gray.shape[:2]

    # ---- build model input (resize to 224, ImageNet-normalise, replicate to 3ch) ----
    img_in   = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_AREA)
    img_rgb  = np.stack([img_in, img_in, img_in], axis=-1).astype(np.float32) / 255.0  # HxWx3
    mean     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std                              # HxWx3
    x = torch.tensor(img_norm.transpose(2, 0, 1)[None]).to(device).requires_grad_(True)  # 1x3x224x224

    # ---- build clinical vector ----
    pid = int(row["patient"])
    clin_np = np.asarray(meta_map.get(pid, np.zeros(tab_dim)), dtype=np.float32)
    clin = torch.tensor(clin_np)[None].to(device)

    # ---- GradCAM ----
    def forward_fn():
        _, logits = model(x, clin)
        return logits

    _, cam_t = cam(forward_fn, class_idx=1)

    cam_up = torch.nn.functional.interpolate(
        cam_t, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()

    heatmap_img = overlay_heatmap(img_gray, cam_up)
    cv2.imwrite(os.path.join(heatmap_dir, out_name), heatmap_img)

    # ---- bounding box overlay ----
    boxes = parse_gt_labels(row["label_file"])
    bbox_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
    cv2.imwrite(os.path.join(bbox_dir, out_name), bbox_img)

    # ---- side-by-side ----
    divider = np.full((H, 4, 3), 200, dtype=np.uint8)  # thin grey separator
    sbs_img = np.concatenate([heatmap_img, divider, bbox_img], axis=1)
    cv2.imwrite(os.path.join(sbs_dir, out_name), sbs_img)

    if (i + 1) % 50 == 0 or i == len(lesion_df) - 1:
        print(f"  {i + 1}/{len(lesion_df)} done ...")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
saved = len(lesion_df) - skipped
print(f"\nDone. {saved} image pairs saved to:")
print(f"  {heatmap_dir}")
print(f"  {bbox_dir}")
print(f"  {sbs_dir}")
if skipped:
    print(f"  ({skipped} skipped due to missing image files)")
