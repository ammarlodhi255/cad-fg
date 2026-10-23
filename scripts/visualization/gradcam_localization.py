#!/usr/bin/env python3
import os
import sys
import json
import argparse
import importlib.util
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradcam import GradCAM
from src.dataio.dataset import CadicaDataset
from src.dataio.transforms import get_val_tf


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SMALL = 128  # resolution of maps and masks for the box-swap control
METRICS = ["energy_in_box", "energy_ratio", "energy_ratio_boxswap", "pointing_hit"]


def load_build_models():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models


def load_model(path, build_models, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    img, _, _, _ = build_models(backbone=ck["backbone"], fusion=ck["fusion"], tab_dim=int(ck["tab_dim"]),
                                pat_channels=ck["cfg"].get("pat_channels", 4), device=device)
    img.load_state_dict(ck["model"])
    img.eval()
    return img, ck


def last_conv_before_pool(model):
    named = list(model.named_modules())
    pool_idx = next(i for i, (_, m) in enumerate(named) if isinstance(m, nn.AdaptiveAvgPool2d))
    return [n for n, m in named[:pool_idx] if isinstance(m, nn.Conv2d)][-1]


def gt_boxes(img_path):
    # CADICA boxes: x_topleft y_topleft width height
    stem = os.path.splitext(os.path.basename(img_path))[0]
    f = os.path.join(os.path.dirname(os.path.dirname(img_path)), "groundtruth", stem + ".txt")
    if not os.path.isfile(f):
        return None
    boxes = []
    for line in open(f):
        p = line.split()
        if len(p) >= 4:
            x, y, w, h = map(int, p[:4])
            boxes.append((x, y, x + w, y + h))
    return boxes or None


def frame_input(gray, device):
    img = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = np.stack([img] * 3, axis=-1).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
    return torch.tensor(x, device=device).requires_grad_(True)


def cam_map(cam, model, x, clin, H, W, fusion):
    def fwd():
        _, logits = model(x, clin) if fusion != "imageonly" else model(x)
        return logits
    _, c = cam(fwd, class_idx=1)
    return F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()


def measures(c, mask):
    total = c.sum()
    area = float(mask.mean())
    if total <= 0:
        return None
    e = float(c[mask].sum() / total)
    yx = np.unravel_index(np.argmax(c), c.shape)
    return {"energy_in_box": e, "area_fraction": area, "energy_ratio": e / area,
            "activated_area": float((c >= 0.5 * c.max()).mean()), "pointing_hit": int(mask[yx])}


def small(a, binary=False):
    a = cv2.resize(a.astype(np.float32), (SMALL, SMALL), interpolation=cv2.INTER_AREA)
    return a > 0.5 if binary else a


def summarize(d, n_boot=5000, seed=1337):
    # empty Grad-CAM maps are counted as no saliency on the lesion (energy 0, ratio 0, no hit)
    out = {"frames": int(len(d)), "videos": int(d.groupby(["patient", "video"]).ngroups),
           "patients": sorted(d["patient"].unique().tolist()),
           "mean_box_area_fraction": float(d["xttn_area_fraction"].mean()),
           "empty_maps": {m: int(d[f"{m}_energy_in_box"].isna().sum()) for m in ["xttn", "imageonly"]},
           "activated_area_nonempty_maps": {m: float(d[f"{m}_activated_area"].mean()) for m in ["xttn", "imageonly"]}}
    d = d.copy()
    for m in ["xttn", "imageonly"]:
        for k in METRICS:
            d[f"{m}_{k}"] = d[f"{m}_{k}"].fillna(0)
        out[m] = {k: float(d[f"{m}_{k}"].mean()) for k in METRICS}
        out[m]["energy_ratio_boxswap_median"] = float(d[f"{m}_energy_ratio_boxswap"].median())
        out[m]["energy_ratio_boxswap_share_above_1"] = float((d[f"{m}_energy_ratio_boxswap"] > 1).mean())
        out[m]["activated_area_median_nonempty"] = float(d.loc[d[f"{m}_activated_area"].notna(), f"{m}_activated_area"].median())
    groups = list(d.groupby(["patient", "video"]).indices.values())
    rng = np.random.default_rng(seed)
    idx = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]
    out["xttn_minus_imageonly"] = {}
    for k in METRICS:
        diff = (d[f"xttn_{k}"] - d[f"imageonly_{k}"]).to_numpy()
        boot = np.array([diff[i].mean() for i in idx])
        out["xttn_minus_imageonly"][k] = {
            "mean": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_xttn", required=True)
    ap.add_argument("--ckpt_img", required=True)
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv")
    ap.add_argument("--out_dir", default="outputs/analysis")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_models = load_build_models()

    models = {"xttn": load_model(args.ckpt_xttn, build_models, args.device),
              "imageonly": load_model(args.ckpt_img, build_models, args.device)}
    cfg = models["xttn"][1]["cfg"]
    sel = cfg.get("metadata_features", {}) or {}
    meta = CadicaDataset(args.test_csv, cfg.get("metadata_path", "data/CADICA/metadata.xlsx"), get_val_tf(224),
                         selected_features={"numeric": sel.get("numeric"), "categorical": sel.get("categorical")},
                         text_features=sel.get("text", None),
                         text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                         text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds")).meta
    cams = {}
    for name, (m, ck) in models.items():
        layer = last_conv_before_pool(m)
        cams[name] = (GradCAM(m, target_layer_name=layer), ck["fusion"])
        print(f"{name}: fusion={ck['fusion']} target layer={layer}", flush=True)

    df = pd.read_csv(args.test_csv)
    df = df[df["label"] == 1].reset_index(drop=True)
    rows, small_maps, small_masks = [], {n: [] for n in cams}, []
    for _, r in df.iterrows():
        boxes = gt_boxes(r["img_path"])
        if boxes is None:
            continue
        gray = cv2.imread(r["img_path"], cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        mask = np.zeros((H, W), dtype=bool)
        for (x1, y1, x2, y2) in boxes:
            mask[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = True
        clin = torch.tensor(np.asarray(meta[int(r["patient"])], dtype=np.float32)[None], device=args.device)
        row = {"patient": int(r["patient"]), "video": int(r["video"]), "frame": int(r["frame"]), "n_boxes": len(boxes)}
        for name, (cam, fusion) in cams.items():
            c = cam_map(cam, models[name][0], frame_input(gray, args.device), clin, H, W, fusion)
            m = measures(c, mask)
            for k in ["energy_in_box", "area_fraction", "energy_ratio", "activated_area", "pointing_hit"]:
                row[f"{name}_{k}"] = (m or {}).get(k, np.nan)
            small_maps[name].append(small(c))
        small_masks.append(small(mask, binary=True))
        rows.append(row)

    d = pd.DataFrame(rows)
    # box-swap control: saliency inside the lesion boxes of frames from other patients
    masks = np.stack(small_masks).reshape(len(d), -1).astype(np.float32)
    pats = d["patient"].to_numpy()
    for n in cams:
        M = np.stack(small_maps[n]).reshape(len(d), -1)
        tot, own, cross = M.sum(1), (M * masks).sum(1), M @ masks.T
        expected = np.array([cross[i, pats != pats[i]].mean() for i in range(len(d))])
        with np.errstate(divide="ignore", invalid="ignore"):
            d[f"{n}_energy_ratio_boxswap"] = np.where(tot > 0, own / np.maximum(expected, 1e-12), np.nan)
    d.to_csv(os.path.join(args.out_dir, "gradcam_localization.csv"), index=False)
    summary = summarize(d, n_boot=args.n_boot)
    with open(os.path.join(args.out_dir, "gradcam_localization.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
