#!/usr/bin/env python3
import os
import json
import argparse
import importlib.util
import urllib.request
import pandas as pd
import torch


TAB_DIM = 406       # metadata vector length of the trained models
PAT_CHANNELS = 4    # context channels of early fusion
YOLOV5_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/{}-cls.pt"
CNN_VIT = [("ResNet50", "resnet50"), ("VGG19", "vgg19"), ("ViT-B16", "vit_b16"),
           ("ConvNeXt-Base", "convnext_base"), ("ConvNeXt-Tiny", "convnext_tiny")]
YOLO = [f"yolov5{s}" for s in "nsmlx"] + [f"yolov8{s}" for s in "nsmlx"]
MID = {"FiLM": "{}_film_multi2_mid", "Xttn": "{}_cls_xattn_mid", "Meta": "{}_meta_adapter"}


def variants():
    rows = []
    for label, bb in CNN_VIT:
        for fu in ("early", "late"):
            rows.append(("9", f"{label} ({fu})", bb, fu, bb))
    for y in YOLO:
        name = y.upper().replace("YOLOV", "YOLOv")
        for fu in ("early", "late"):
            rows.append(("9", f"{name} ({fu})", f"{y}-cls", fu, f"{y}-cls"))
        if y[-1] in "smlx":
            for tag, pattern in MID.items():
                rows.append(("10", f"{name} ({tag})", pattern.format(y), "early", f"{y}-cls"))
    return rows


def load_build_models():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models


def ensure_yolov5_weights():
    for s in "nsmlx":
        f = f"yolov5{s}-cls.pt"
        if not os.path.isfile(f):
            urllib.request.urlretrieve(YOLOV5_URL.format(f"yolov5{s}"), f)


def n_params(params):
    # total and trainable (requires_grad) parameters handed to the optimizer, each counted once
    seen, total, trainable = set(), 0, 0
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
    return total, trainable


def gflops(img_model, clin_mlp, head, fusion):
    from torch.utils.flop_counter import FlopCounterMode
    x, z = torch.zeros(1, 3, 224, 224), torch.zeros(1, TAB_DIM)
    for m in (img_model, clin_mlp, head):
        if m is not None:
            m.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        if fusion == "late":
            feats, _ = img_model(x)
            head(feats, clin_mlp(z))
        elif fusion == "early":
            img_model(x, z)
        else:
            img_model(x)
    return fc.get_total_flops() / 1e9


def film_sites(img_model):
    sites = getattr(img_model, "_site_mlps", None)
    if not sites:
        return None
    names = {id(m): n for n, m in img_model.base.named_modules()}
    return [{"layer": names.get(id(mod), type(mod).__name__), "type": type(mod).__name__,
             "channels": int(mlp[-1].out_features // 2)} for mod, mlp in sites.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ensure_yolov5_weights()
    build_models = load_build_models()

    cache, rows = {}, []

    def build(backbone, fusion):
        key = (backbone, fusion)
        if key not in cache:
            img, clin, head, params = build_models(backbone=backbone, fusion=fusion, tab_dim=TAB_DIM,
                                                   pat_channels=PAT_CHANNELS, device="cpu")
            total, trainable = n_params(params)
            cache[key] = {"total": total, "trainable": trainable,
                          "gflops": round(gflops(img, clin, head, fusion), 3), "film_sites": film_sites(img)}
            del img, clin, head, params
        return cache[key]

    for table, label, backbone, fusion, base in variants():
        m, ref = build(backbone, fusion), build(base, "imageonly")
        rows.append({"table": table, "model": label, "backbone": backbone, "fusion": fusion,
                     "total": m["total"], "trainable": m["trainable"],
                     "total_imageonly": ref["total"], "trainable_imageonly": ref["trainable"],
                     "added_total": m["total"] - ref["total"],
                     "added_trainable": m["trainable"] - ref["trainable"],
                     "added_pct_of_total": round(100 * (m["total"] - ref["total"]) / ref["total"], 2),
                     "gflops": m["gflops"], "gflops_imageonly": ref["gflops"], "film_sites": m["film_sites"]})
        r = rows[-1]
        print(f"{label:24s} total {r['total']:>12,d} (+{r['added_total']:,d}) | trainable {r['trainable']:>12,d} "
              f"(image-only {r['trainable_imageonly']:,d}) | {m['gflops']} GFLOPs", flush=True)

    with open(os.path.join(args.out_dir, "model_param_counts.json"), "w") as f:
        json.dump({"tab_dim": TAB_DIM, "pat_channels": PAT_CHANNELS, "rows": rows}, f, indent=2)
    pd.DataFrame(rows).drop(columns=["film_sites"]).to_csv(os.path.join(args.out_dir, "model_param_counts.csv"), index=False)


if __name__ == "__main__":
    main()
