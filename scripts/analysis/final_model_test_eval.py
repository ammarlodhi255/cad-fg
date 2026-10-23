#!/usr/bin/env python3
import os
import sys
import json
import argparse
import importlib.util
import torch
import yaml
from torch.utils.data import DataLoader

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dataio.dataset import CadicaDataset
from src.dataio.transforms import get_val_tf
from src.trainloop.engine import evaluate


def load_build_models():
    # build_models lives in scripts/02_train_kfold.py (module name starts with a digit)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="final_full_train.pt written by 02_train_kfold.py --only_final")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--video_reduce", default="tsmooth_max")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or yaml.safe_load(open("configs/default.yaml"))
    fusion, backbone, tab_dim = ck["fusion"], ck["backbone"], int(ck["tab_dim"])
    print(f"checkpoint: backbone={backbone} fusion={fusion} tab_dim={tab_dim}")

    sel = cfg.get("metadata_features", {}) or {}
    test_ds = CadicaDataset(
        args.test_csv, cfg.get("metadata_path", "data/CADICA/metadata.xlsx"),
        get_val_tf(cfg.get("img_size", 224)),
        selected_features={"numeric": sel.get("numeric"), "categorical": sel.get("categorical")},
        text_features=sel.get("text", None),
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    assert test_ds.tab_dim == tab_dim, f"tab_dim mismatch: data {test_ds.tab_dim} vs ckpt {tab_dim}"
    loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=2)

    build_models = load_build_models()
    img_model, clin_mlp, late_head, _ = build_models(
        backbone=backbone, fusion=fusion, tab_dim=tab_dim,
        pat_channels=cfg.get("pat_channels", 4), device=args.device)
    img_model.load_state_dict(ck["model"])
    if clin_mlp is not None and ck.get("clin_mlp") is not None:
        clin_mlp.load_state_dict(ck["clin_mlp"])
    if late_head is not None and ck.get("late_head") is not None:
        late_head.load_state_dict(ck["late_head"])

    fmet, y_frame, p_frame, vmet, vdf = evaluate(
        img_model, loader, args.device, mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
        also_video=True, video_reduce=args.video_reduce)

    video = {k: float(v) for k, v in vmet.items() if k != "_multi"}
    metrics = {"backbone": backbone, "fusion": fusion, "checkpoint": os.path.abspath(args.ckpt),
               "test_csv": args.test_csv, "n_frames": int(len(y_frame)),
               "frame": {k: float(v) for k, v in fmet.items()}, "video": video}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    frames = test_ds.df.copy()
    assert len(frames) == len(p_frame), "prediction count does not match test rows"
    frames["y_true"] = y_frame
    frames["p_lesion"] = p_frame
    frames.to_csv(os.path.join(args.out_dir, "frame_predictions.csv"), index=False)
    if vdf is not None:
        vdf.to_csv(os.path.join(args.out_dir, "video_predictions.csv"), index=False)

    print("frame:", {k: round(v, 4) for k, v in metrics["frame"].items()})
    print("video:", {k: round(v, 4) for k, v in video.items()})


if __name__ == "__main__":
    main()
