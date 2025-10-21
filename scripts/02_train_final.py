#!/usr/bin/env python3
import os
import sys
import json
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_models(backbone: str, fusion: str, tab_dim: int, pat_channels: int, device: str):
    backbone = backbone.lower(); fusion = fusion.lower()
    clin_mlp = head = None

    if fusion == "late":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            img_model = ResNetBackbone(pretrained=True).to(device); img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            img_model = ResNet18Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7Backbone
            img_model = EfficientNetB7Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            img_model = InceptionV3Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="tiny", pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="base", pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone.startswith("convnextv2_") or backbone == "convnextv2":
            from src.models.convnextv2_backbone import ConvNeXtV2Backbone
            variant = backbone.split("_", 1)[1] if "_" in backbone else "tiny"
            img_model = ConvNeXtV2Backbone(variant=variant, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone
            img_model = MobileNetV2Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone
            img_model = DenseNet121Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone
            img_model = AlexNetBackbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone
            img_model = ViTBackbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True, cbam=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM
            img_model = ConvNeXtBaseBackboneCBAM(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "metafusionnet_base":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            img_model = MetaFusionConvNeXtBackbone(variant="base", pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "metafusionnet_tiny":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            img_model = MetaFusionConvNeXtBackbone(variant="tiny", pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone.startswith("regnet_"):
            from src.models.regnet_backbone import RegNetBackbone
            img_model = RegNetBackbone(variant=backbone, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone.startswith("repvgg"):
            from src.models.repvgg_backbone import RepVGGBackbone
            img_model = RepVGGBackbone(model_name=backbone, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
        return img_model, clin_mlp, head, params

    if fusion == "early":
        if backbone == "resnet50":
            from src.models.early_fusion_patient import ResNetEarlyFusionPatient
            img_model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18EarlyFusionPatient
            img_model = ResNet18EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0EarlyFusionPatient
            img_model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7EarlyFusionPatient
            img_model = EfficientNetB7EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3EarlyFusionPatient
            img_model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(variant="tiny", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(variant="base", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("convnextv2_") or backbone == "convnextv2":
            from src.models.convnextv2_backbone import ConvNeXtV2EarlyFusionPatient
            variant = backbone.split("_", 1)[1] if "_" in backbone else "tiny"
            img_model = ConvNeXtV2EarlyFusionPatient(variant=variant, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2EarlyFusionPatient
            img_model = MobileNetV2EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121EarlyFusionPatient
            img_model = DenseNet121EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetEarlyFusionPatient
            img_model = AlexNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTEarlyFusionPatient
            img_model = ViTEarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0EarlyFusionPatient
            img_model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2, cbam=True).to(device)
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseEarlyFusionPatientCBAM
            img_model = ConvNeXtBaseEarlyFusionPatientCBAM(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "metafusionnet_base":
            from src.models.metafusionnet_backbone import MetaFusionNetEarlyFusionPatient
            img_model = MetaFusionNetEarlyFusionPatient(variant="base", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "metafusionnet_tiny":
            from src.models.metafusionnet_backbone import MetaFusionNetEarlyFusionPatient
            img_model = MetaFusionNetEarlyFusionPatient(variant="tiny", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(variant="tiny", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(variant="base", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("regnet_"):
            from src.models.regnet_backbone import RegNetEarlyFusionPatient
            img_model = RegNetEarlyFusionPatient(variant=backbone, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("repvgg"):
            from src.models.repvgg_backbone import RepVGGEarlyFusionPatient
            img_model = RepVGGEarlyFusionPatient(model_name=backbone, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        params = list(img_model.parameters())
        return img_model, None, None, params

    if fusion == "imageonly":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            img_model = ResNetBackbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            img_model = ResNet18Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            img_model = InceptionV3Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="tiny", pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="base", pretrained=True, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone
            img_model = MobileNetV2Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone
            img_model = DenseNet121Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone
            img_model = AlexNetBackbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone
            img_model = ViTBackbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2, cbam=True).to(device)
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM
            img_model = ConvNeXtBaseBackboneCBAM(pretrained=True, out_dim=2).to(device)
        elif backbone == "metafusionnet_base":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            img_model = MetaFusionConvNeXtBackbone(variant="base", pretrained=True, out_dim=2).to(device)
        elif backbone == "metafusionnet_tiny":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            img_model = MetaFusionConvNeXtBackbone(variant="tiny", pretrained=True, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        params = list(img_model.parameters())
        return img_model, None, None, params

    raise ValueError("fusion must be one of {'late','early','imageonly'}")


def main():
    ap = argparse.ArgumentParser(description="Final training on train+val, evaluate on test.")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--backbone", default=None, help="override cfg backbone")
    ap.add_argument("--fusion", default=None, help="override cfg fusion: {imageonly,late,early}")
    ap.add_argument("--no_aug", action="store_true", help="use validation transforms for training (no augs)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None, help="optional early stopping on training set is disabled; kept for API parity")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outroot", default="outputs/checkpoints/final_runs")
    ap.add_argument(
        "--video_reduce",
        default="all",
        choices=[
            "max","mean","median","topk2","topk3","logit_mean","ew_mean","tsmooth_mean","tsmooth_max","all"
        ],
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    if args.backbone: cfg["backbone"] = args.backbone
    if args.fusion: cfg["fusion"] = args.fusion
    if args.epochs is not None: cfg["epochs"] = args.epochs

    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # combine train + val
    tcsv, vcsv = "outputs/index/train.csv", "outputs/index/val.csv"
    os.makedirs("outputs/index", exist_ok=True)
    df_tr = pd.read_csv(tcsv)
    df_va = pd.read_csv(vcsv)
    df_combined = pd.concat([df_tr, df_va], axis=0).drop_duplicates().reset_index(drop=True)
    comb_csv = "outputs/index/trainval.csv"
    df_combined.to_csv(comb_csv, index=False)
    print(f"[final] combined train+val -> {len(df_combined)} frames")

    # datasets
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    sel = cfg.get("metadata_features", {}) or {}
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)

    tf_train = get_val_tf(cfg["img_size"]) if args.no_aug else get_train_tf(cfg["img_size"])
    train_ds = CadicaDataset(
        comb_csv, metadata, tf_train,
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    test_ds = CadicaDataset(
        "outputs/index/test.csv", metadata, get_val_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )

    tr = DataLoader(train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(test_ds,  batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=args.num_workers)

    # build model
    backbone = cfg.get("backbone", "resnet50")
    fusion = cfg.get("fusion", "imageonly")
    img_model, clin_mlp, late_head, params = build_models(
        backbone=backbone,
        fusion=fusion,
        tab_dim=train_ds.tab_dim,
        pat_channels=cfg.get("pat_channels", 4),
        device=device,
    )

    # optimizer
    opt_name = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 1e-4)
    if opt_name in ("adamw", "adam"): optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name in ("sgd", "sgdm"): optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    elif opt_name == "rmsprop": optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    else: optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # out dirs
    run_root = os.path.join(args.outroot, f"{backbone}_{fusion}")
    os.makedirs(run_root, exist_ok=True)
    ckpt_path = os.path.join(run_root, "final.pt")

    # train full epochs (no early stopping on combined set)
    for epoch in range(1, cfg.get("epochs", 100) + 1):
        print(f"\n[final] {backbone}/{fusion} | Epoch {epoch}/{cfg.get('epochs', 100)}")
        train_one_epoch(
            img_model, optimizer, tr, device,
            mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
        )

    # save final model
    torch.save(
        {
            "cfg": cfg,
            "fusion": fusion,
            "backbone": backbone,
            "tab_dim": int(train_ds.tab_dim),
            "model": img_model.state_dict(),
            "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
            "late_head": late_head.state_dict() if late_head else None,
        },
        ckpt_path,
    )
    print(f"[final] saved -> {ckpt_path}")

    # evaluate on test set (frame + video)
    frame_metrics, y_true_f, y_prob_f, video_metrics, video_df = evaluate(
        img_model, te, device,
        mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
        also_video=True, video_reduce=args.video_reduce,
    )

    print("[test/frame]", {k: round(float(v), 4) for k, v in frame_metrics.items()})
    print("[test/video]", {k: round(float(v), 4) for k, v in video_metrics.items()})

    # persist results
    video_df.to_csv(os.path.join(run_root, "test_video_probs.csv"), index=False)
    with open(os.path.join(run_root, "test_metrics.json"), "w") as f:
        json.dump({
            "frame": {k: float(v) for k, v in frame_metrics.items()},
            "video": {k: float(v) for k, v in video_metrics.items()},
        }, f, indent=2)

    print("[done] Final test metrics saved.")


if __name__ == "__main__":
    main()
