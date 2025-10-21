#!/usr/bin/env python3
"""
5-fold CV trainer WITH augmentation for training.
- Uses get_train_tf for training and get_val_tf for validation
- Computes frame- and video-level metrics per fold and aggregates results
"""
import os
import sys
import json
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_list(s: str, cast=str):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def create_balanced_sampler(dataset: CadicaDataset) -> WeightedRandomSampler:
    labels = [dataset[i]["y"].item() for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (len(counts) * counts)
    sample_weights = [weights[y] for y in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def build_models(backbone: str, fusion: str, tab_dim: int, pat_channels: int, device: str):
    backbone = backbone.lower()
    fusion = fusion.lower()
    clin_mlp = head = None

    if fusion == "late":
        # feature extractors + tabular MLP + late fusion head
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            img_model = ResNetBackbone(pretrained=True).to(device); img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            img_model = ResNet18Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            img_model = InceptionV3Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="tiny", pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="base", pretrained=True).to(device); img_feat_dim = img_model.feat_dim
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
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3EarlyFusionPatient
            img_model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(variant="tiny", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(variant="base", tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
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
    # args
    ap = argparse.ArgumentParser(description="K-fold CV training (with augmentation) for image-only / late / early fusion.")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--fusions", default="imageonly,late,early")
    ap.add_argument("--backbones", default="resnet50,resnet18,effnet_b0,convnext_tiny,inception_v3,densenet121,mobilenet_v2,vit_b16")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outroot", default="outputs/checkpoints/cv_runs_aug")
    ap.add_argument("--video_reduce", default="max", choices=["max","mean"], help="video-level prob reduce")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    patience = args.patience if args.patience is not None else cfg.get("early_stopping_patience", 20)

    os.makedirs(args.outroot, exist_ok=True)

    fusions = [x.lower() for x in parse_list(args.fusions, str)]
    backbones = [x.lower() for x in parse_list(args.backbones, str)]

    # per-config common dataset knobs
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    sel = cfg.get("metadata_features", {}) or {}
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)
    img_size = cfg.get("img_size", 224)

    # discover folds
    fold_dirs = [d for d in sorted(os.listdir(args.folds_root)) if d.startswith("fold_")]
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {args.folds_root}. Generate with scripts/01b_make_patient_stratified_folds.py")

    summary_rows = []

    for backbone in backbones:
        for fusion in fusions:
            run_root = os.path.join(args.outroot, f"{backbone}_{fusion}")
            os.makedirs(run_root, exist_ok=True)
            print(f"\n=== Running CV (aug) for backbone={backbone} fusion={fusion} ===")

            fold_metrics = []

            for fold_name in fold_dirs:
                fold_idx = int(fold_name.split("_")[-1])
                fold_path = os.path.join(args.folds_root, fold_name)
                train_csv = os.path.join(fold_path, "train.csv")
                val_csv = os.path.join(fold_path, "val.csv")

                # datasets (WITH augmentation for training)
                train_ds = CadicaDataset(
                    train_csv, metadata, get_train_tf(img_size),
                    selected_features=sel_nc,
                    text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                )
                val_ds = CadicaDataset(
                    val_csv, metadata, get_val_tf(img_size),
                    selected_features=sel_nc,
                    text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                )

                tr_loader = DataLoader(
                    train_ds,
                    batch_size=cfg.get("batch_size", 64),
                    sampler=create_balanced_sampler(train_ds),
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                va_loader = DataLoader(
                    val_ds,
                    batch_size=cfg.get("batch_size", 64),
                    shuffle=False,
                    num_workers=args.num_workers,
                )

                # model(s)
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
                if opt_name in ("adamw", "adam"):
                    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
                elif opt_name in ("sgd", "sgdm"):
                    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
                elif opt_name == "rmsprop":
                    optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
                else:
                    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

                # training loop
                best = -1.0
                best_metrics = None
                fold_out = os.path.join(run_root, fold_name)
                os.makedirs(fold_out, exist_ok=True)
                ckpt_path = os.path.join(fold_out, "best.pt")
                epochs_no_improve = 0

                for epoch in range(1, epochs + 1):
                    print(f"\n{backbone}/{fusion} | {fold_name} | Epoch {epoch}/{epochs}")
                    train_one_epoch(
                        img_model, optimizer, tr_loader, device,
                        mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                    )
                    frame_metrics, y_true_f, y_prob_f, video_metrics, video_df = evaluate(
                        img_model, va_loader, device,
                        mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                        also_video=True, video_reduce=args.video_reduce,
                    )
                    print("[val/frame]", {k: round(float(v), 4) for k, v in frame_metrics.items()})
                    print("[val/video]", {k: round(float(v), 4) for k, v in video_metrics.items()})

                    score = float(frame_metrics.get("bal_acc", 0.0))
                    if score > best:
                        best = score
                        best_metrics = {
                            "frame": {k: float(v) for k, v in frame_metrics.items()},
                            "video": {k: float(v) for k, v in video_metrics.items()},
                        }
                        torch.save(
                            {
                                "cfg": cfg,
                                "fusion": fusion,
                                "backbone": backbone,
                                "tab_dim": int(train_ds.tab_dim),
                                "model": img_model.state_dict(),
                                "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
                                "late_head": late_head.state_dict() if late_head else None,
                                "best_val": best_metrics,
                            },
                            ckpt_path,
                        )
                        video_df.to_csv(os.path.join(fold_out, "val_video_probs.csv"), index=False)
                        epochs_no_improve = 0
                        print(f"[ckpt] new best saved: bal_acc={best:.4f}")
                    else:
                        epochs_no_improve += 1
                        print(f"[early-stopping] no improvement for {epochs_no_improve} epochs")

                    if patience and epochs_no_improve >= patience:
                        print(f"[early-stopping] stopping at epoch {epoch}")
                        break

                # record fold best
                row = {
                    "backbone": backbone,
                    "fusion": fusion,
                    "fold": fold_idx,
                    **{f"val_frame_{k}": best_metrics["frame"][k] for k in best_metrics["frame"]},
                    **{f"val_video_{k}": best_metrics["video"][k] for k in best_metrics["video"]},
                }
                fold_metrics.append(row)
                with open(os.path.join(fold_out, "val_best.json"), "w") as f:
                    json.dump(row, f, indent=2)

            # aggregate across folds
            df = pd.DataFrame(fold_metrics)
            df.to_csv(os.path.join(run_root, "cv_val_folds.csv"), index=False)
            for col in [c for c in df.columns if c.startswith("val_")]:
                mu, sd = float(df[col].mean()), float(df[col].std())
                print(f"[summary] {backbone}/{fusion} {col}: {mu:.4f} ± {sd:.4f}")

    print("\n[done] Wrote per-run fold results with augmentation.")


if __name__ == "__main__":
    main()
