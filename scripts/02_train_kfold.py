#!/usr/bin/env python3
import os
# Suppress HuggingFace tokenizers fork/parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import json
import cv2
import time
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate
from src.trainloop.metrics import metrics_from_probs


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def parse_list(s: str, cast=str):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


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
        elif backbone == "vit_b16_384":
            from src.models.vit384_backbone import ViTBackbone384
            img_model = ViTBackbone384(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True, cbam=True).to(device); img_feat_dim = img_model.feat_dim
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
        elif backbone == "vit_b16_meta":
            from src.models.vit_early_fusion_variants import ViTEarlyFusionMetaToken
            img_model = ViTEarlyFusionMetaToken(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_film":
            from src.models.vit_early_fusion_variants import ViTEarlyFusionFiLM
            img_model = ViTEarlyFusionFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_prompt":
            from src.models.vit_early_fusion_variants import ViTEarlyFusionPrompt
            img_model = ViTEarlyFusionPrompt(tab_dim=tab_dim, K=4, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_prompt_deep":
            from src.models.vit_early_fusion_variants import ViTMidFusionDeepPrompt
            img_model = ViTMidFusionDeepPrompt(tab_dim=tab_dim, K=4, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_cls_xattn":
            from src.models.vit_early_fusion_variants import ViTEarlyFusionCLSCrossAttn
            img_model = ViTEarlyFusionCLSCrossAttn(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_cls_xattn_deep":
            from src.models.vit_mid_fusion_variants import ViTMidFusionCLSPeriodicXAttn
            img_model = ViTMidFusionCLSPeriodicXAttn(tab_dim=tab_dim, M=2, nheads=8, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_film_deep":
            from src.models.vit_mid_fusion_variants import ViTMidFusionResidualFiLM
            img_model = ViTMidFusionResidualFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_meta_adapter":
            from src.models.vit_mid_fusion_variants import ViTMidFusionMetaAdapter
            img_model = ViTMidFusionMetaAdapter(tab_dim=tab_dim, r=64, pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_384":
            from src.models.vit384_backbone import ViTEarlyFusionPatient384
            img_model = ViTEarlyFusionPatient384(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0EarlyFusionPatient
            img_model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2, cbam=True).to(device)
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
        elif backbone in ("swin_tiny_224", "swin_t_224", "swint224", "swin-t-224"):
            from src.models.swin224_backbone import SwinTiny224Backbone
            img_model = SwinTiny224Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            img_model = ResNet18Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7Backbone
            img_model = EfficientNetB7Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            img_model = InceptionV3Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone in ("inception_resnet_v2", "inceptionresnetv2"):
            from src.models.inception_resnet_backbone import InceptionResNetV2Backbone
            img_model = InceptionResNetV2Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="tiny", pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="base", pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("convnextv2_"):
            from src.models.convnextv2_backbone import ConvNeXtV2Backbone
            variant = backbone.split("_", 1)[1]
            img_model = ConvNeXtV2Backbone(variant=variant, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnextv2":  # default to tiny
            from src.models.convnextv2_backbone import ConvNeXtV2Backbone
            img_model = ConvNeXtV2Backbone(variant="tiny", pretrained=True, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone
            img_model = MobileNetV2Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3Backbone
            img_model = MobileNetV3Backbone(variant="large", pretrained=True, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_small":
            from src.models.mobilenetv3_backbone import MobileNetV3Backbone
            img_model = MobileNetV3Backbone(variant="small", pretrained=True, out_dim=2).to(device)
        elif backbone == "mobilenet_v3":  # default alias -> large
            from src.models.mobilenetv3_backbone import MobileNetV3Backbone
            img_model = MobileNetV3Backbone(variant="large", pretrained=True, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone
            img_model = DenseNet121Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "densenet201":
            from src.models.densenet201_backbone import DenseNet201Backbone
            img_model = DenseNet201Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone
            img_model = AlexNetBackbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone
            img_model = ViTBackbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "vit_b16_384":
            from src.models.vit384_backbone import ViTBackbone384
            img_model = ViTBackbone384(pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
                          "yolov8n_cls", "yolov8s_cls", "yolov8m_cls", "yolov8l_cls", "yolov8x_cls"):
            from src.models.yolov8_cls_backbone import YOLOv8ClsBackbone
            # normalize to hyphenated convention for ultralytics
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv8ClsBackbone(model_name=model_name, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5s-cls", "yolov5m-cls", "yolov5l-cls", "yolov5x-cls",
                          "yolov5s_cls", "yolov5m_cls", "yolov5l_cls", "yolov5x_cls"):
            from src.models.yolov5_cls_backbone import YOLOv5ClsBackbone
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv5ClsBackbone(model_name=model_name, pretrained=True, out_dim=2).to(device)
        elif backbone in ("vgg16", "vgg-16"):
            from src.models.vgg_backbone import VGG16Backbone
            img_model = VGG16Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone in ("vgg19", "vgg-19"):
            from src.models.vgg_backbone import VGG19Backbone
            img_model = VGG19Backbone(pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("regnet_"):
            from src.models.regnet_backbone import RegNetBackbone
            img_model = RegNetBackbone(variant=backbone, pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("repvgg"):
            from src.models.repvgg_backbone import RepVGGBackbone
            img_model = RepVGGBackbone(model_name=backbone, pretrained=True, out_dim=2).to(device)
        elif backbone in ("nasnetalarge", "nasnet_a_large", "nasnet_alarge"):
            from src.models.nasnet_alarge_backbone import NASNetALargeBackbone
            img_model = NASNetALargeBackbone(pretrained=True, out_dim=2).to(device)
        elif backbone == "xception" or backbone.startswith("xception"):
            from src.models.xception_backbone import XceptionBackbone
            # if a specific xception variant is requested, pass it through
            model_name = backbone if backbone != "xception" else None
            img_model = XceptionBackbone(model_name=model_name, pretrained=True, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2, cbam=True).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        params = list(img_model.parameters())
        return img_model, None, None, params

    raise ValueError("fusion must be one of {'late','early','imageonly'}")


def main():
    # args
    ap = argparse.ArgumentParser(description="K-fold CV training for image-only / late / early fusion (non-PIM backbones).")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--fusions", default="imageonly,late,early")
    ap.add_argument("--backbones", default="resnet50,resnet18,effnet_b0,convnext_tiny,inception_v3,densenet121,mobilenet_v2,vit_b16")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outroot", default="outputs/checkpoints/cv")
    ap.add_argument(
        "--video_reduce",
        default="tsmooth_max",
        choices=[
            "max","mean","median","topk2","topk3","logit_mean","ew_mean","tsmooth_mean","tsmooth_max","all"
        ],
        help="video-level probability aggregation strategy",
    )
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--save_checkpoints", action="store_true", help="Save best.pt per fold (off by default)")
    ap.add_argument("--save_val_probs", action="store_true", help="Save val_video_probs.csv per fold (off by default)")
    ap.add_argument("--select_domain", default="frame", choices=["frame","video"], help="Which validation domain to select best epoch on")
    ap.add_argument("--select_metric", default="bal_acc", help="Validation metric key to select best epoch (e.g., bal_acc)")
    ap.add_argument("--inner_val_frac", type=float, default=0.2, help="Fraction of outer-train used as inner validation for selection (grouped by patient, stratified if possible)")
    ap.add_argument("--eval_outer_each_epoch", action="store_true", help="Also evaluate on outer held-out val every epoch (off by default)")
    # Optional inner-train augmentation (post inner split, inner-val untouched)
    ap.add_argument("--augment", action="store_true", help="Augment inner-train only (after inner split). Uses on-disk image synthesis; inner-val remains pristine.")
    ap.add_argument("--aug_label", type=int, choices=[0,1], default=0, help="Label to augment when --augment is set: 0=negatives, 1=positives")
    ap.add_argument("--aug_match_count", action="store_true", help="If set, augment chosen label to match the opposite label count in inner-train")
    ap.add_argument("--aug_factor", type=float, default=1.0, help="If --aug_match_count is not set, multiply chosen label count by this factor (e.g., 2.0 doubles)")
    ap.add_argument("--aug_max_per_frame", type=int, default=5, help="Max number of augmented copies per original frame in inner-train")
    ap.add_argument("--aug_img_size", type=int, default=None, help="Optional resize max-size for augmentation images")
    ap.add_argument("--aug_root", default="data/augmented-cadica", help="Directory to write augmented images under selectedVideos/p#/v#/input/")
    ap.add_argument("--final_test_eval", action="store_true", help="After CV per model, retrain on all non-test data and evaluate on pristine test")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv", help="Path to pristine test CSV")
    ap.add_argument("--trainval_pool_csv", default="outputs/index/trainval_pool.csv", help="Combined non-test pool for final training")
    # PIM-specific knobs mirroring original hyperparams
    ap.add_argument("--amp", default="on", choices=["on","off"], help="Enable AMP")
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

    def _compute_stats(df: pd.DataFrame):
        stats = {}
        cols = [c for c in df.columns if c.startswith("val_")]
        for col in sorted(cols):
            s = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "n": int(s.shape[0]),
            }
        return stats

    def _fmt_hms(seconds: float) -> str:
        try:
            seconds = float(seconds or 0.0)
        except Exception:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _write_model_summary_txt(path: str, backbone: str, fusion: str, df: pd.DataFrame, final_test: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stats = _compute_stats(df)
        lines = []
        lines.append(f"Model: {backbone}/{fusion}")
        lines.append(f"Folds completed: {max(df['fold'])+1 if 'fold' in df.columns else df.shape[0]}")
        for col in sorted(stats.keys()):
            st = stats[col]
            lines.append(f"{col}: {st['mean']:.4f} ± {st['std']:.4f} (min={st['min']:.4f}, max={st['max']:.4f}, n={st['n']})")

        # Convergence and timing info (if present)
        if "convergence_epoch" in df.columns:
            try:
                cavg = float(pd.to_numeric(df["convergence_epoch"], errors="coerce").mean())
                lines.append(f"Avg convergence epoch: {cavg:.2f}")
            except Exception:
                pass
        if "train_seconds" in df.columns:
            try:
                tsum = float(pd.to_numeric(df["train_seconds"], errors="coerce").sum())
                tmean = float(pd.to_numeric(df["train_seconds"], errors="coerce").mean())
                lines.append(f"Total train time: {_fmt_hms(tsum)} (mean per fold: {_fmt_hms(tmean)})")
            except Exception:
                pass
        # GPU peak memory (if present)
        if "cuda_mem_alloc_peak_mb" in df.columns:
            try:
                mu = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").mean())
                sd = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").std())
                lines.append(f"GPU peak alloc: {mu:.1f} ± {sd:.1f} MB")
            except Exception:
                pass
        if "cuda_mem_reserved_peak_mb" in df.columns:
            try:
                mu = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").mean())
                sd = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").std())
                lines.append(f"GPU peak reserved: {mu:.1f} ± {sd:.1f} MB")
            except Exception:
                pass
        if "gpu_name" in df.columns and "device" in df.columns:
            try:
                dev = str(df["device"].dropna().iloc[0])
                gname = str(df["gpu_name"].dropna().iloc[0])
                lines.append(f"Device: {dev} [{gname}]")
            except Exception:
                pass
        # Per-fold quick line for time and convergence
        if set(["fold","train_seconds"]).issubset(df.columns):
            try:
                for _, r in df.sort_values("fold").iterrows():
                    ce = r.get("convergence_epoch", None)
                    ce_str = "-" if ce is None or pd.isna(ce) else str(int(ce))
                    lines.append(f"fold_{int(r['fold'])}: train={_fmt_hms(r.get('train_seconds', 0))}, converge_epoch={ce_str}")
            except Exception:
                pass
        if final_test is not None:
            lines.append("")
            lines.append("Final Test (lockbox)")
            for dom in ("frame", "video"):
                key = final_test.get(dom, {})
                for k in sorted(key.keys()):
                    lines.append(f"test_{dom}_{k}: {float(key[k]):.4f}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_global_summary_txt(path: str, rows: list):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Reconstruct text from rows that contain mean/std/min/max for each metric
        lines = []
        lines.append("Cross-Model CV Summary")
        for row in rows:
            lines.append("")
            lines.append(f"Model: {row['backbone']}/{row['fusion']}")
            # collect metric prefixes
            metrics = {}
            for k, v in row.items():
                if k in ("backbone", "fusion", "fold"): continue
                if k.startswith("mean_"):
                    m = k[len("mean_"):]
                    metrics.setdefault(m, {})['mean'] = v
                elif k.startswith("std_"):
                    m = k[len("std_"):]
                    metrics.setdefault(m, {})['std'] = v
                elif k.startswith("min_"):
                    m = k[len("min_"):]
                    metrics.setdefault(m, {})['min'] = v
                elif k.startswith("max_"):
                    m = k[len("max_"):]
                    metrics.setdefault(m, {})['max'] = v
            for m in sorted(metrics.keys()):
                st = metrics[m]
                mu = st.get('mean', float('nan'))
                sd = st.get('std', float('nan'))
                mn = st.get('min', float('nan'))
                mx = st.get('max', float('nan'))
                lines.append(f"{m}: {mu:.4f} ± {sd:.4f} (min={mn:.4f}, max={mx:.4f})")
            # include final test if attached
            ft = {k: v for k, v in row.items() if k.startswith('final_test_')}
            if ft:
                lines.append("Final Test (lockbox)")
                # separate frame/video namespaces
                for k, v in sorted(ft.items()):
                    lines.append(f"{k}: {float(v):.4f}")
            # extras: convergence/time/gpu if present
            if 'avg_convergence_epoch' in row:
                lines.append(f"Avg convergence epoch: {float(row['avg_convergence_epoch']):.2f}")
            if 'total_train_seconds' in row:
                lines.append(f"Total train time: {_fmt_hms(float(row['total_train_seconds']))}")
            if 'mean_cuda_alloc_mb' in row and 'std_cuda_alloc_mb' in row:
                lines.append(f"GPU peak alloc: {float(row['mean_cuda_alloc_mb']):.1f} ± {float(row['std_cuda_alloc_mb']):.1f} MB")
            if 'mean_cuda_reserved_mb' in row and 'std_cuda_reserved_mb' in row:
                lines.append(f"GPU peak reserved: {float(row['mean_cuda_reserved_mb']):.1f} ± {float(row['std_cuda_reserved_mb']):.1f} MB")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    for backbone in backbones:
        for fusion in fusions:
            run_root = os.path.join(args.outroot, f"{backbone}_{fusion}")
            os.makedirs(run_root, exist_ok=True)
            print(f"\n=== Running CV for backbone={backbone} fusion={fusion} ===")

            fold_metrics = []

            for fold_name in fold_dirs:
                fold_idx = int(fold_name.split("_")[-1])
                fold_path = os.path.join(args.folds_root, fold_name)
                train_csv = os.path.join(fold_path, "train.csv")  # outer-train pool
                val_csv = os.path.join(fold_path, "val.csv")      # outer held-out validation

                # read outer-train CSV to determine patients used for fitting metadata stats (avoid leakage)
                train_df = pd.read_csv(train_csv)
                fit_patients = sorted(train_df["patient"].astype(int).unique().tolist())

                # outer-train base datasets (train tf for training subset; val tf for inner val subset)
                base_train_ds = CadicaDataset(
                    train_csv, metadata, get_train_tf(img_size),
                    selected_features=sel_nc,
                    text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    fit_patients=fit_patients,
                )
                # reuse train-fitted stats for other datasets in this fold
                _stats = base_train_ds.stats
                base_train_val_ds = CadicaDataset(
                    train_csv, metadata, get_val_tf(img_size),
                    selected_features=sel_nc,
                    text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    stats=_stats,
                )
                # outer held-out validation dataset
                outer_val_ds = CadicaDataset(
                    val_csv, metadata, get_val_tf(img_size),
                    selected_features=sel_nc,
                    text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    stats=_stats,
                )

                # inner split on patients from the outer-train CSV (stratified by patient-level label if possible)
                pl_tr = train_df.groupby("patient")["label"].max().reset_index().rename(columns={"label": "patient_label"})
                y_p = pl_tr["patient_label"].astype(int).values
                if len(np.unique(y_p)) >= 2:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=seed)
                    tr_pidx, va_pidx = next(sss.split(np.zeros(len(pl_tr)), y_p))
                    tr_pats = set(pl_tr.iloc[tr_pidx]["patient"].tolist())
                    va_pats = set(pl_tr.iloc[va_pidx]["patient"].tolist())
                else:
                    gss = GroupShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=seed)
                    tr_idx_tmp, va_idx_tmp = next(gss.split(train_df, groups=train_df["patient"]))
                    tr_pats = set(train_df.iloc[tr_idx_tmp]["patient"].unique().tolist())
                    va_pats = set(train_df.iloc[va_idx_tmp]["patient"].unique().tolist())

                inner_tr_idx = np.where(train_df["patient"].isin(tr_pats))[0]
                inner_va_idx = np.where(train_df["patient"].isin(va_pats))[0]

                # ---- Report dataset sizes and class balance before training ----
                def _counts(df: pd.DataFrame):
                    n = int(df.shape[0])
                    pos = int((df["label"] == 1).sum())
                    neg = int((df["label"] == 0).sum())
                    return n, pos, neg

                # Outer-train pool (full train_df)
                ot_n, ot_pos, ot_neg = _counts(train_df)
                # Inner splits derived from outer-train
                inner_tr_df = train_df.iloc[inner_tr_idx]
                inner_va_df = train_df.iloc[inner_va_idx]
                it_n, it_pos, it_neg = _counts(inner_tr_df)
                iv_n, iv_pos, iv_neg = _counts(inner_va_df)
                # Outer held-out validation
                val_df = pd.read_csv(val_csv)
                ov_n, ov_pos, ov_neg = _counts(val_df)

                print(f"[data] {fold_name} sizes:")
                print(f"  outer-train: n={ot_n}  pos={ot_pos}  neg={ot_neg}")
                print(f"  inner-train: n={it_n}  pos={it_pos}  neg={it_neg}")
                print(f"  inner-val  : n={iv_n}  pos={iv_pos}  neg={iv_neg}")
                print(f"  outer-val  : n={ov_n}  pos={ov_pos}  neg={ov_neg}")

                from torch.utils.data import Subset
                # Optional on-disk augmentation for inner-train only
                def _try_import_albu():
                    try:
                        import albumentations as A
                        return A
                    except Exception:
                        return None

                def _build_aug_pipe(img_size=None):
                    A = _try_import_albu()
                    if A is None:
                        return None
                    ops = [
                        A.HorizontalFlip(p=0.5),
                        A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-10,10), p=0.7),
                        A.RandomBrightnessContrast(0.15,0.15,p=0.5),
                        A.MotionBlur(blur_limit=3,p=0.2),
                    ]
                    try:
                        import inspect
                        params = inspect.signature(A.GaussNoise.__init__).parameters
                        if 'var_limit' in params:
                            ops.append(A.GaussNoise(var_limit=(10.0,50.0), p=0.3))
                        else:
                            ops.append(A.GaussNoise(p=0.3))
                    except Exception:
                        try:
                            ops.append(A.GaussNoise(p=0.3))
                        except Exception:
                            pass
                    ops.append(A.CLAHE(clip_limit=(2,4), p=0.2))
                    if img_size is not None:
                        ops.insert(0, A.LongestMaxSize(max_size=img_size))
                    return A.Compose(ops)

                def _apply_aug(img3, aug):
                    if aug is None:
                        # minimal fallback: h-flip or mild blur
                        return cv2.flip(img3, 1) if random.random() < 0.5 else cv2.GaussianBlur(img3, (3,3), 0)
                    out = aug(image=img3)
                    return out['image']

                # Build default inner-train dataset (no aug)
                inner_tr_ds = Subset(base_train_ds, inner_tr_idx)
                inner_va_ds = Subset(base_train_val_ds, inner_va_idx)

                # If augmentation is requested, synthesize extra rows/images and rebuild inner_tr_ds
                if args.augment:
                    fold_out = os.path.join(run_root, fold_name)
                    os.makedirs(fold_out, exist_ok=True)

                    inner_tr_df = train_df.iloc[inner_tr_idx].copy()
                    pos_ct = int((inner_tr_df['label'] == 1).sum())
                    neg_ct = int((inner_tr_df['label'] == 0).sum())
                    tgt_label = int(args.aug_label)
                    orig_ct = pos_ct if tgt_label == 1 else neg_ct
                    other_ct = neg_ct if tgt_label == 1 else pos_ct
                    if args.aug_match_count:
                        extra_needed = max(0, other_ct - orig_ct)
                    else:
                        factor = max(1.0, float(args.aug_factor))
                        extra_needed = int(max(0, round(orig_ct * (factor - 1.0))))

                    if extra_needed > 0 and orig_ct > 0:
                        aug = _build_aug_pipe(args.aug_img_size)
                        # Candidate rows of chosen label
                        cand = inner_tr_df[inner_tr_df['label'] == tgt_label].copy()
                        used_count = {}
                        produced = 0
                        max_per = max(0, int(args.aug_max_per_frame))
                        capacity = len(cand) * max_per
                        to_add = min(extra_needed, capacity)

                        # Precompute next frame id function per (p,v)
                        def _next_frame_id(p, v):
                            # max existing in inner set plus any produced so far for this (p,v)
                            sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                            mx = int(sub['frame'].max()) if len(sub) else 0
                            # also consider augmented ones we added
                            key = (p, v, '__max_aug__')
                            mx2 = used_count.get(key, 0)
                            return mx + mx2 + 1

                        aug_rows = []
                        # Rounds to ensure every frame used before reuse
                        for r in range(max_per):
                            if produced >= to_add:
                                break
                            cand_shuf = cand.sample(frac=1.0, random_state=random.randint(0, 1_000_000)).itertuples(index=False)
                            for row in cand_shuf:
                                if produced >= to_add:
                                    break
                                p = int(row.patient); v = int(row.video); k = int(row.frame)
                                ck = (p, v, k)
                                if used_count.get(ck, 0) >= max_per:
                                    continue
                                img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    print(f"[warn] cannot read {row.img_path}; skipping")
                                    continue
                                img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                aug_img = _apply_aug(img3, aug)
                                aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                new_k = _next_frame_id(p, v)
                                # update max tracker for (p,v)
                                used_count[(p, v, '__max_aug__')] = used_count.get((p, v, '__max_aug__'), 0) + 1
                                out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                os.makedirs(out_dir, exist_ok=True)
                                out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                out_path = os.path.join(out_dir, out_name)
                                ok = cv2.imwrite(out_path, aug_gray)
                                if not ok:
                                    print(f"[warn] failed to write {out_path}; skipping")
                                    continue
                                aug_rows.append({
                                    'patient': p,
                                    'video': v,
                                    'frame': new_k,
                                    'img_path': out_path.replace('\\','/'),
                                    'label': tgt_label,
                                })
                                used_count[ck] = used_count.get(ck, 0) + 1
                                produced += 1

                        if produced > 0:
                            inner_aug_df = pd.concat([inner_tr_df, pd.DataFrame(aug_rows)], ignore_index=True)
                            # Write temp CSV for this fold
                            aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                            inner_aug_df.to_csv(aug_csv, index=False)
                            # Rebuild inner-train dataset from augmented CSV using train transforms and existing stats
                            inner_tr_ds = CadicaDataset(
                                aug_csv, metadata, get_train_tf(img_size),
                                selected_features=sel_nc,
                                text_features=text_cols,
                                text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                stats=_stats,
                            )
                            # Print counts
                            new_pos = int((inner_aug_df['label'] == 1).sum())
                            new_neg = int((inner_aug_df['label'] == 0).sum())
                            print(f"[augment] {fold_name} inner-train: +{produced} label={tgt_label} -> pos={new_pos} neg={new_neg} (was pos={pos_ct} neg={neg_ct})")

                tr_loader = DataLoader(
                    inner_tr_ds,
                    batch_size=cfg.get("batch_size", 64),
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                inner_va_loader = DataLoader(
                    inner_va_ds,
                    batch_size=cfg.get("batch_size", 64),
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                outer_va_loader = DataLoader(
                    outer_val_ds,
                    batch_size=cfg.get("batch_size", 64),
                    shuffle=False,
                    num_workers=args.num_workers,
                )

                # model(s)
                img_model, clin_mlp, late_head, params = build_models(
                    backbone=backbone,
                    fusion=fusion,
                    tab_dim=base_train_ds.tab_dim,
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

                # training loop (select on inner validation)
                grad_accum_steps = 1
                use_amp_flag = (args.amp == "on")
                best = -1.0
                best_metrics = None  # inner-val best metrics
                best_inner_frame = {"y": None, "p": None}
                best_inner_video_df = None
                best_epoch = None
                best_state = {"img": None, "clin": None, "late": None}
                fold_out = os.path.join(run_root, fold_name)
                os.makedirs(fold_out, exist_ok=True)
                ckpt_path = os.path.join(fold_out, "best.pt")
                epochs_no_improve = 0
                # timing + GPU peak memory tracking
                t0 = time.time()
                if device.startswith("cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                for epoch in range(1, epochs + 1):
                    print(f"\n{backbone}/{fusion} | {fold_name} | Epoch {epoch}/{epochs}")
                    train_one_epoch(
                        img_model, optimizer, tr_loader, device,
                        mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                        pim_lambdas=None,
                        pim_update_freq=grad_accum_steps, use_amp=use_amp_flag,
                    )
                    frame_metrics, y_true_f, y_prob_f, video_metrics, video_df = evaluate(
                        img_model, inner_va_loader, device,
                        mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                        also_video=True, video_reduce=args.video_reduce,
                    )
                    print("[inner_val/frame]", {k: round(float(v), 4) for k, v in frame_metrics.items()})
                    print("[inner_val/video]", {k: round(float(v), 4) for k, v in video_metrics.items() if k != "_multi"})
                    if args.eval_outer_each_epoch:
                        ofm, _, _, ovm, _ = evaluate(
                            img_model, outer_va_loader, device,
                            mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                            also_video=True, video_reduce=args.video_reduce,
                        )
                        print("[outer_val-each-epoch/frame]", {k: round(float(v), 4) for k, v in ofm.items()})
                        print("[outer_val-each-epoch/video]", {k: round(float(v), 4) for k, v in ovm.items() if k != "_multi"})

                    # selection score based on requested domain/metric
                    sel_dict = frame_metrics if args.select_domain == "frame" else {k: v for k, v in video_metrics.items() if k != "_multi"}
                    score = float(sel_dict.get(args.select_metric, 0.0)) if args.select_metric in sel_dict else 0.0
                    if score > best:
                        best = score
                        best_metrics = {
                            "frame": {k: float(v) for k, v in frame_metrics.items()},
                            "video": {k: float(v) for k, v in video_metrics.items() if k != "_multi"},
                        }
                        best_epoch = int(epoch)
                        # keep best weights in-memory
                        best_state["img"] = deepcopy(img_model.state_dict())
                        best_state["clin"] = deepcopy(clin_mlp.state_dict()) if clin_mlp else None
                        best_state["late"] = deepcopy(late_head.state_dict()) if late_head else None
                        if args.save_checkpoints:
                            torch.save(
                                {
                                    "cfg": cfg,
                                    "fusion": fusion,
                                    "backbone": backbone,
                                    "tab_dim": int(base_train_ds.tab_dim),
                                    "model": img_model.state_dict(),
                                    "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
                                    "late_head": late_head.state_dict() if late_head else None,
                                    "best_val": best_metrics,
                                    "best_epoch": best_epoch,
                                    "select_domain": args.select_domain,
                                    "select_metric": args.select_metric,
                                },
                                ckpt_path,
                            )
                        # capture inner-val predictions at best epoch for threshold tuning
                        best_inner_frame = {"y": y_true_f, "p": y_prob_f}
                        best_inner_video_df = video_df.copy()
                        # optionally persist inner validation probs for traceability
                        if args.save_val_probs:
                            video_df.to_csv(os.path.join(fold_out, "inner_val_video_probs.csv"), index=False)
                        epochs_no_improve = 0
                        print(f"[ckpt] new best saved: {args.select_domain}/{args.select_metric}={best:.4f}")
                    else:
                        epochs_no_improve += 1
                        print(f"[early-stopping] no improvement for {epochs_no_improve} epochs")

                    if patience and epochs_no_improve >= patience:
                        print(f"[early-stopping] stopping at epoch {epoch}")
                        break

                # training duration for this fold
                fold_train_seconds = time.time() - t0

                # restore best weights and evaluate on outer held-out validation
                if best_state["img"] is not None:
                    img_model.load_state_dict(best_state["img"])
                    if clin_mlp and best_state["clin"] is not None:
                        clin_mlp.load_state_dict(best_state["clin"])
                    if late_head and best_state["late"] is not None:
                        late_head.load_state_dict(best_state["late"])

                outer_frame_metrics, yf_outer, pf_outer, outer_video_metrics, outer_video_df = evaluate(
                    img_model, outer_va_loader, device,
                    mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                    also_video=True, video_reduce=args.video_reduce,
                )
                print("[outer_val/frame]", {k: round(float(v), 4) for k, v in outer_frame_metrics.items()})
                print("[outer_val/video]", {k: round(float(v), 4) for k, v in outer_video_metrics.items() if k != "_multi"})
                if args.save_val_probs:
                    outer_video_df.to_csv(os.path.join(fold_out, "outer_val_video_probs.csv"), index=False)

                # GPU peak memory stats for this fold
                gpu_name = "cpu"
                alloc_mb = None
                reserved_mb = None
                if device.startswith("cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        dev_idx = torch.cuda.current_device()
                        gpu_name = torch.cuda.get_device_name(dev_idx)
                        alloc_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024.0**2)
                        reserved_mb = torch.cuda.max_memory_reserved(dev_idx) / (1024.0**2)
                    except Exception:
                        pass

                # threshold tuning using inner-val predictions from best epoch
                def _best_thr_from_arrays(y_true, y_prob, metric_key: str) -> float:
                    if metric_key == "auc":
                        return 0.5
                    uniq = np.unique(y_prob)
                    if uniq.size > 200:
                        cand = np.linspace(0.0, 1.0, num=201)
                    else:
                        cand = np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0)
                    best_t, best_s = 0.5, -1.0
                    for t in cand:
                        m = metrics_from_probs(y_true, y_prob, thr=float(t)).get(metric_key, float("nan"))
                        s = float(m) if np.isfinite(m) else -1.0
                        if s > best_s:
                            best_s, best_t = s, float(t)
                    return best_t

                thr_frame = 0.5
                thr_video = 0.5
                if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                    thr_frame = _best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.select_metric)
                if best_inner_video_df is not None:
                    yv_in = best_inner_video_df["y_true"].values
                    pv_in = best_inner_video_df["y_prob"].values
                    thr_video = _best_thr_from_arrays(yv_in, pv_in, args.select_metric)

                tuned_frame_metrics = metrics_from_probs(yf_outer, pf_outer, thr=thr_frame)
                yv = outer_video_df["y_true"].values
                pv = outer_video_df["y_prob"].values
                tuned_video_metrics = metrics_from_probs(yv, pv, thr=thr_video)

                # record fold best (outer evaluation used for aggregation)
                row = {
                    "backbone": backbone,
                    "fusion": fusion,
                    "fold": fold_idx,
                    "best_epoch": int(best_epoch) if best_epoch is not None else None,
                    "convergence_epoch": int(best_epoch) if best_epoch is not None else None,
                    "train_seconds": float(fold_train_seconds),
                    "device": str(device),
                    "gpu_name": str(gpu_name),
                    "cuda_mem_alloc_peak_mb": float(alloc_mb) if alloc_mb is not None else None,
                    "cuda_mem_reserved_peak_mb": float(reserved_mb) if reserved_mb is not None else None,
                    **{f"val_frame_{k}": float(tuned_frame_metrics[k]) for k in tuned_frame_metrics},
                    **{f"val_video_{k}": float(tuned_video_metrics[k]) for k in tuned_video_metrics},
                }
                mv = outer_video_metrics.get("_multi") if isinstance(outer_video_metrics, dict) else None
                if isinstance(mv, dict):
                    for agg_name, md in mv.items():
                        for mk, mvv in md.items():
                            row[f"val_video_{agg_name}_{mk}"] = float(mvv)
                fold_metrics.append(row)
                # also dump per-fold json
                fold_report = dict(row)
                fold_report.update({
                    "selection": {"domain": args.select_domain, "metric": args.select_metric},
                    "inner_best": best_metrics,
                    "thresholds": {"frame": float(thr_frame), "video": float(thr_video)},
                })
                with open(os.path.join(fold_out, "val_best.json"), "w") as f:
                    json.dump(fold_report, f, indent=2)

                # update per-model summary text incrementally
                try:
                    df_so_far = pd.DataFrame(fold_metrics)
                    _write_model_summary_txt(os.path.join(run_root, "summary.txt"), backbone, fusion, df_so_far)
                except Exception as _e:
                    pass

            # aggregate across folds
            df = pd.DataFrame(fold_metrics)
            df.to_csv(os.path.join(run_root, "cv_val_folds.csv"), index=False)
            # quick summary
            for col in [c for c in df.columns if c.startswith("val_")]:
                mu, sd = float(df[col].mean()), float(df[col].std())
                print(f"[summary] {backbone}/{fusion} {col}: {mu:.4f} ± {sd:.4f}")
            # optional final retraining on full non-test pool and evaluate on pristine test
            final_test_metrics = None
            if args.final_test_eval:
                comb_csv = args.trainval_pool_csv
                if not os.path.exists(comb_csv):
                    print(f"[warn] trainval pool CSV not found: {comb_csv}. Skipping final test eval.")
                elif not os.path.exists(args.test_csv):
                    print(f"[warn] pristine test CSV not found: {args.test_csv}. Skipping final test eval.")
                else:
                    tf_train = get_train_tf(img_size)
                    full_train_ds = CadicaDataset(
                        comb_csv, metadata, tf_train,
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )
                    test_ds = CadicaDataset(
                        args.test_csv, metadata, get_val_tf(img_size),
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )

                    tr_full = DataLoader(full_train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    te = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=args.num_workers)

                    # fresh model
                    img_model_f, clin_mlp_f, late_head_f, params_f = build_models(
                        backbone=backbone, fusion=fusion, tab_dim=full_train_ds.tab_dim, pat_channels=cfg.get("pat_channels", 4), device=device,
                    )
                    # optimizer
                    if opt_name in ("adamw", "adam"):
                        optimizer_f = torch.optim.AdamW(params_f, lr=lr, weight_decay=wd)
                    elif opt_name in ("sgd", "sgdm"):
                        optimizer_f = torch.optim.SGD(params_f, lr=lr, momentum=0.9, weight_decay=wd)
                    elif opt_name == "rmsprop":
                        optimizer_f = torch.optim.RMSprop(params_f, lr=lr, weight_decay=wd)
                    else:
                        optimizer_f = torch.optim.AdamW(params_f, lr=lr, weight_decay=wd)

                    # train full schedule
                    for epoch in range(1, epochs + 1):
                        print(f"[final-train] {backbone}/{fusion} | Epoch {epoch}/{epochs}")
                        train_one_epoch(
                            img_model_f, optimizer_f, tr_full, device,
                            mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f,
                        )

                    # evaluate on pristine test
                    fmet, _, _, vmet, vdf = evaluate(
                        img_model_f, te, device,
                        mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f,
                        also_video=True, video_reduce=args.video_reduce,
                    )
                    final_test_metrics = {
                        "frame": {k: float(v) for k, v in fmet.items()},
                        "video": {k: float(v) for k, v in vmet.items()},
                    }
                    print("[final-test/frame]", {k: round(float(v), 4) for k, v in fmet.items()})
                    print("[final-test/video]", {k: round(float(v), 4) for k, v in vmet.items()})
                    # persist
                    with open(os.path.join(run_root, "final_test_metrics.json"), "w") as f:
                        json.dump(final_test_metrics, f, indent=2)
                    # optionally save final checkpoint
                    if args.save_checkpoints:
                        torch.save(
                            {
                                "cfg": cfg,
                                "fusion": fusion,
                                "backbone": backbone,
                                "tab_dim": int(full_train_ds.tab_dim),
                                "model": img_model_f.state_dict(),
                                "clin_mlp": clin_mlp_f.state_dict() if clin_mlp_f else None,
                                "late_head": late_head_f.state_dict() if late_head_f else None,
                            },
                            os.path.join(run_root, "final_full_train.pt"),
                        )

            # write final per-model text summary (include final test if available)
            _write_model_summary_txt(os.path.join(run_root, "summary.txt"), backbone, fusion, df, final_test=final_test_metrics)
            # add summary rows
            summary_row = {"backbone": backbone, "fusion": fusion}
            for col in [c for c in df.columns if c.startswith("val_")]:
                summary_row[f"mean_{col}"] = float(df[col].mean())
                summary_row[f"std_{col}"] = float(df[col].std())
                summary_row[f"min_{col}"] = float(df[col].min())
                summary_row[f"max_{col}"] = float(df[col].max())
            # extras for global summary
            if "convergence_epoch" in df.columns:
                try:
                    summary_row["avg_convergence_epoch"] = float(pd.to_numeric(df["convergence_epoch"], errors="coerce").mean())
                except Exception:
                    pass
            if "train_seconds" in df.columns:
                try:
                    summary_row["total_train_seconds"] = float(pd.to_numeric(df["train_seconds"], errors="coerce").sum())
                    summary_row["mean_train_seconds"] = float(pd.to_numeric(df["train_seconds"], errors="coerce").mean())
                except Exception:
                    pass
            if "cuda_mem_alloc_peak_mb" in df.columns:
                try:
                    summary_row["mean_cuda_alloc_mb"] = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").mean())
                    summary_row["std_cuda_alloc_mb"] = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").std())
                except Exception:
                    pass
            if "cuda_mem_reserved_peak_mb" in df.columns:
                try:
                    summary_row["mean_cuda_reserved_mb"] = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").mean())
                    summary_row["std_cuda_reserved_mb"] = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").std())
                except Exception:
                    pass
            if final_test_metrics is not None:
                for dom in ("frame", "video"):
                    for k, v in final_test_metrics[dom].items():
                        summary_row[f"final_test_{dom}_{k}"] = float(v)
            summary_rows.append(summary_row)
            # update global text summary after each model
            try:
                _write_global_summary_txt(os.path.join("outputs", "cv_models_summary.txt"), summary_rows)
            except Exception:
                pass

    # global summary across all runs
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        out_csv = os.path.join(args.outroot, "cv_summary.csv")
        sdf.to_csv(out_csv, index=False)
        print(f"\n[done] wrote global summary -> {out_csv}")


if __name__ == "__main__":
    main()
