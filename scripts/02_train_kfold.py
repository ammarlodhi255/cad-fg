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
from sklearn.metrics import roc_curve, precision_recall_curve
try:
    from sklearn.calibration import calibration_curve
except Exception:
    calibration_curve = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_train_tf_basic, get_val_tf
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
        # YOLO Late-FiLM variants (custom head)
        if backbone in ("yolov8n_film_late", "yolov8s_film_late", "yolov8m_film_late", "yolov8l_film_late", "yolov8x_film_late"):
            from src.models.yolov8_cls_backbone import YOLOv8ClsBackbone
            from src.models.late_fusion import TabularMLP
            from src.models.yolo_early_late_fusion_variants import LateFiLMHead
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8ClsBackbone(model_name=model_name, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
            clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
            head = LateFiLMHead(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
            params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
            return img_model, clin_mlp, head, params
        if backbone in ("yolo11n_film_late", "yolo11s_film_late", "yolo11m_film_late", "yolo11l_film_late", "yolo11x_film_late"):
            from src.models.yolo11_cls_backbone import YOLO11ClsBackbone
            from src.models.late_fusion import TabularMLP
            from src.models.yolo_early_late_fusion_variants import LateFiLMHead
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11ClsBackbone(model_name=model_name, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
            clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
            head = LateFiLMHead(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
            params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
            return img_model, clin_mlp, head, params
        if backbone in ("yolov5n_film_late", "yolov5s_film_late", "yolov5m_film_late", "yolov5l_film_late", "yolov5x_film_late"):
            from src.models.yolov5_cls_backbone import YOLOv5ClsBackbone
            from src.models.late_fusion import TabularMLP
            from src.models.yolo_early_late_fusion_variants import LateFiLMHead
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5ClsBackbone(model_name=model_name, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
            clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
            head = LateFiLMHead(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
            params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
            return img_model, clin_mlp, head, params
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
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone
            img_model = MobileNetV2Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone
            img_model = DenseNet121Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone
            img_model = AlexNetBackbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone in ("vgg19", "vgg-19"):
            from src.models.vgg_backbone import VGG19Backbone
            img_model = VGG19Backbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone
            img_model = ViTBackbone(pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone.startswith("regnet_"):
            from src.models.regnet_backbone import RegNetBackbone
            img_model = RegNetBackbone(variant=backbone, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone.startswith("repvgg"):
            from src.models.repvgg_backbone import RepVGGBackbone
            img_model = RepVGGBackbone(model_name=backbone, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone in (
            "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
            "yolov8n_cls", "yolov8s_cls", "yolov8m_cls", "yolov8l_cls", "yolov8x_cls",
        ):
            from src.models.yolov8_cls_backbone import YOLOv8ClsBackbone
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv8ClsBackbone(model_name=model_name, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone in (
            "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
            "yolo11n_cls", "yolo11s_cls", "yolo11m_cls", "yolo11l_cls", "yolo11x_cls",
        ):
            from src.models.yolo11_cls_backbone import YOLO11ClsBackbone
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLO11ClsBackbone(model_name=model_name, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        elif backbone in (
            "yolov5n-cls", "yolov5s-cls", "yolov5m-cls", "yolov5l-cls", "yolov5x-cls",
            "yolov5n_cls", "yolov5s_cls", "yolov5m_cls", "yolov5l_cls", "yolov5x_cls",
        ):
            from src.models.yolov5_cls_backbone import YOLOv5ClsBackbone
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv5ClsBackbone(model_name=model_name, pretrained=True).to(device); img_feat_dim = img_model.feat_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
        return img_model, clin_mlp, head, params

    if fusion == "early":

        # YOLO Early-FiLM-at-stem variants (no channels added)
        if backbone in ("yolov8n_film_early", "yolov8s_film_early", "yolov8m_film_early", "yolov8l_film_early", "yolov8x_film_early"):
            from src.models.yolo_early_late_fusion_variants import YOLOv8EarlyFiLMStem
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8EarlyFiLMStem(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
            params = list(img_model.parameters())
            return img_model, None, None, params
        if backbone in ("yolo11n_film_early", "yolo11s_film_early", "yolo11m_film_early", "yolo11l_film_early", "yolo11x_film_early"):
            from src.models.yolo_early_late_fusion_variants import YOLO11EarlyFiLMStem
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11EarlyFiLMStem(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
            params = list(img_model.parameters())
            return img_model, None, None, params
        if backbone in ("yolov5n_film_early", "yolov5s_film_early", "yolov5m_film_early", "yolov5l_film_early", "yolov5x_film_early"):
            from src.models.yolo_early_late_fusion_variants import YOLOv5EarlyFiLMStem
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5EarlyFiLMStem(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
            params = list(img_model.parameters())
            return img_model, None, None, params

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
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2EarlyFusionPatient
            img_model = MobileNetV2EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121EarlyFusionPatient
            img_model = DenseNet121EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetEarlyFusionPatient
            img_model = AlexNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone in ("vgg19", "vgg-19"):
            from src.models.vgg_backbone import VGG19EarlyFusionPatient
            img_model = VGG19EarlyFusionPatient(tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
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
        # ConvNeXt (tiny/base) fusion variants
        elif backbone == "convnext_tiny_film":
            from src.models.convnext_early_fusion_variants_tiny import ConvNeXtTinyEarlyFusionFiLM
            img_model = ConvNeXtTinyEarlyFusionFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base_film":
            from src.models.convnext_early_fusion_variants_base import ConvNeXtBaseEarlyFusionFiLM
            img_model = ConvNeXtBaseEarlyFusionFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny_prompt":
            from src.models.convnext_early_fusion_variants_tiny import ConvNeXtTinyEarlyFusionPrompt
            img_model = ConvNeXtTinyEarlyFusionPrompt(tab_dim=tab_dim, K=4, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base_prompt":
            from src.models.convnext_early_fusion_variants_base import ConvNeXtBaseEarlyFusionPrompt
            img_model = ConvNeXtBaseEarlyFusionPrompt(tab_dim=tab_dim, K=4, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny_film_deep":
            from src.models.convnext_mid_fusion_variants_tiny import ConvNeXtTinyMidFusionResidualFiLM
            img_model = ConvNeXtTinyMidFusionResidualFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base_film_deep":
            from src.models.convnext_mid_fusion_variants_base import ConvNeXtBaseMidFusionResidualFiLM
            img_model = ConvNeXtBaseMidFusionResidualFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny_cls_xattn":
            from src.models.convnext_mid_fusion_variants_tiny import ConvNeXtTinyMidFusionCLSCrossAttn
            img_model = ConvNeXtTinyMidFusionCLSCrossAttn(tab_dim=tab_dim, nheads=8, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base_cls_xattn":
            from src.models.convnext_mid_fusion_variants_base import ConvNeXtBaseMidFusionCLSCrossAttn
            img_model = ConvNeXtBaseMidFusionCLSCrossAttn(tab_dim=tab_dim, nheads=8, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_tiny_meta_adapter":
            from src.models.convnext_mid_fusion_variants_tiny import ConvNeXtTinyMidFusionMetaAdapter
            img_model = ConvNeXtTinyMidFusionMetaAdapter(tab_dim=tab_dim, r=64, pretrained=True, out_dim=2).to(device)
        elif backbone == "convnext_base_meta_adapter":
            from src.models.convnext_mid_fusion_variants_base import ConvNeXtBaseMidFusionMetaAdapter
            img_model = ConvNeXtBaseMidFusionMetaAdapter(tab_dim=tab_dim, r=64, pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("regnet_"):
            from src.models.regnet_backbone import RegNetEarlyFusionPatient
            img_model = RegNetEarlyFusionPatient(variant=backbone, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone.startswith("repvgg"):
            from src.models.repvgg_backbone import RepVGGEarlyFusionPatient
            img_model = RepVGGEarlyFusionPatient(model_name=backbone, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone in (
            "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
            "yolov8n_cls", "yolov8s_cls", "yolov8m_cls", "yolov8l_cls", "yolov8x_cls",
        ):
            from src.models.yolov8_cls_backbone import YOLOv8EarlyFusionPatient
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv8EarlyFusionPatient(model_name=model_name, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone in (
            "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
            "yolo11n_cls", "yolo11s_cls", "yolo11m_cls", "yolo11l_cls", "yolo11x_cls",
        ):
            from src.models.yolo11_cls_backbone import YOLO11EarlyFusionPatient
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLO11EarlyFusionPatient(model_name=model_name, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        elif backbone in (
            "yolov5n-cls", "yolov5s-cls", "yolov5m-cls", "yolov5l-cls", "yolov5x-cls",
            "yolov5n_cls", "yolov5s_cls", "yolov5m_cls", "yolov5l_cls", "yolov5x_cls",
        ):
            from src.models.yolov5_cls_backbone import YOLOv5EarlyFusionPatient
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv5EarlyFusionPatient(model_name=model_name, tab_dim=tab_dim, c_pat=pat_channels, pretrained=True, out_dim=2).to(device)
        # YOLOv8 mid-fusion variants
        elif backbone in ("yolov8n_film_mid", "yolov8s_film_mid", "yolov8m_film_mid", "yolov8l_film_mid", "yolov8x_film_mid"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionResidualFiLM
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8MidFusionResidualFiLM(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n_gate_mid", "yolov8s_gate_mid", "yolov8m_gate_mid", "yolov8l_gate_mid", "yolov8x_gate_mid"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionSEGate
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8MidFusionSEGate(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n_cls_xattn_mid", "yolov8s_cls_xattn_mid", "yolov8m_cls_xattn_mid", "yolov8l_cls_xattn_mid", "yolov8x_cls_xattn_mid"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionCLSCrossAttn
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8MidFusionCLSCrossAttn(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n_meta_adapter", "yolov8s_meta_adapter", "yolov8m_meta_adapter", "yolov8l_meta_adapter", "yolov8x_meta_adapter"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionMetaAdapter
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8MidFusionMetaAdapter(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n_spatial_mid", "yolov8s_spatial_mid", "yolov8m_spatial_mid", "yolov8l_spatial_mid", "yolov8x_spatial_mid"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionSpatialGate
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8MidFusionSpatialGate(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n_hyper1x1_mid", "yolov8s_hyper1x1_mid", "yolov8m_hyper1x1_mid", "yolov8l_hyper1x1_mid", "yolov8x_hyper1x1_mid"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionHyper1x1
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv8MidFusionHyper1x1(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov8n_film_multi2_mid", "yolov8s_film_multi2_mid", "yolov8m_film_multi2_mid", "yolov8l_film_multi2_mid", "yolov8x_film_multi2_mid",
                          "yolov8n_film_multi3_mid", "yolov8s_film_multi3_mid", "yolov8m_film_multi3_mid", "yolov8l_film_multi3_mid", "yolov8x_film_multi3_mid"):
            from src.models.yolov8_mid_fusion_variants import YOLOv8MidFusionMultiSiteFiLM
            model_name = backbone.split('_')[0] + "-cls"
            nsites = 3 if "multi3" in backbone else 2
            img_model = YOLOv8MidFusionMultiSiteFiLM(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2, nsites=nsites).to(device)
        elif backbone in ("yolo11n_film_mid", "yolo11s_film_mid", "yolo11m_film_mid", "yolo11l_film_mid", "yolo11x_film_mid"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionResidualFiLM
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11MidFusionResidualFiLM(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n_gate_mid", "yolo11s_gate_mid", "yolo11m_gate_mid", "yolo11l_gate_mid", "yolo11x_gate_mid"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionSEGate
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11MidFusionSEGate(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n_cls_xattn_mid", "yolo11s_cls_xattn_mid", "yolo11m_cls_xattn_mid", "yolo11l_cls_xattn_mid", "yolo11x_cls_xattn_mid"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionCLSCrossAttn
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11MidFusionCLSCrossAttn(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n_meta_adapter", "yolo11s_meta_adapter", "yolo11m_meta_adapter", "yolo11l_meta_adapter", "yolo11x_meta_adapter"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionMetaAdapter
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11MidFusionMetaAdapter(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n_spatial_mid", "yolo11s_spatial_mid", "yolo11m_spatial_mid", "yolo11l_spatial_mid", "yolo11x_spatial_mid"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionSpatialGate
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11MidFusionSpatialGate(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n_hyper1x1_mid", "yolo11s_hyper1x1_mid", "yolo11m_hyper1x1_mid", "yolo11l_hyper1x1_mid", "yolo11x_hyper1x1_mid"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionHyper1x1
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLO11MidFusionHyper1x1(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n_film_multi2_mid", "yolo11s_film_multi2_mid", "yolo11m_film_multi2_mid", "yolo11l_film_multi2_mid", "yolo11x_film_multi2_mid",
                          "yolo11n_film_multi3_mid", "yolo11s_film_multi3_mid", "yolo11m_film_multi3_mid", "yolo11l_film_multi3_mid", "yolo11x_film_multi3_mid"):
            from src.models.yolo11_mid_fusion_variants import YOLO11MidFusionMultiSiteFiLM
            model_name = backbone.split('_')[0] + "-cls"
            nsites = 3 if "multi3" in backbone else 2
            img_model = YOLO11MidFusionMultiSiteFiLM(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2, nsites=nsites).to(device)
        # YOLOv5 mid-fusion variants
        elif backbone in ("yolov5n_film_mid", "yolov5s_film_mid", "yolov5m_film_mid", "yolov5l_film_mid", "yolov5x_film_mid"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionResidualFiLM
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5MidFusionResidualFiLM(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n_gate_mid", "yolov5s_gate_mid", "yolov5m_gate_mid", "yolov5l_gate_mid", "yolov5x_gate_mid"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionSEGate
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5MidFusionSEGate(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n_cls_xattn_mid", "yolov5s_cls_xattn_mid", "yolov5m_cls_xattn_mid", "yolov5l_cls_xattn_mid", "yolov5x_cls_xattn_mid"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionCLSCrossAttn
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5MidFusionCLSCrossAttn(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n_meta_adapter", "yolov5s_meta_adapter", "yolov5m_meta_adapter", "yolov5l_meta_adapter", "yolov5x_meta_adapter"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionMetaAdapter
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5MidFusionMetaAdapter(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n_spatial_mid", "yolov5s_spatial_mid", "yolov5m_spatial_mid", "yolov5l_spatial_mid", "yolov5x_spatial_mid"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionSpatialGate
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5MidFusionSpatialGate(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n_hyper1x1_mid", "yolov5s_hyper1x1_mid", "yolov5m_hyper1x1_mid", "yolov5l_hyper1x1_mid", "yolov5x_hyper1x1_mid"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionHyper1x1
            model_name = backbone.split('_')[0] + "-cls"
            img_model = YOLOv5MidFusionHyper1x1(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n_film_multi2_mid", "yolov5s_film_multi2_mid", "yolov5m_film_multi2_mid", "yolov5l_film_multi2_mid", "yolov5x_film_multi2_mid",
                          "yolov5n_film_multi3_mid", "yolov5s_film_multi3_mid", "yolov5m_film_multi3_mid", "yolov5l_film_multi3_mid", "yolov5x_film_multi3_mid"):
            from src.models.yolov5_mid_fusion_variants import YOLOv5MidFusionMultiSiteFiLM
            model_name = backbone.split('_')[0] + "-cls"
            nsites = 3 if "multi3" in backbone else 2
            img_model = YOLOv5MidFusionMultiSiteFiLM(model_name=model_name, tab_dim=tab_dim, pretrained=True, out_dim=2, nsites=nsites).to(device)
        # New: YOLOv8/YOLOv5 mid+late dual fusion
        elif backbone in ("yolov8n_film_dual", "yolov8s_film_dual", "yolov8m_film_dual", "yolov8l_film_dual", "yolov8x_film_dual"):
            from src.models.midlate_fusion_variants import YOLOv8MidLateFiLM
            model_name = backbone.replace('_dual','').replace('_film','').split('_')[0] + "-cls"
            img_model = YOLOv8MidLateFiLM(model_name=model_name, tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone in ("yolov8n_gate_dual", "yolov8s_gate_dual", "yolov8m_gate_dual", "yolov8l_gate_dual", "yolov8x_gate_dual"):
            from src.models.midlate_fusion_variants import YOLOv8MidLateSEGate
            model_name = backbone.replace('_dual','').replace('_gate','').split('_')[0] + "-cls"
            img_model = YOLOv8MidLateSEGate(model_name=model_name, tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone in ("yolov5n_film_dual", "yolov5s_film_dual", "yolov5m_film_dual", "yolov5l_film_dual", "yolov5x_film_dual"):
            from src.models.midlate_fusion_variants import YOLOv5MidLateFiLM
            model_name = backbone.replace('_dual','').replace('_film','').split('_')[0] + "-cls"
            img_model = YOLOv5MidLateFiLM(model_name=model_name, tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone in ("yolov5n_gate_dual", "yolov5s_gate_dual", "yolov5m_gate_dual", "yolov5l_gate_dual", "yolov5x_gate_dual"):
            from src.models.midlate_fusion_variants import YOLOv5MidLateSEGate
            model_name = backbone.replace('_dual','').replace('_gate','').split('_')[0] + "-cls"
            img_model = YOLOv5MidLateSEGate(model_name=model_name, tab_dim=tab_dim, out_dim=2).to(device)
        # New: ConvNeXt Tiny/Base mid+late dual fusion
        elif backbone == "convnext_tiny_film_dual":
            from src.models.midlate_fusion_variants import ConvNeXtTinyMidLateFiLM
            img_model = ConvNeXtTinyMidLateFiLM(tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone == "convnext_base_film_dual":
            from src.models.midlate_fusion_variants import ConvNeXtBaseMidLateFiLM
            img_model = ConvNeXtBaseMidLateFiLM(tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone == "convnext_tiny_gate_dual":
            from src.models.midlate_fusion_variants import ConvNeXtTinyMidLateGate
            img_model = ConvNeXtTinyMidLateGate(tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone == "convnext_base_gate_dual":
            from src.models.midlate_fusion_variants import ConvNeXtBaseMidLateGate
            img_model = ConvNeXtBaseMidLateGate(tab_dim=tab_dim, out_dim=2).to(device)
        # New: VGG19 mid+late dual fusion
        elif backbone == "vgg19_film_dual":
            from src.models.midlate_fusion_variants import VGG19MidLateFiLM
            img_model = VGG19MidLateFiLM(tab_dim=tab_dim, out_dim=2).to(device)
        elif backbone == "vgg19_gate_dual":
            from src.models.midlate_fusion_variants import VGG19MidLateGate
            img_model = VGG19MidLateGate(tab_dim=tab_dim, out_dim=2).to(device)
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
        elif backbone in ("yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
                          "yolov8n_cls", "yolov8s_cls", "yolov8m_cls", "yolov8l_cls", "yolov8x_cls"):
            from src.models.yolov8_cls_backbone import YOLOv8ClsBackbone
            # normalize to hyphenated convention for ultralytics
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLOv8ClsBackbone(model_name=model_name, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
                          "yolo11n_cls", "yolo11s_cls", "yolo11m_cls", "yolo11l_cls", "yolo11x_cls"):
            from src.models.yolo11_cls_backbone import YOLO11ClsBackbone
            model_name = backbone.replace('_cls', '-cls')
            img_model = YOLO11ClsBackbone(model_name=model_name, pretrained=True, out_dim=2).to(device)
        elif backbone in ("yolov5n-cls", "yolov5s-cls", "yolov5m-cls", "yolov5l-cls", "yolov5x-cls",
                          "yolov5n_cls", "yolov5s_cls", "yolov5m_cls", "yolov5l_cls", "yolov5x_cls"):
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
    # Select best epoch by ROC-AUC by default
    ap.add_argument("--select_metric", default="acc", help="Validation metric key to select best epoch (e.g., auc, bal_acc)")
    # Tune threshold on inner-val using a potentially different metric (default: bal_acc)
    ap.add_argument("--thr_metric", default="bal_acc", help="Metric to tune decision threshold on inner-val (e.g., bal_acc, f1, sensitivity)")
    # Optional constraints: enforce minimum specificity/sensitivity during threshold tuning/selection
    ap.add_argument("--min_specificity", type=float, default=None, help="If set, maximize --thr_metric among thresholds with specificity >= this value; falls back unconstrained if none meet it")
    ap.add_argument("--min_sensitivity", type=float, default=None, help="If set, maximize --thr_metric among thresholds with sensitivity >= this value; falls back unconstrained if none meet it")
    # Classic selection constraint (no per-epoch threshold tuning): require a minimum sensitivity at fixed thr=0.5 when selecting best epoch
    ap.add_argument("--min_classic_sensitivity", type=float, default=None, help="Minimum sensitivity (at thr=0.5) required for classic best-epoch updates; ignored with --joint_select")
    ap.add_argument("--require_floors", action="store_true", help="Require thresholds to meet both min_specificity/min_sensitivity for best selection; if infeasible at an epoch, do not update best")
    ap.add_argument("--inner_val_frac", type=float, default=0.3, help="Fraction of outer-train used as inner validation for selection (grouped by patient, stratified if possible)")
    ap.add_argument("--balanced_inner_val", action="store_true", help="Pick inner-val patients to balance frame-level pos fraction (greedy, keeps patient integrity)")
    ap.add_argument("--inner_val_target_pos_frac", type=float, default=0.5, help="Target positive-frame fraction for balanced inner-val when --balanced_inner_val is set")
    ap.add_argument("--inner_val_max_instances", type=int, default=None, help="Optional max number of instances (frames) allowed in inner-val when --balanced_inner_val is set (e.g., 800). Caps total inner-val size; selection still targets --inner_val_target_pos_frac.")
    ap.add_argument("--joint_select", action="store_true", help="Select (epoch, threshold) jointly by maximizing --thr_metric on inner-val each epoch for the selection domain")
    # Tune thresholds at end (post-hoc) only when explicitly requested; ignored if --joint_select is provided
    ap.add_argument("--tune_end", "--tune-end", action="store_true", help="Tune thresholds at end using best epoch inner-val predictions (ignored with --joint_select). Default: disabled")
    ap.add_argument("--eval_outer_each_epoch", action="store_true", help="Also evaluate on outer held-out val every epoch (off by default)")
    # Optional inner-train augmentation (post inner split, inner-val untouched)
    ap.add_argument("--augment", action="store_true", help="Augment inner-train only (after inner split). Uses on-disk image synthesis; inner-val remains pristine.")
    ap.add_argument("--special_aug", action="store_true", help="Use extended/legacy augmentation (flips, brightness/contrast, noise, blur, CLAHE). By default only affine (translate±25px, rotate±25°, scale[0.8,1.7]) is applied.")
    ap.add_argument("--aug_label", type=int, choices=[0,1], default=0, help="Label to augment when --augment is set: 0=negatives, 1=positives")
    ap.add_argument("--aug_match_count", action="store_true", help="If set, augment chosen label to match the opposite label count in inner-train")
    ap.add_argument("--aug_factor", type=float, default=1.0, help="If --aug_match_count is not set, multiply chosen label count by this factor (e.g., 2.0 doubles)")
    ap.add_argument("--aug_max_per_frame", type=int, default=5, help="Max number of augmented copies per original frame in inner-train")
    ap.add_argument("--aug_img_size", type=int, default=None, help="Optional resize max-size for augmentation images")
    ap.add_argument("--aug_root", default="data/augmented-cadica", help="Directory to write augmented images under selectedVideos/p#/v#/input/")
    ap.add_argument("--log_aug_examples", action="store_true", help="If set, prints a few augmented file paths per fold for verification")
    # New augmentation mode: per-positive factor and negatives to half of augmented positives
    ap.add_argument("--augment_pos_factor_neg_half", type=int, default=None, help="If set, for inner-train: add this many augmented copies per positive sample, then augment negatives so total negatives (orig+aug) = 0.5 * total positives after augmentation. Overrides other --augment* flags.")
    # New augmentation mode: target negative percentage of total after augmentation
    ap.add_argument("--aug_neg_percent", type=float, default=None, help="If set (e.g., 0.3), augment negatives so that negatives are this fraction of total (pos+neg) after augmentation in inner-train. Overrides --augment*, but lower precedence than --augment_pos_factor_neg_half.")
    # Independent class-weighted loss toggle
    ap.add_argument("--weighted", action="store_true", help="Use class-weighted cross-entropy based on inner-train class counts (post-augmentation if any)")
    ap.add_argument("--class_weights", type=str, default=None, help="Manual class weights as 'w_neg,w_pos'. Used only if --weighted is set; overrides automatic weights.")
    ap.add_argument("--final_test_eval", action="store_true", help="After CV per model, retrain on all non-test data and evaluate on pristine test")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv", help="Path to pristine test CSV")
    ap.add_argument("--trainval_pool_csv", default="outputs/index/trainval_pool.csv", help="Combined non-test pool for final training")
    ap.add_argument("--only_final", action="store_true", help="Skip CV entirely; train on --trainval_pool_csv and evaluate on --test_csv only")
    # Early stopping for final training
    ap.add_argument("--es_final", action="store_true", help="Enable early stopping in final full-pool training based on train CE loss")
    ap.add_argument("--es_final_patience", type=int, default=10, help="Patience (epochs without train-loss improvement) for --es_final")
    # PIM-specific knobs mirroring original hyperparams
    ap.add_argument("--amp", default="off", choices=["on","off"], help="Enable AMP")
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

    # Optional short-circuit: only final full-pool training/eval
    if args.only_final:
        for backbone in backbones:
            for fusion in fusions:
                run_root = os.path.join(args.outroot, f"{backbone}_{fusion}")
                os.makedirs(run_root, exist_ok=True)
                print(f"\n=== Final-only: backbone={backbone} fusion={fusion} ===")

                # sanity check CSVs
                if not os.path.exists(args.trainval_pool_csv):
                    print(f"[warn] trainval pool CSV not found: {args.trainval_pool_csv}. Skipping {backbone}/{fusion}.")
                    continue
                if not os.path.exists(args.test_csv):
                    print(f"[warn] pristine test CSV not found: {args.test_csv}. Skipping {backbone}/{fusion}.")
                    continue

                # datasets
                tf_train = get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)
                full_train_ds = CadicaDataset(
                    args.trainval_pool_csv, metadata, tf_train,
                    selected_features=sel_nc, text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                )
                test_ds = CadicaDataset(
                    args.test_csv, metadata, get_val_tf(img_size),
                    selected_features=sel_nc, text_features=text_cols,
                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                )

                # Optional augmentation for final train (reuses logic in final_test_eval block)
                class_weights_full = None
                try:
                    # import cv2/Numpy here if needed
                    import cv2 as _cv2
                    import numpy as _np
                    import pandas as _pd
                    # minimal inline of earlier augmentation helper
                    try:
                        import albumentations as A  # type: ignore
                    except Exception:
                        A = None
                    def _make_augmenter(img_size=None, special=False):
                        if A is None:
                            return None
                        if special:
                            ops = [
                                A.HorizontalFlip(p=0.5),
                                A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-10,10), p=0.7),
                                A.RandomBrightnessContrast(0.15,0.15,p=0.5),
                                A.MotionBlur(blur_limit=3,p=0.2),
                            ]
                            try:
                                import inspect as _inspect
                                params = _inspect.signature(A.GaussNoise.__init__).parameters
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
                        try:
                            use_px = 'translate_px' in __import__('inspect').signature(A.Affine.__init__).parameters
                        except Exception:
                            use_px = False
                        ops = [A.LongestMaxSize(max_size=img_size)] if img_size is not None else []
                        if use_px:
                            ops.append(A.Affine(translate_px={"x": (-25,25), "y": (-25,25)}, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                        else:
                            base = img_size if img_size is not None else 224
                            shift = 25.0/float(base)
                            ops.append(A.Affine(translate_percent=shift, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                        return A.Compose(ops)
                    def _apply_aug(img3, aug):
                        if aug is None:
                            h, w = img3.shape[:2]
                            cx, cy = w/2.0, h/2.0
                            angle = random.uniform(-25.0, 25.0)
                            scale = random.uniform(0.8, 1.7)
                            dx = random.uniform(-25.0, 25.0)
                            dy = random.uniform(-25.0, 25.0)
                            M = _cv2.getRotationMatrix2D((cx, cy), angle, scale)
                            M[0,2] += dx; M[1,2] += dy
                            return _cv2.warpAffine(img3, M, (w, h), flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_REFLECT_101)
                        out = aug(image=img3); return out['image']

                    if args.augment_pos_factor_neg_half is not None or args.aug_neg_percent is not None or args.augment:
                        df_full = _pd.read_csv(args.trainval_pool_csv).copy()
                        aug = _make_augmenter(img_size, special=args.special_aug)
                        fold_out = os.path.join(run_root, 'final_full')
                        os.makedirs(fold_out, exist_ok=True)
                        aug_rows = []
                        if args.augment_pos_factor_neg_half is not None:
                            fpos = max(0, int(args.augment_pos_factor_neg_half))
                            pos_df = df_full[df_full['label'] == 1]
                            neg_df = df_full[df_full['label'] == 0]
                            pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                            pos_after = pos_ct + pos_ct * fpos
                            target_neg_total = int(_np.ceil(0.5 * pos_after))
                            neg_needed = max(0, target_neg_total - neg_ct)
                            # positives
                            for r in pos_df.itertuples(index=False):
                                p = int(r.patient); v = int(r.video)
                                sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                base_max = int(sub['frame'].max()) if len(sub) else 0
                                for i in range(fpos):
                                    img = _cv2.imread(r.img_path, _cv2.IMREAD_GRAYSCALE)
                                    if img is None: continue
                                    img3 = _cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug)
                                    aug_gray = _cv2.cvtColor(aug_img, _cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    new_k = base_max + 1 + i
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    if _cv2.imwrite(out_path, aug_gray):
                                        aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 1})
                            # negatives
                            produced = 0
                            for row in neg_df.itertuples(index=False):
                                if produced >= neg_needed: break
                                p = int(row.patient); v = int(row.video)
                                sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                base_max = int(sub['frame'].max()) if len(sub) else 0
                                extra_for_pv = sum(1 for rr in aug_rows if rr['patient']==p and rr['video']==v)
                                new_k = base_max + extra_for_pv + 1
                                img = _cv2.imread(row.img_path, _cv2.IMREAD_GRAYSCALE)
                                if img is None: continue
                                img3 = _cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                aug_img = _apply_aug(img3, aug)
                                aug_gray = _cv2.cvtColor(aug_img, _cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                os.makedirs(out_dir, exist_ok=True)
                                out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                out_path = os.path.join(out_dir, out_name)
                                if _cv2.imwrite(out_path, aug_gray):
                                    aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0}); produced += 1
                        elif args.aug_neg_percent is not None:
                            r = float(args.aug_neg_percent)
                            if 0.0 < r < 1.0:
                                pos_df = df_full[df_full['label'] == 1]
                                neg_df = df_full[df_full['label'] == 0]
                                pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                                target_neg_total = int(_np.ceil((r / max(1e-6, 1.0 - r)) * float(pos_ct)))
                                neg_needed = max(0, target_neg_total - neg_ct)
                                produced = 0
                                for row in neg_df.itertuples(index=False):
                                    if produced >= neg_needed: break
                                    p = int(row.patient); v = int(row.video)
                                    sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                    base_max = int(sub['frame'].max()) if len(sub) else 0
                                    extra_for_pv = sum(1 for rr in aug_rows if rr['patient']==p and rr['video']==v)
                                    new_k = base_max + extra_for_pv + 1
                                    img = _cv2.imread(row.img_path, _cv2.IMREAD_GRAYSCALE)
                                    if img is None: continue
                                    img3 = _cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug)
                                    aug_gray = _cv2.cvtColor(aug_img, _cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    if _cv2.imwrite(out_path, aug_gray):
                                        aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0}); produced += 1
                        elif args.augment:
                            tgt_label = int(args.aug_label)
                            factor = float(args.aug_factor)
                            src_df = df_full[df_full['label'] == tgt_label]
                            opp_df = df_full[df_full['label'] != tgt_label]
                            S = int(len(src_df)); O = int(len(opp_df))
                            target_ct = max(0, O - S) if args.aug_match_count else max(0, int(round(S * max(0.0, float(factor) - 1.0))))
                            produced = 0
                            for row in src_df.itertuples(index=False):
                                if produced >= target_ct: break
                                p = int(row.patient); v = int(row.video)
                                sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                base_max = int(sub['frame'].max()) if len(sub) else 0
                                per = 0
                                while per < int(args.aug_max_per_frame) and produced < target_ct:
                                    img = _cv2.imread(row.img_path, _cv2.IMREAD_GRAYSCALE)
                                    if img is None: break
                                    img3 = _cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug)
                                    aug_gray = _cv2.cvtColor(aug_img, _cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    extra_for_pv = sum(1 for rr in aug_rows if rr['patient']==p and rr['video']==v)
                                    new_k = base_max + extra_for_pv + 1
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    if _cv2.imwrite(out_path, aug_gray):
                                        aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': tgt_label}); produced += 1; per += 1
                        if len(aug_rows) > 0:
                            aug_df = _pd.concat([df_full, _pd.DataFrame(aug_rows)], ignore_index=True)
                            aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                            is_aug = aug_df['img_path'].astype(str).str.contains(aug_tag)
                            def _ok(pth: str) -> bool:
                                try: return os.path.exists(pth) and (_cv2.imread(pth, _cv2.IMREAD_GRAYSCALE) is not None)
                                except Exception: return False
                            ok_mask = (~is_aug) | aug_df['img_path'].astype(str).apply(_ok)
                            aug_df = aug_df[ok_mask].reset_index(drop=True)
                            aug_csv = os.path.join(run_root, 'final_full', 'final_train_aug.csv')
                            aug_df.to_csv(aug_csv, index=False)
                            full_train_ds = CadicaDataset(
                                aug_csv, metadata, tf_train,
                                selected_features=sel_nc, text_features=text_cols,
                                text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                stats=full_train_ds.stats,
                            )
                            if args.weighted:
                                new_pos = int((aug_df['label'] == 1).sum()); new_neg = int((aug_df['label'] == 0).sum())
                                tot = float(new_pos + new_neg)
                                w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                class_weights_full = torch.tensor([w0, w1], dtype=torch.float32)
                except Exception as _e:
                    print(f"[warn] final-only augmentation skipped: {_e}")

                # loaders
                tr_full = DataLoader(full_train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=args.num_workers, pin_memory=True)
                tr_full_eval = DataLoader(full_train_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=args.num_workers)
                te = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=args.num_workers)

                # model + optimizer
                img_model_f, clin_mlp_f, late_head_f, params_f = build_models(
                    backbone=backbone, fusion=fusion, tab_dim=full_train_ds.tab_dim, pat_channels=cfg.get("pat_channels", 4), device=device,
                )
                opt_name = cfg.get("optimizer", "adamw").lower(); lr = cfg.get("lr", 3e-4); wd = cfg.get("weight_decay", 1e-4)
                if opt_name in ("adamw", "adam"):
                    optimizer_f = torch.optim.AdamW(params_f, lr=lr, weight_decay=wd)
                elif opt_name in ("sgd", "sgdm"):
                    optimizer_f = torch.optim.SGD(params_f, lr=lr, momentum=0.9, weight_decay=wd)
                elif opt_name == "rmsprop":
                    optimizer_f = torch.optim.RMSprop(params_f, lr=lr, weight_decay=wd)
                else:
                    optimizer_f = torch.optim.AdamW(params_f, lr=lr, weight_decay=wd)

                # train with training curves (frame acc, video acc tsmooth_max, CE loss)
                tr_acc_hist, tr_acc_video_hist, tr_loss_hist = [], [], []
                best_loss = float('inf')
                best_state = {"img": None, "clin": None, "late": None}
                best_epoch = None
                no_improve = 0
                for epoch in range(1, epochs + 1):
                    print(f"[final-only-train] {backbone}/{fusion} | Epoch {epoch}/{epochs}")
                    train_one_epoch(
                        img_model_f, optimizer_f, tr_full, device,
                        mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f,
                        use_amp=(args.amp=="on"), class_weights=class_weights_full,
                    )
                    # capture curves
                    try:
                        fm_tr, _, _ = evaluate(
                            img_model_f, tr_full_eval, device,
                            mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f, also_video=False,
                        )
                        tr_acc_hist.append(float(fm_tr.get("acc", float("nan"))))
                    except Exception:
                        tr_acc_hist.append(float("nan"))
                    try:
                        # video acc with tsmooth_max
                        _, _, _, v_tr, _ = evaluate(
                            img_model_f, tr_full_eval, device,
                            mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f, also_video=True, video_reduce="tsmooth_max",
                        )
                        acc_v = None
                        if isinstance(v_tr, dict):
                            try:
                                acc_v = float(v_tr.get("acc", float("nan")))
                            except Exception:
                                acc_v = float("nan")
                        tr_acc_video_hist.append(acc_v if acc_v is not None else float("nan"))
                    except Exception:
                        tr_acc_video_hist.append(float("nan"))
                    try:
                        # CE loss
                        ce = torch.nn.CrossEntropyLoss(weight=(class_weights_full.to(device) if isinstance(class_weights_full, torch.Tensor) else class_weights_full))
                        total, n = 0.0, 0
                        img_model_f.eval()
                        if clin_mlp_f: clin_mlp_f.eval()
                        if late_head_f: late_head_f.eval()
                        with torch.no_grad():
                            for batch in tr_full_eval:
                                x = batch["x"].to(device)
                                y = batch["y"].to(device)
                                clin = batch["clin"].to(device)
                                if fusion == "late":
                                    feats, _ = img_model_f(x)
                                    logits = late_head_f(feats, clin_mlp_f(clin))
                                elif fusion == "early":
                                    _, logits = img_model_f(x, clin)
                                else:
                                    _, logits = img_model_f(x)
                                loss = ce(logits, y)
                                bs = int(y.shape[0])
                                total += float(loss.item()) * bs
                                n += bs
                        tr_loss_hist.append(float(total / max(1, n)))
                    except Exception:
                        tr_loss_hist.append(float("nan"))

                    # Early stopping on training loss (final-only)
                    if args.es_final:
                        cur_loss = tr_loss_hist[-1]
                        if np.isfinite(cur_loss) and (cur_loss + 1e-12) < best_loss:
                            best_loss = float(cur_loss)
                            best_epoch = int(epoch)
                            best_state["img"] = deepcopy(img_model_f.state_dict())
                            best_state["clin"] = deepcopy(clin_mlp_f.state_dict()) if clin_mlp_f else None
                            best_state["late"] = deepcopy(late_head_f.state_dict()) if late_head_f else None
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= int(args.es_final_patience):
                                print(f"[final-only/es] early stop at epoch {epoch}; best_loss={best_loss:.6f} @ epoch {best_epoch}")
                                break
                # save training curves figure
                try:
                    PALETTE = ["#2078B3", "#FA9B9A", "#B2DF8A", "#FEBE70", "#FA9B9A"]
                    figs_dir = os.path.join(run_root, "final_full", "figs")
                    os.makedirs(figs_dir, exist_ok=True)
                    if plt is not None:
                        epochs_axis = list(range(1, len(tr_acc_hist)+1))
                        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'figure.titlesize':12})
                        fig, ax1 = plt.subplots(figsize=(8,5), dpi=150)
                        ax1.set_xlabel("Epoch")
                        ax1.set_ylabel("Train CE loss", color=PALETTE[1])
                        l_loss, = ax1.plot(epochs_axis, tr_loss_hist, color=PALETTE[1], lw=2.0, label="Loss")
                        ax1.tick_params(axis='y', labelcolor=PALETTE[1])
                        ax1.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)
                        ax1.minorticks_on(); ax1.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)
                        ax2 = ax1.twinx(); ax2.set_ylabel("Train acc", color=PALETTE[0])
                        l_acc_f, = ax2.plot(epochs_axis, tr_acc_hist, color=PALETTE[0], lw=2.2, label="Acc (frame)")
                        l_acc_v, = ax2.plot(epochs_axis, tr_acc_video_hist, color=PALETTE[2], lw=2.0, ls='--', label="Acc (video tsmooth_max)")
                        ax2.tick_params(axis='y', labelcolor=PALETTE[0])
                        lines = [l_loss, l_acc_f, l_acc_v]; labels = [ln.get_label() for ln in lines]
                        # Legend above axes to avoid overlapping the curves
                        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
                        fig.tight_layout(rect=[0, 0, 1, 0.92])
                        for ext in ("png","pdf","svg"):
                            plt.savefig(os.path.join(figs_dir, f"training_curves.{ext}"))
                        plt.close()
                except Exception as _e:
                    print(f"[warn] final-only training curves failed: {_e}")

                # If ES used and a best state was captured, restore it before evaluation
                if args.es_final and best_state["img"] is not None:
                    try:
                        img_model_f.load_state_dict(best_state["img"])
                        if clin_mlp_f and best_state["clin"] is not None:
                            clin_mlp_f.load_state_dict(best_state["clin"])
                        if late_head_f and best_state["late"] is not None:
                            late_head_f.load_state_dict(best_state["late"])
                    except Exception:
                        pass

                # evaluate on pristine test
                fmet, yf_test, pf_test, vmet, vdf = evaluate(
                    img_model_f, te, device, mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f,
                    also_video=True, video_reduce=args.video_reduce,
                )
                # Exclude nested aggregator map ('_multi') from video metrics when casting
                final_video = {}
                for k, v in (vmet.items() if isinstance(vmet, dict) else []):
                    if k == "_multi":
                        continue
                    try:
                        final_video[k] = float(v)
                    except Exception:
                        pass
                final_test_metrics = {"frame": {k: float(v) for k, v in fmet.items()}, "video": final_video}
                print("[final-only-test/frame]", {k: round(float(v), 4) for k, v in fmet.items()})
                print("[final-only-test/video]", {k: round(float(v), 4) for k, v in final_video.items()})
                with open(os.path.join(run_root, "final_test_metrics.json"), "w") as f:
                    json.dump(final_test_metrics, f, indent=2)
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
                # Write summary.txt with final test metrics
                try:
                    df_row = {"fold": 0}
                    for k, v in fmet.items():
                        df_row[f"val_frame_{k}"] = float(v)
                    # choose video_reduce result already in vdf["y_prob"], but we also include metrics from vmet (excluding _multi)
                    for k, v in (vmet.items() if isinstance(vmet, dict) else []):
                        if k == "_multi":
                            continue
                        df_row[f"val_video_{k}"] = float(v)
                    df_one = pd.DataFrame([df_row])
                    _write_model_summary_txt(os.path.join(run_root, "summary.txt"), backbone, fusion, df_one, final_test=final_test_metrics)
                except Exception as _e:
                    print(f"[warn] failed to write final-only summary.txt: {_e}")

                # Save final-only figures (training curves are already captured during epochs using final-only mode? Not yet; do a quick compute for test only)
                try:
                    PALETTE = ["#2078B3", "#FA9B9A", "#B2DF8A", "#FEBE70", "#FA9B9A"]
                    figs_dir = os.path.join(run_root, "final_full", "figs")
                    os.makedirs(figs_dir, exist_ok=True)
                    if plt is not None:
                        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
                        import numpy as _np
                        # ROC
                        try:
                            fpr, tpr, _ = roc_curve(yf_test, pf_test)
                            plt.figure(figsize=(5,4), dpi=150)
                            plt.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"ROC (AUC={float(fmet.get('auc', 0.0)):.3f})")
                            plt.plot([0,1],[0,1], color="#888", ls="--", lw=1)
                            plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)"); plt.title("ROC (test)")
                            plt.grid(alpha=0.25); plt.legend(loc="lower right"); plt.tight_layout()
                            for ext in ("png","pdf","svg"):
                                plt.savefig(os.path.join(figs_dir, f"roc_test.{ext}"))
                            plt.close()
                        except Exception as _e:
                            print(f"[warn] ROC plot (final-only) failed: {_e}")
                        # PR
                        try:
                            prec, rec, _ = precision_recall_curve(yf_test, pf_test)
                            plt.figure(figsize=(5,4), dpi=150)
                            plt.plot(rec, prec, color=PALETTE[3], lw=2, label=f"PR (AP={float(fmet.get('ap', 0.0)):.3f})")
                            plt.xlabel("Recall (Sensitivity)"); plt.ylabel("Precision (PPV)"); plt.title("PR (test)")
                            plt.grid(alpha=0.25); plt.legend(loc="lower left"); plt.tight_layout()
                            for ext in ("png","pdf","svg"):
                                plt.savefig(os.path.join(figs_dir, f"pr_test.{ext}"))
                            plt.close()
                        except Exception as _e:
                            print(f"[warn] PR plot (final-only) failed: {_e}")
                        # Confusion at 0.5
                        try:
                            yhat = (pf_test >= 0.5).astype(int)
                            cm = confusion_matrix(yf_test, yhat, labels=[0,1])
                            plt.figure(figsize=(4.5,4.0), dpi=150)
                            plt.imshow(cm, cmap="Blues")
                            plt.xticks([0,1],["Pred 0","Pred 1"]); plt.yticks([0,1],["True 0","True 1"])
                            for i in range(2):
                                for j in range(2):
                                    plt.text(j, i, str(int(cm[i,j])), ha='center', va='center', color="#111", fontsize=12)
                            plt.title("Confusion Matrix (thr=0.5)"); plt.tight_layout()
                            for ext in ("png","pdf","svg"):
                                plt.savefig(os.path.join(figs_dir, f"confusion_test.{ext}"))
                            plt.close()
                        except Exception as _e:
                            print(f"[warn] Conf matrix (final-only) failed: {_e}")
                except Exception as _e:
                    print(f"[warn] failed to save final-only figures: {_e}")
        # short-circuit completes here
        return

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
                    train_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
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

                # inner split on patients from the outer-train CSV
                if args.balanced_inner_val:
                    # Greedy patient selection to balance frame-level pos fraction in inner-val
                    pats = []
                    for pid, g in train_df.groupby("patient"):
                        pid = int(pid)
                        pos = int((g["label"] == 1).sum())
                        neg = int((g["label"] == 0).sum())
                        tot = pos + neg
                        if tot <= 0:
                            continue
                        frac = pos / float(tot)
                        pats.append({"patient": pid, "pos": pos, "neg": neg, "tot": tot, "pos_frac": frac})
                    # sort to have deterministic base order (closest to target, larger tot first, then id)
                    target = float(args.inner_val_target_pos_frac)
                    pats.sort(key=lambda x: (abs(x["pos_frac"] - target), -x["tot"], x["patient"]))
                    n_target = max(1, int(round(len(pats) * float(args.inner_val_frac))))
                    selected = []
                    remaining = pats[:]
                    pos_sum = 0
                    neg_sum = 0
                    cap = int(args.inner_val_max_instances) if args.inner_val_max_instances is not None else None
                    while len(selected) < n_target and remaining:
                        # stop if capacity reached
                        if cap is not None and (pos_sum + neg_sum) >= cap:
                            break
                        best_i = None
                        best_key = None
                        for i, cand in enumerate(remaining):
                            # capacity check: skip candidates that would exceed cap
                            if cap is not None and (pos_sum + neg_sum + cand["tot"]) > cap:
                                continue
                            new_pos = pos_sum + cand["pos"]
                            new_neg = neg_sum + cand["neg"]
                            new_tot = max(1, new_pos + new_neg)
                            new_frac = new_pos / float(new_tot)
                            delta = abs(new_frac - target)
                            key = (delta, -cand["tot"], cand["patient"])  # prefer larger patients, then smaller id
                            if best_i is None or key < best_key:
                                best_i, best_key = i, key
                        if best_i is None:
                            # no candidate fits remaining capacity
                            break
                        chosen = remaining.pop(best_i)
                        selected.append(chosen)
                        pos_sum += chosen["pos"]
                        neg_sum += chosen["neg"]
                    va_pats = set(int(p["patient"]) for p in selected)
                    tr_pats = set(int(p["patient"]) for p in pats if int(p["patient"]) not in va_pats)
                else:
                    # Stratified by patient-level label if possible
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
                def _patient_ct(df: pd.DataFrame):
                    # Robust patient counting: prefer numeric IDs; fallback to string IDs
                    if "patient" not in df.columns:
                        return 0
                    try:
                        ser = pd.to_numeric(df["patient"], errors="coerce")
                        n = int(ser.dropna().nunique())
                        if n > 0:
                            return n
                        # Fallback for non-numeric formats (e.g., 'p12')
                        return int(df["patient"].astype(str).str.strip().nunique())
                    except Exception:
                        return int(df["patient"].astype(str).str.strip().nunique())

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

                # Patient counts for each split
                ot_p = _patient_ct(train_df)
                it_p = _patient_ct(inner_tr_df)
                iv_p = _patient_ct(inner_va_df)
                ov_p = _patient_ct(val_df)

                print(f"[data] {fold_name} sizes:")
                if args.balanced_inner_val:
                    try:
                        cap = int(args.inner_val_max_instances) if args.inner_val_max_instances is not None else None
                        tgt = float(args.inner_val_target_pos_frac)
                        if cap is not None:
                            print(f"  [inner-val target] pos_frac={tgt:.3f}, max_instances={cap}")
                        else:
                            print(f"  [inner-val target] pos_frac={tgt:.3f}, max_instances=None (using inner_val_frac by patients)")
                    except Exception:
                        pass
                print(f"  outer-train: patients={ot_p}  n={ot_n}  pos={ot_pos}  neg={ot_neg}")
                print(f"  inner-train: patients={it_p}  n={it_n}  pos={it_pos}  neg={it_neg}")
                print(f"  inner-val  : patients={iv_p}  n={iv_n}  pos={iv_pos}  neg={iv_neg}")
                print(f"  outer-val  : patients={ov_p}  n={ov_n}  pos={ov_pos}  neg={ov_neg}")

                from torch.utils.data import Subset
                # Optional on-disk augmentation for inner-train only
                def _try_import_albu():
                    try:
                        import albumentations as A
                        return A
                    except Exception:
                        return None

                def _build_aug_pipe(img_size=None, special=False):
                    A = _try_import_albu()
                    if A is None:
                        return None
                    # Special (legacy) pipeline with flips/color/noise if requested
                    if special:
                        ops = [
                            A.HorizontalFlip(p=0.5),
                            A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-10,10), p=0.7),
                            A.RandomBrightnessContrast(0.15,0.15,p=0.5),
                            A.MotionBlur(blur_limit=3,p=0.2),
                        ]
                        try:
                            import inspect as _inspect
                            params = _inspect.signature(A.GaussNoise.__init__).parameters
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
                    # Default/basic: only translate±25px, rotate±25°, scale[0.8,1.7]
                    try:
                        use_px = 'translate_px' in __import__('inspect').signature(A.Affine.__init__).parameters
                    except Exception:
                        use_px = False
                    if img_size is not None:
                        ops = [A.LongestMaxSize(max_size=img_size)]
                    else:
                        ops = []
                    if use_px:
                        ops.append(A.Affine(translate_px={"x": (-25,25), "y": (-25,25)}, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                    else:
                        # approximate 25 px after resize using percent; assume img_size if given else 224
                        base = img_size if img_size is not None else 224
                        shift = 25.0/float(base)
                        ops.append(A.Affine(translate_percent=shift, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                    return A.Compose(ops)

                def _apply_aug(img3, aug):
                    if aug is None:
                        # Fallback without Albumentations: implement basic affine using OpenCV
                        h, w = img3.shape[:2]
                        cx, cy = w/2.0, h/2.0
                        angle = random.uniform(-25.0, 25.0)
                        scale = random.uniform(0.8, 1.7)
                        dx = random.uniform(-25.0, 25.0)
                        dy = random.uniform(-25.0, 25.0)
                        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
                        M[0,2] += dx
                        M[1,2] += dy
                        return cv2.warpAffine(img3, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    out = aug(image=img3)
                    return out['image']

                # Build default inner-train dataset (no aug)
                inner_tr_ds = Subset(base_train_ds, inner_tr_idx)
                inner_va_ds = Subset(base_train_val_ds, inner_va_idx)

                # If new special augmentation is requested, it overrides other aug args
                class_weights = None
                if args.augment_pos_factor_neg_half is not None:
                    fpos = max(0, int(args.augment_pos_factor_neg_half))
                    fold_out = os.path.join(run_root, fold_name)
                    os.makedirs(fold_out, exist_ok=True)

                    inner_tr_df = train_df.iloc[inner_tr_idx].copy()
                    pos_df = inner_tr_df[inner_tr_df['label'] == 1]
                    neg_df = inner_tr_df[inner_tr_df['label'] == 0]
                    pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                    # target positives after augmentation
                    pos_after = pos_ct + pos_ct * fpos
                    # target negatives total should be half of positives after aug
                    target_neg_total = int(np.ceil(0.5 * pos_after))
                    neg_needed = max(0, target_neg_total - neg_ct)

                    aug = _build_aug_pipe(args.aug_img_size, special=bool(args.special_aug))
                    print(f"[augment-pos-neg-half] {fold_name} enabled | pos_factor={fpos} | target_neg_total={target_neg_total} | neg_needed={neg_needed} | albu={'on' if aug is not None else 'off'}")

                    aug_rows = []

                    # Augment each positive exactly fpos times
                    if fpos > 0 and pos_ct > 0:
                        for row in pos_df.itertuples(index=False):
                            p = int(row.patient); v = int(row.video); k = int(row.frame)
                            # base next frame id tracker per (p,v)
                            sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                            base_max = int(sub['frame'].max()) if len(sub) else 0
                            cur_max_key = (p, v, '__max_aug__')
                            cur_max = 0
                            for i in range(fpos):
                                img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    print(f"[warn] cannot read {row.img_path}; skipping pos aug")
                                    break
                                img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                aug_img = _apply_aug(img3, aug)
                                aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                new_k = base_max + cur_max + 1
                                cur_max += 1
                                out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                os.makedirs(out_dir, exist_ok=True)
                                out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                out_path = os.path.join(out_dir, out_name)
                                ok = cv2.imwrite(out_path, aug_gray)
                                if not ok:
                                    print(f"[warn] failed to write {out_path}; skipping pos aug")
                                    continue
                                aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 1})

                    # Augment negatives round-robin until neg_needed is met
                    if neg_needed > 0 and neg_ct > 0:
                        neg_list = list(neg_df.itertuples(index=False))
                        produced = 0
                        rr = 0
                        while produced < neg_needed and len(neg_list) > 0:
                            for row in neg_list:
                                if produced >= neg_needed:
                                    break
                                p = int(row.patient); v = int(row.video); k = int(row.frame)
                                sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                                base_max = int(sub['frame'].max()) if len(sub) else 0
                                # also consider any previously added for this (p,v) in this loop
                                # count existing aug rows for this (p,v)
                                extra_for_pv = sum(1 for r in aug_rows if r['patient']==p and r['video']==v)
                                new_k = base_max + extra_for_pv + 1
                                img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    print(f"[warn] cannot read {row.img_path}; skipping neg aug")
                                    continue
                                img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                aug_img = _apply_aug(img3, aug)
                                aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                os.makedirs(out_dir, exist_ok=True)
                                out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                out_path = os.path.join(out_dir, out_name)
                                ok = cv2.imwrite(out_path, aug_gray)
                                if not ok:
                                    print(f"[warn] failed to write {out_path}; skipping neg aug")
                                    continue
                                aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0})
                                produced += 1
                            rr += 1

                    # Build augmented inner-train dataframe and filter broken paths
                    if len(aug_rows) > 0:
                        inner_aug_df = pd.concat([inner_tr_df, pd.DataFrame(aug_rows)], ignore_index=True)
                        aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                        is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                        def _ok_path(pth: str) -> bool:
                            try:
                                return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                            except Exception:
                                return False
                        ok_mask = (~is_aug) | inner_aug_df['img_path'].astype(str).apply(_ok_path)
                        dropped = int((~ok_mask).sum())
                        if dropped > 0:
                            print(f"[augment-pos-neg-half] {fold_name} dropping {dropped} broken augmented rows before training")
                        inner_aug_df = inner_aug_df[ok_mask].reset_index(drop=True)
                        aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                        inner_aug_df.to_csv(aug_csv, index=False)
                        # Rebuild inner-train dataset
                        inner_tr_ds = CadicaDataset(
                            aug_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
                            selected_features=sel_nc,
                            text_features=text_cols,
                            text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                            text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                            stats=_stats,
                        )
                        # Log counts; optionally set class weights (neg is minority)
                        new_pos = int((inner_aug_df['label'] == 1).sum())
                        new_neg = int((inner_aug_df['label'] == 0).sum())
                        print(f"[augment-pos-neg-half] {fold_name} inner-train: pos={new_pos} neg={new_neg} (was pos={pos_ct} neg={neg_ct})")
                        # Also report patients and total n after augmentation
                        pat_aug = _patient_ct(inner_aug_df)
                        n_aug = int(inner_aug_df.shape[0])
                        print(f"[augment-pos-neg-half] {fold_name} inner-train (post-aug): patients={pat_aug}  n={n_aug}  pos={new_pos}  neg={new_neg}")
                        if args.weighted:
                            manual = args.class_weights
                            if manual:
                                try:
                                    cw = [float(x) for x in parse_list(manual, float)]
                                    if len(cw) == 2 and all(np.isfinite(c) and c > 0 for c in cw):
                                        w0, w1 = cw[0], cw[1]
                                        class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                                        print(f"[loss] class_weights set manually [neg={w0:.3f}, pos={w1:.3f}]")
                                    else:
                                        print("[warn] invalid --class_weights; falling back to automatic weights")
                                except Exception as _e:
                                    print(f"[warn] failed to parse --class_weights: {_e}; falling back to automatic weights")
                            if class_weights is None:
                                # inverse frequency weights
                                tot = float(new_pos + new_neg)
                                w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                                print(f"[loss] class_weights set to [neg={w0:.3f}, pos={w1:.3f}]")
                        else:
                            print("[loss] weighted CE disabled (use --weighted to enable)")
                        if args.log_aug_examples:
                            ex = inner_aug_df[inner_aug_df['img_path'].astype(str).str.contains(aug_tag)].head(3)['img_path'].tolist()
                            print(f"[augment-pos-neg-half] {fold_name} sample_aug_paths: {ex}")
                    else:
                        print(f"[augment-pos-neg-half] {fold_name}: no augmented rows produced; proceeding without augmentation")
                # Else, target negative percentage mode
                elif args.aug_neg_percent is not None:
                    r = float(args.aug_neg_percent)
                    if not (0.0 < r < 1.0):
                        print(f"[augment-neg-percent] {fold_name} invalid aug_neg_percent={r}; skipping this mode")
                    else:
                        fold_out = os.path.join(run_root, fold_name)
                        os.makedirs(fold_out, exist_ok=True)
                        inner_tr_df = train_df.iloc[inner_tr_idx].copy()
                        pos_ct = int((inner_tr_df['label'] == 1).sum())
                        neg_ct = int((inner_tr_df['label'] == 0).sum())
                        # N' / (P + N') = r -> N' = r*P / (1-r)
                        target_neg_total = int(np.ceil((r * pos_ct) / max(1e-8, (1.0 - r))))
                        neg_needed = max(0, target_neg_total - neg_ct)
                        aug = _build_aug_pipe(args.aug_img_size, special=bool(args.special_aug))
                        print(f"[augment-neg-percent] {fold_name} enabled | r={r:.3f} | target_neg_total={target_neg_total} | neg_needed={neg_needed} | albu={'on' if aug is not None else 'off'}")
                        if neg_needed > 0 and neg_ct > 0:
                            aug_rows = []
                            neg_df = inner_tr_df[inner_tr_df['label'] == 0]
                            neg_list = list(neg_df.itertuples(index=False))
                            produced = 0
                            while produced < neg_needed and len(neg_list) > 0:
                                for row in neg_list:
                                    if produced >= neg_needed:
                                        break
                                    p = int(row.patient); v = int(row.video); k = int(row.frame)
                                    sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                                    base_max = int(sub['frame'].max()) if len(sub) else 0
                                    extra_for_pv = sum(1 for r0 in aug_rows if r0['patient']==p and r0['video']==v)
                                    new_k = base_max + extra_for_pv + 1
                                    img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is None:
                                        print(f"[warn] cannot read {row.img_path}; skipping neg aug")
                                        continue
                                    img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug)
                                    aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    ok = cv2.imwrite(out_path, aug_gray)
                                    if not ok:
                                        print(f"[warn] failed to write {out_path}; skipping neg aug")
                                        continue
                                    aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0})
                                    produced += 1
                            if produced > 0:
                                inner_aug_df = pd.concat([inner_tr_df, pd.DataFrame(aug_rows)], ignore_index=True)
                                # filter broken paths
                                aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                                is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                                def _ok_path2(pth: str) -> bool:
                                    try:
                                        return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                    except Exception:
                                        return False
                                ok_mask = (~is_aug) | inner_aug_df['img_path'].astype(str).apply(_ok_path2)
                                dropped = int((~ok_mask).sum())
                                if dropped > 0:
                                    print(f"[augment-neg-percent] {fold_name} dropping {dropped} broken augmented rows before training")
                                inner_aug_df = inner_aug_df[ok_mask].reset_index(drop=True)
                                aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                                inner_aug_df.to_csv(aug_csv, index=False)
                                inner_tr_ds = CadicaDataset(
                                    aug_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
                                    selected_features=sel_nc,
                                    text_features=text_cols,
                                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                    stats=_stats,
                                )
                                new_pos = int((inner_aug_df['label'] == 1).sum())
                                new_neg = int((inner_aug_df['label'] == 0).sum())
                                frac = new_neg / max(1, (new_pos + new_neg))
                                print(f"[augment-neg-percent] {fold_name} inner-train: pos={new_pos} neg={new_neg} (neg_frac={frac:.3f}) from was pos={pos_ct} neg={neg_ct}")
                                # Also report patients and total n after augmentation
                                pat_aug = _patient_ct(inner_aug_df)
                                n_aug = int(inner_aug_df.shape[0])
                                print(f"[augment-neg-percent] {fold_name} inner-train (post-aug): patients={pat_aug}  n={n_aug}  pos={new_pos}  neg={new_neg}")
                                if args.weighted:
                                    manual = args.class_weights
                                    if manual:
                                        try:
                                            cw = [float(x) for x in parse_list(manual, float)]
                                            if len(cw) == 2 and all(np.isfinite(c) and c > 0 for c in cw):
                                                w0, w1 = cw[0], cw[1]
                                                class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                                                print(f"[loss] class_weights set manually [neg={w0:.3f}, pos={w1:.3f}]")
                                            else:
                                                print("[warn] invalid --class_weights; falling back to automatic weights")
                                        except Exception as _e:
                                            print(f"[warn] failed to parse --class_weights: {_e}; falling back to automatic weights")
                                    if class_weights is None:
                                        tot = float(new_pos + new_neg)
                                        w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                        w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                        class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                                        print(f"[loss] class_weights set to [neg={w0:.3f}, pos={w1:.3f}]")
                                else:
                                    print("[loss] weighted CE disabled (use --weighted to enable)")
                                if args.log_aug_examples:
                                    ex = inner_aug_df[inner_aug_df['img_path'].astype(str).str.contains(aug_tag)].head(3)['img_path'].tolist()
                                    print(f"[augment-neg-percent] {fold_name} sample_aug_paths: {ex}")
                            else:
                                print(f"[augment-neg-percent] {fold_name}: no augmented rows produced; proceeding without augmentation")

                # Else, original augmentation path for inner-train only
                elif args.augment:
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
                        aug = _build_aug_pipe(args.aug_img_size, special=bool(args.special_aug))
                        print(f"[augment] {fold_name} enabled | label={tgt_label} | match={bool(args.aug_match_count)} | factor={None if args.aug_match_count else factor} | max_per={int(args.aug_max_per_frame)} | albu={'on' if aug is not None else 'off'}")
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
                            # Validate augmented files exist and are readable; drop any broken rows
                            aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                            is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                            def _ok_path(p: str) -> bool:
                                try:
                                    return os.path.exists(p) and (cv2.imread(p, cv2.IMREAD_GRAYSCALE) is not None)
                                except Exception:
                                    return False
                            ok_mask = (~is_aug) | inner_aug_df['img_path'].astype(str).apply(_ok_path)
                            dropped = int((~ok_mask).sum())
                            if dropped > 0:
                                print(f"[augment] {fold_name} dropping {dropped} broken augmented rows before training")
                            inner_aug_df = inner_aug_df[ok_mask].reset_index(drop=True)
                            # Write temp CSV for this fold
                            aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                            inner_aug_df.to_csv(aug_csv, index=False)
                            # Recompute counts after dropping
                            is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                            aug_rows_ct = int(is_aug.sum())
                            aug_neg_rows_ct = int(((inner_aug_df['label']==0) & is_aug).sum())
                            # Rebuild inner-train dataset from augmented CSV using train transforms and existing stats
                            inner_tr_ds = CadicaDataset(
                                aug_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
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
                            print(f"[augment] {fold_name} csv={aug_csv} | rows={len(inner_aug_df)} | aug_rows={aug_rows_ct} | aug_neg_rows={aug_neg_rows_ct}")
                            # Also report patients and total n after augmentation
                            pat_aug = _patient_ct(inner_aug_df)
                            n_aug = int(inner_aug_df.shape[0])
                            print(f"[augment] {fold_name} inner-train (post-aug): patients={pat_aug}  n={n_aug}  pos={new_pos}  neg={new_neg}")
                            if args.log_aug_examples:
                                ex = inner_aug_df[is_aug].head(3)['img_path'].tolist()
                                print(f"[augment] {fold_name} sample_aug_paths: {ex}")

                bs = cfg.get("batch_size", 64)
                tr_loader = DataLoader(
                    inner_tr_ds,
                    batch_size=bs,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                # If --weighted requested but no weights set yet, use manual override or compute from current inner-train
                if args.weighted and class_weights is None:
                    manual = args.class_weights
                    if manual:
                        try:
                            cw = [float(x) for x in parse_list(manual, float)]
                            if len(cw) == 2 and all(np.isfinite(c) and c > 0 for c in cw):
                                w0, w1 = cw[0], cw[1]
                                class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                                print(f"[loss] class_weights set manually [neg={w0:.3f}, pos={w1:.3f}] (pre-train)")
                            else:
                                print("[warn] invalid --class_weights; falling back to automatic weights")
                        except Exception as _e:
                            print(f"[warn] failed to parse --class_weights: {_e}; falling back to automatic weights")
                    if class_weights is None:
                        try:
                            if 'inner_aug_df' in locals():
                                cnt_pos = int((inner_aug_df['label'] == 1).sum())
                                cnt_neg = int((inner_aug_df['label'] == 0).sum())
                            else:
                                cnt_pos = int((inner_tr_df['label'] == 1).sum())
                                cnt_neg = int((inner_tr_df['label'] == 0).sum())
                            tot = float(cnt_pos + cnt_neg)
                            w0 = tot / (2.0 * max(1.0, float(cnt_neg)))
                            w1 = tot / (2.0 * max(1.0, float(cnt_pos)))
                            class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                            print(f"[loss] class_weights set to [neg={w0:.3f}, pos={w1:.3f}] (from current inner-train)")
                        except Exception as _e:
                            print(f"[warn] failed to compute class weights: {_e}")
                try:
                    print(f"[loader] {fold_name} train_len={len(inner_tr_ds)} batches={len(tr_loader)} batch_size={bs}")
                except Exception:
                    pass
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
                best_sel_sens = -1.0
                best_sel_thr = None
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

                # Helper: choose best threshold by metric with optional min specificity, exact cutpoints, deterministic tie-breaks
                def _best_thr_from_arrays(y_true, y_prob, metric_key: str, min_spec: float | None = None, min_sens: float | None = None) -> float:
                    if metric_key == "auc":
                        return 0.5
                    uniq = np.unique(y_prob)
                    cand = np.unique(np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
                    best_t, best_s, best_sens = 0.5, -1.0, -1.0
                    fb_t, fb_s, fb_sens = 0.5, -1.0, -1.0
                    eps = 1e-12
                    for t in cand:
                        md = metrics_from_probs(y_true, y_prob, thr=float(t))
                        m = md.get(metric_key, float("nan"))
                        s = float(m) if np.isfinite(m) else -1.0
                        sens = float(md.get("sensitivity", float("nan")))
                        if not np.isfinite(sens):
                            sens = -1.0
                        # update fallback (unconstrained)
                        fb_update = False
                        if s > fb_s:
                            fb_update = True
                        elif abs(s - fb_s) <= eps and sens > fb_sens:
                            fb_update = True
                        elif abs(s - fb_s) <= eps and abs(sens - fb_sens) <= eps and float(t) < float(fb_t):
                            fb_update = True
                        if fb_update:
                            fb_s, fb_sens, fb_t = s, sens, float(t)
                        # enforce specificity constraint if provided
                        if min_spec is not None:
                            spec = float(md.get("specificity", float("nan")))
                            if not np.isfinite(spec) or spec + eps < float(min_spec):
                                continue
                        # enforce sensitivity constraint if provided
                        if min_sens is not None and (not np.isfinite(sens) or sens + eps < float(min_sens)):
                            continue
                        update = False
                        if s > best_s:
                            update = True
                        elif abs(s - best_s) <= eps and sens > best_sens:
                            update = True
                        elif abs(s - best_s) <= eps and abs(sens - best_sens) <= eps and float(t) < float(best_t):
                            update = True
                        if update:
                            best_s, best_sens, best_t = s, sens, float(t)
                    if (min_spec is not None or min_sens is not None) and best_s < 0.0:
                        return fb_t
                    return best_t

                for epoch in range(1, epochs + 1):
                    # track if any improvement happened this epoch
                    improved_this_epoch = False
                    print(f"\n{backbone}/{fusion} | {fold_name} | Epoch {epoch}/{epochs} | since_best={epochs_no_improve}")
                    # Indicate whether class-weighted loss is active
                    if class_weights is not None:
                        try:
                            w_neg, w_pos = [float(x) for x in class_weights.tolist()]
                            print(f"[loss] using class-weighted CE: neg={w_neg:.4f}, pos={w_pos:.4f}")
                        except Exception:
                            print(f"[loss] using class-weighted CE")
                    else:
                        print(f"[loss] using unweighted CE")
                    train_one_epoch(
                        img_model, optimizer, tr_loader, device,
                        mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                        pim_lambdas=None,
                        pim_update_freq=grad_accum_steps, use_amp=use_amp_flag,
                        class_weights=class_weights,
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

                    # selection: either classic (metric on domain) or joint (thr_metric with best threshold)
                    if args.joint_select:
                        # Compute best threshold and metric for the selection domain using thr_metric
                        if args.select_domain == "frame":
                            y_sel, p_sel = y_true_f, y_prob_f
                        else:
                            y_sel = video_df["y_true"].values
                            p_sel = video_df["y_prob"].values
                        # exact sweep with tie-breaks (uses helper below defined later)
                        thr_cand = _best_thr_from_arrays(y_sel, p_sel, args.thr_metric, args.min_specificity, args.min_sensitivity)
                        md = metrics_from_probs(y_sel, p_sel, thr=float(thr_cand))
                        score = float(md.get(args.thr_metric, 0.0))
                        sens = float(md.get("sensitivity", 0.0)) if np.isfinite(md.get("sensitivity", float("nan"))) else 0.0
                        spec = float(md.get("specificity", 0.0)) if np.isfinite(md.get("specificity", float("nan"))) else 0.0
                        # If floors are required, ensure both are met; otherwise skip update this epoch
                        if args.require_floors:
                            need_spec = args.min_specificity if args.min_specificity is not None else None
                            need_sens = args.min_sensitivity if args.min_sensitivity is not None else None
                            feasible = True
                            if need_spec is not None and spec + 1e-12 < float(need_spec):
                                feasible = False
                            if need_sens is not None and sens + 1e-12 < float(need_sens):
                                feasible = False
                            if not feasible:
                                epochs_no_improve += 1
                                print(f"[joint] constraints unmet at epoch {epoch}: spec={spec:.3f}, sens={sens:.3f} (need spec>={need_spec}, sens>={need_sens}); not updating best")
                                # skip to next epoch
                                continue
                        # tie-break across epochs: higher score, then higher sensitivity, then smaller threshold
                        update = False
                        eps = 1e-12
                        print(f"\n[debug] Score comparison:")
                        print(f"[debug] Current score: {score:.6f}, Best so far: {best:.6f}")
                        print(f"[debug] Current sensitivity: {sens:.6f}, Best sensitivity: {best_sel_sens:.6f}")
                        print(f"[debug] Current threshold: {float(thr_cand):.6f}, Best threshold: {float(best_sel_thr if best_sel_thr is not None else 1.0):.6f}")
                        print(f"[debug] Epsilon for comparisons: {eps:.6f}")

                        if score > best:
                            print(f"[debug] Update reason: New score {score:.6f} better than best {best:.6f}")
                            update = True
                        elif abs(score - best) <= eps and sens > best_sel_sens:
                            print(f"[debug] Update reason: Scores equal (diff={abs(score - best):.6f}), better sensitivity {sens:.6f} > {best_sel_sens:.6f}")
                            update = True
                        elif abs(score - best) <= eps and abs(sens - best_sel_sens) <= eps and float(thr_cand) < float(best_sel_thr if best_sel_thr is not None else 1.0):
                            print(f"[debug] Update reason: Scores and sensitivities equal, lower threshold {float(thr_cand):.6f} < {float(best_sel_thr if best_sel_thr is not None else 1.0):.6f}")
                            update = True
                        if update:
                            best = score
                            best_sel_sens = sens
                            best_sel_thr = float(thr_cand)
                            best_metrics = {
                                "frame": {k: float(v) for k, v in frame_metrics.items()},
                                "video": {k: float(v) for k, v in video_metrics.items() if k != "_multi"},
                            }
                            best_epoch = int(epoch)
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
                                        "thr_metric": args.thr_metric,
                                        "joint_select": True,
                                        "sel_thr": best_sel_thr,
                                    },
                                    ckpt_path,
                                )
                            # capture predictions from this epoch for later thresholding on the other domain
                            best_inner_frame = {"y": y_true_f, "p": y_prob_f}
                            best_inner_video_df = video_df.copy()
                            if args.save_val_probs:
                                video_df.to_csv(os.path.join(fold_out, "inner_val_video_probs.csv"), index=False)
                            improved_this_epoch = True
                            print(f"[ckpt] new best saved (joint): domain={args.select_domain} {args.thr_metric}={best:.4f} thr={best_sel_thr:.4f}")
                    else:
                        # classic selection score based on requested domain/metric (e.g., auc)
                        sel_dict = frame_metrics if args.select_domain == "frame" else {k: v for k, v in video_metrics.items() if k != "_multi"}
                        score = float(sel_dict.get(args.select_metric, 0.0)) if args.select_metric in sel_dict else 0.0
                        # optional classic constraint: require minimum sensitivity at thr=0.5 to update best
                        if args.min_classic_sensitivity is not None:
                            try:
                                cur_sens = float(sel_dict.get("sensitivity", float("nan")))
                            except Exception:
                                cur_sens = float("nan")
                            if not np.isfinite(cur_sens) or cur_sens + 1e-12 < float(args.min_classic_sensitivity):
                                print(f"[classic] constraints unmet at epoch {epoch}: sens={cur_sens:.3f} (need >= {args.min_classic_sensitivity}); not updating best")
                                cur_satisfies = False
                            else:
                                cur_satisfies = True
                        else:
                            cur_satisfies = True
                        if cur_satisfies and score > best:
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
                            improved_this_epoch = True
                            print(f"[ckpt] new best saved: {args.select_domain}/{args.select_metric}={best:.4f}")
                    # finalize early-stopping counter update for this epoch
                    if improved_this_epoch:
                        epochs_no_improve = 0
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

                # threshold tuning using inner-val predictions from best epoch uses the helper above

                thr_frame = 0.5
                thr_video = 0.5
                did_tune = False
                if args.joint_select and best_sel_thr is not None:
                    # Use the jointly selected threshold for the selection domain, tune the other domain now
                    if args.select_domain == "frame":
                        thr_frame = float(best_sel_thr)
                        if best_inner_video_df is not None:
                            yv_in = best_inner_video_df["y_true"].values
                            pv_in = best_inner_video_df["y_prob"].values
                            thr_video = _best_thr_from_arrays(yv_in, pv_in, args.thr_metric, args.min_specificity, args.min_sensitivity)
                    else:
                        thr_video = float(best_sel_thr)
                        if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                            thr_frame = _best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.thr_metric, args.min_specificity, args.min_sensitivity)
                    did_tune = True
                else:
                    # Classic: tune both from the best epoch's inner predictions ONLY if --tune_end was requested
                    if args.tune_end:
                        if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                            thr_frame = _best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.thr_metric, args.min_specificity, args.min_sensitivity)
                        if best_inner_video_df is not None:
                            yv_in = best_inner_video_df["y_true"].values
                            pv_in = best_inner_video_df["y_prob"].values
                            thr_video = _best_thr_from_arrays(yv_in, pv_in, args.thr_metric, args.min_specificity, args.min_sensitivity)
                        did_tune = True

                tuned_frame_metrics = metrics_from_probs(yf_outer, pf_outer, thr=thr_frame)
                yv = outer_video_df["y_true"].values
                pv = outer_video_df["y_prob"].values
                tuned_video_metrics = metrics_from_probs(yv, pv, thr=thr_video)

                # Report tuned metrics at chosen thresholds, focusing on clinical metrics
                def _subset_metrics(d: dict):
                    keys = ["specificity","precision","sensitivity","ppv","npv","f1","acc","bal_acc","auc","ap"]
                    return {k: round(float(d[k]), 4) for k in keys if k in d}
                label = "outer_val-tuned" if did_tune else "outer_val-fixed"
                print(f"[{label}/frame thr={thr_frame:.3f}]", _subset_metrics(tuned_frame_metrics))
                print(f"[{label}/video thr={thr_video:.3f}]", _subset_metrics(tuned_video_metrics))

                # --------- Save evaluation plots (per fold) ---------
                def _save_eval_plots(y_true, y_prob, thr, domain_label: str, out_dir: str):
                    if plt is None:
                        print("[warn] matplotlib not available; skipping plots")
                        return
                    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
                    figs_dir = os.path.join(out_dir, "figs")
                    # Color palette (from fig_comorbidities_bars.py)
                    PALETTE = ["#A5CDE2","#2078B3","#B2DF8A","#FEBE70","#FA9B9A","#CAB2D6","#8DD3C7"]

                    # ROC
                    try:
                        fpr, tpr, _ = roc_curve(y_true, y_prob)
                        auc_val = float(tuned_frame_metrics["auc"] if domain_label=="frame" else tuned_video_metrics["auc"])
                        plt.figure(figsize=(5,4), dpi=150)
                        plt.plot(fpr, tpr, color=PALETTE[1], lw=2, label=f"ROC (AUC={auc_val:.3f})")
                        # mark tuned point
                        from src.trainloop.metrics import metrics_from_probs as _mfp
                        md = _mfp(y_true, y_prob, thr=thr)
                        plt.scatter([md.get("fpr", 0.0)], [md.get("sensitivity", 0.0)], color=PALETTE[3], s=24, zorder=5, label=f"thr={thr:.3f}")
                        plt.plot([0,1],[0,1], color="#888", ls="--", lw=1)
                        plt.xlabel("1 - Specificity (FPR)")
                        plt.ylabel("Sensitivity (TPR)")
                        plt.title(f"ROC ({domain_label})")
                        plt.grid(alpha=0.25)
                        plt.legend(loc="lower right")
                        plt.tight_layout()
                        for ext in ("png","pdf"):
                            plt.savefig(os.path.join(figs_dir, f"roc_{domain_label}.{ext}"))
                        plt.close()
                    except Exception as e:
                        print(f"[warn] failed to save ROC plot ({domain_label}): {e}")

                    # PR
                    try:
                        prec, rec, _ = precision_recall_curve(y_true, y_prob)
                        ap_val = float(tuned_frame_metrics["ap"] if domain_label=="frame" else tuned_video_metrics["ap"])
                        plt.figure(figsize=(5,4), dpi=150)
                        plt.plot(rec, prec, color=PALETTE[3], lw=2, label=f"PR (AP={ap_val:.3f})")
                        from src.trainloop.metrics import metrics_from_probs as _mfp
                        md = _mfp(y_true, y_prob, thr=thr)
                        plt.scatter([md.get("sensitivity", 0.0)], [md.get("ppv", 0.0)], color=PALETTE[1], s=24, zorder=5, label=f"thr={thr:.3f}")
                        plt.xlabel("Recall (Sensitivity)")
                        plt.ylabel("Precision (PPV)")
                        plt.title(f"PR ({domain_label})")
                        plt.grid(alpha=0.25)
                        plt.legend(loc="lower left")
                        plt.tight_layout()
                        for ext in ("png","pdf"):
                            plt.savefig(os.path.join(figs_dir, f"pr_{domain_label}.{ext}"))
                        plt.close()
                    except Exception as e:
                        print(f"[warn] failed to save PR plot ({domain_label}): {e}")

                    # Sensitivity-Specificity tradeoff
                    try:
                        uniq = np.unique(y_prob)
                        cand = np.unique(np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
                        sens_list, spec_list = [], []
                        from src.trainloop.metrics import metrics_from_probs as _mfp
                        for t in cand:
                            md = _mfp(y_true, y_prob, thr=float(t))
                            sens_list.append(float(md.get("sensitivity", 0.0)))
                            spec_list.append(float(md.get("specificity", 0.0)))
                        plt.figure(figsize=(5,4), dpi=150)
                        plt.plot(spec_list, sens_list, color=PALETTE[2], lw=2)
                        md_thr = _mfp(y_true, y_prob, thr=thr)
                        plt.scatter([md_thr.get("specificity",0.0)], [md_thr.get("sensitivity",0.0)], color=PALETTE[4], s=24, zorder=5, label=f"thr={thr:.3f}")
                        plt.xlabel("Specificity")
                        plt.ylabel("Sensitivity")
                        plt.title(f"Sensitivity vs Specificity ({domain_label})")
                        plt.grid(alpha=0.25)
                        plt.legend(loc="lower left")
                        plt.tight_layout()
                        for ext in ("png","pdf"):
                            plt.savefig(os.path.join(figs_dir, f"sens_spec_{domain_label}.{ext}"))
                        plt.close()
                    except Exception as e:
                        print(f"[warn] failed to save Sens-Spec plot ({domain_label}): {e}")

                    # Calibration plot
                    try:
                        if calibration_curve is None:
                            raise RuntimeError("sklearn.calibration not available")
                        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
                        plt.figure(figsize=(5,4), dpi=150)
                        plt.plot(prob_pred, prob_true, marker="o", color=PALETTE[5], lw=2, label="Reliability")
                        plt.plot([0,1],[0,1], ls="--", color="#888", lw=1)
                        # ECE (uniform bins)
                        # Re-bin to 10 bins and compute abs gap weighted by bin count
                        bins = np.linspace(0,1,11)
                        idx = np.digitize(y_prob, bins) - 1
                        ece = 0.0
                        for b in range(10):
                            mask = idx == b
                            if not np.any(mask):
                                continue
                            conf = np.mean(y_prob[mask])
                            acc = np.mean(y_true[mask])
                            ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
                        plt.xlabel("Predicted probability")
                        plt.ylabel("Observed prevalence")
                        plt.title(f"Calibration ({domain_label}) ECE={ece:.3f}")
                        plt.grid(alpha=0.25)
                        plt.tight_layout()
                        for ext in ("png","pdf"):
                            plt.savefig(os.path.join(figs_dir, f"calibration_{domain_label}.{ext}"))
                        plt.close()
                    except Exception as e:
                        print(f"[warn] failed to save calibration plot ({domain_label}): {e}")

                    # Score distributions
                    try:
                        plt.figure(figsize=(5,4), dpi=150)
                        pos_scores = y_prob[y_true==1]
                        neg_scores = y_prob[y_true==0]
                        plt.hist(neg_scores, bins=30, range=(0,1), density=True, alpha=0.6, color=PALETTE[1], label="neg")
                        plt.hist(pos_scores, bins=30, range=(0,1), density=True, alpha=0.6, color=PALETTE[4], label="pos")
                        plt.axvline(thr, color="#333", ls="--", lw=1, label=f"thr={thr:.3f}")
                        plt.xlabel("Predicted probability")
                        plt.ylabel("Density")
                        plt.title(f"Score distributions ({domain_label})")
                        plt.legend()
                        plt.tight_layout()
                        for ext in ("png","pdf"):
                            plt.savefig(os.path.join(figs_dir, f"score_dist_{domain_label}.{ext}"))
                        plt.close()
                    except Exception as e:
                        print(f"[warn] failed to save score distribution plot ({domain_label}): {e}")

                # Save for both domains
                try:
                    _save_eval_plots(yf_outer, pf_outer, thr_frame, "frame", fold_out)
                    _save_eval_plots(yv, pv, thr_video, "video", fold_out)
                except Exception as e:
                    print(f"[warn] failed to generate plots: {e}")

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
                    "selection": {"domain": args.select_domain, "metric": args.select_metric, "thr_metric": args.thr_metric, "joint": bool(args.joint_select), "tune_end": bool(args.tune_end), "min_specificity": args.min_specificity, "min_sensitivity": args.min_sensitivity, "min_classic_sensitivity": args.min_classic_sensitivity},
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
                    tf_train = get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)
                    full_train_ds = CadicaDataset(
                        comb_csv, metadata, tf_train,
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )
                    # Optional on-disk augmentation for full retrain
                    class_weights_full = None
                    if args.augment_pos_factor_neg_half is not None or args.aug_neg_percent is not None or args.augment:
                        try:
                            import albumentations as A  # type: ignore
                        except Exception:
                            A = None  # type: ignore

                        def _make_augmenter(img_size=None, special=False):
                            if A is None:
                                return None
                            if special:
                                ops = [
                                    A.HorizontalFlip(p=0.5),
                                    A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-10,10), p=0.7),
                                    A.RandomBrightnessContrast(0.15,0.15,p=0.5),
                                    A.MotionBlur(blur_limit=3,p=0.2),
                                ]
                                try:
                                    import inspect as _inspect
                                    params = _inspect.signature(A.GaussNoise.__init__).parameters
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
                            try:
                                use_px = 'translate_px' in __import__('inspect').signature(A.Affine.__init__).parameters
                            except Exception:
                                use_px = False
                            ops = [A.LongestMaxSize(max_size=img_size)] if img_size is not None else []
                            if use_px:
                                ops.append(A.Affine(translate_px={"x": (-25,25), "y": (-25,25)}, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                            else:
                                base = img_size if img_size is not None else 224
                                shift = 25.0/float(base)
                                ops.append(A.Affine(translate_percent=shift, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                            return A.Compose(ops)

                        def _apply_aug(img3, aug):
                            if aug is None:
                                # Basic affine using OpenCV only
                                h, w = img3.shape[:2]
                                cx, cy = w/2.0, h/2.0
                                angle = random.uniform(-25.0, 25.0)
                                scale = random.uniform(0.8, 1.7)
                                dx = random.uniform(-25.0, 25.0)
                                dy = random.uniform(-25.0, 25.0)
                                M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
                                M[0,2] += dx
                                M[1,2] += dy
                                return cv2.warpAffine(img3, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                            out = aug(image=img3)
                            return out['image']

                        # Build augmented CSV for full pool if requested
                        df_full = pd.read_csv(comb_csv).copy()
                        aug = _make_augmenter(img_size, special=args.special_aug) if 'img_size' in locals() else _make_augmenter(cfg.get("img_size", 224), special=args.special_aug)
                        fold_out = os.path.join(run_root, "final_full")
                        os.makedirs(fold_out, exist_ok=True)

                        if args.augment_pos_factor_neg_half is not None:
                            fpos = max(0, int(args.augment_pos_factor_neg_half))
                            pos_df = df_full[df_full['label'] == 1]
                            neg_df = df_full[df_full['label'] == 0]
                            pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                            pos_after = pos_ct + pos_ct * fpos
                            target_neg_total = int(np.ceil(0.5 * pos_after))
                            neg_needed = max(0, target_neg_total - neg_ct)
                            print(f"[final-augment-pos-neg-half] enabled | pos_factor={fpos} | target_neg_total={target_neg_total} | neg_needed={neg_needed} | albu={'on' if aug is not None else 'off'}")
                            aug_rows = []
                            # positives
                            if fpos > 0 and pos_ct > 0:
                                for r in pos_df.itertuples(index=False):
                                    p = int(r.patient); v = int(r.video); k = int(r.frame)
                                    sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                    base_max = int(sub['frame'].max()) if len(sub) else 0
                                    for i in range(fpos):
                                        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
                                        if img is None:
                                            continue
                                        img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                        aug_img = _apply_aug(img3, aug)
                                        aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                        out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                        os.makedirs(out_dir, exist_ok=True)
                                        new_k = base_max + 1 + i
                                        out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                        out_path = os.path.join(out_dir, out_name)
                                        if cv2.imwrite(out_path, aug_gray):
                                            aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 1})
                            # negatives to 0.5 of positives
                            if neg_needed > 0 and neg_ct > 0:
                                produced = 0
                                neg_list = list(neg_df.itertuples(index=False))
                                while produced < neg_needed and len(neg_list) > 0:
                                    for row in neg_list:
                                        if produced >= neg_needed:
                                            break
                                        p = int(row.patient); v = int(row.video)
                                        sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                        base_max = int(sub['frame'].max()) if len(sub) else 0
                                        extra_for_pv = sum(1 for rr in aug_rows if rr['patient']==p and rr['video']==v)
                                        new_k = base_max + extra_for_pv + 1
                                        img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                        if img is None:
                                            continue
                                        img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                        aug_img = _apply_aug(img3, aug)
                                        aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                        out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                        os.makedirs(out_dir, exist_ok=True)
                                        out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                        out_path = os.path.join(out_dir, out_name)
                                        if cv2.imwrite(out_path, aug_gray):
                                            aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0})
                                            produced += 1
                            if len(aug_rows) > 0:
                                aug_df = pd.concat([df_full, pd.DataFrame(aug_rows)], ignore_index=True)
                                aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                                is_aug = aug_df['img_path'].astype(str).str.contains(aug_tag)
                                def _ok_path(pth: str) -> bool:
                                    try:
                                        return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                    except Exception:
                                        return False
                                ok_mask = (~is_aug) | aug_df['img_path'].astype(str).apply(_ok_path)
                                aug_df = aug_df[ok_mask].reset_index(drop=True)
                                aug_csv = os.path.join(fold_out, 'final_train_aug.csv')
                                aug_df.to_csv(aug_csv, index=False)
                                full_train_ds = CadicaDataset(
                                    aug_csv, metadata, tf_train,
                                    selected_features=sel_nc,
                                    text_features=text_cols,
                                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                    stats=full_train_ds.stats,
                                )
                                if args.weighted:
                                    new_pos = int((aug_df['label'] == 1).sum())
                                    new_neg = int((aug_df['label'] == 0).sum())
                                    tot = float(new_pos + new_neg)
                                    w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                    w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                    class_weights_full = torch.tensor([w0, w1], dtype=torch.float32)
                        elif args.aug_neg_percent is not None:
                            r = float(args.aug_neg_percent)
                            if 0.0 < r < 1.0:
                                pos_df = df_full[df_full['label'] == 1]
                                neg_df = df_full[df_full['label'] == 0]
                                pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                                target_neg_total = int(np.ceil((r / max(1e-6, 1.0 - r)) * float(pos_ct)))
                                neg_needed = max(0, target_neg_total - neg_ct)
                                print(f"[final-augment-neg-percent] enabled | r={r:.3f} | target_neg_total={target_neg_total} | neg_needed={neg_needed} | albu={'on' if aug is not None else 'off'}")
                                aug_rows = []
                                if neg_needed > 0 and neg_ct > 0:
                                    produced = 0
                                    neg_list = list(neg_df.itertuples(index=False))
                                    while produced < neg_needed and len(neg_list) > 0:
                                        for row in neg_list:
                                            if produced >= neg_needed:
                                                break
                                            p = int(row.patient); v = int(row.video)
                                            sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                            base_max = int(sub['frame'].max()) if len(sub) else 0
                                            extra_for_pv = sum(1 for rr in aug_rows if rr['patient']==p and rr['video']==v)
                                            new_k = base_max + extra_for_pv + 1
                                            img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                            if img is None:
                                                continue
                                            img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                            aug_img = _apply_aug(img3, aug)
                                            aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                            out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                            os.makedirs(out_dir, exist_ok=True)
                                            out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                            out_path = os.path.join(out_dir, out_name)
                                            if cv2.imwrite(out_path, aug_gray):
                                                aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0})
                                                produced += 1
                                if len(aug_rows) > 0:
                                    aug_df = pd.concat([df_full, pd.DataFrame(aug_rows)], ignore_index=True)
                                    aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                                    is_aug = aug_df['img_path'].astype(str).str.contains(aug_tag)
                                    def _ok_path(pth: str) -> bool:
                                        try:
                                            return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                        except Exception:
                                            return False
                                    ok_mask = (~is_aug) | aug_df['img_path'].astype(str).apply(_ok_path)
                                    aug_df = aug_df[ok_mask].reset_index(drop=True)
                                    aug_csv = os.path.join(fold_out, 'final_train_aug.csv')
                                    aug_df.to_csv(aug_csv, index=False)
                                    full_train_ds = CadicaDataset(
                                        aug_csv, metadata, tf_train,
                                        selected_features=sel_nc,
                                        text_features=text_cols,
                                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                        stats=full_train_ds.stats,
                                    )
                                    if args.weighted:
                                        new_pos = int((aug_df['label'] == 1).sum())
                                        new_neg = int((aug_df['label'] == 0).sum())
                                        tot = float(new_pos + new_neg)
                                        w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                        w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                        class_weights_full = torch.tensor([w0, w1], dtype=torch.float32)
                        elif args.augment:
                            tgt_label = int(args.aug_label)
                            factor = float(args.aug_factor)
                            print(f"[final-augment] enabled | label={tgt_label} | match={bool(args.aug_match_count)} | factor={None if args.aug_match_count else factor} | max_per={int(args.aug_max_per_frame)} | albu={'on' if aug is not None else 'off'}")
                            src_df = df_full[df_full['label'] == tgt_label]
                            opp_df = df_full[df_full['label'] != tgt_label]
                            S = int(len(src_df)); O = int(len(opp_df))
                            if args.aug_match_count:
                                target_ct = max(0, O - S)
                            else:
                                target_ct = max(0, int(round(S * max(0.0, float(factor) - 1.0))))
                            aug_rows = []
                            produced = 0
                            for row in src_df.itertuples(index=False):
                                if produced >= target_ct:
                                    break
                                p = int(row.patient); v = int(row.video)
                                sub = df_full[(df_full['patient']==p) & (df_full['video']==v)]
                                base_max = int(sub['frame'].max()) if len(sub) else 0
                                per = 0
                                while per < int(args.aug_max_per_frame) and produced < target_ct:
                                    img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is None:
                                        break
                                    img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug)
                                    aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    extra_for_pv = sum(1 for rr in aug_rows if rr['patient']==p and rr['video']==v)
                                    new_k = base_max + extra_for_pv + 1
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    if cv2.imwrite(out_path, aug_gray):
                                        aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': tgt_label})
                                        produced += 1
                                        per += 1
                            if len(aug_rows) > 0:
                                aug_df = pd.concat([df_full, pd.DataFrame(aug_rows)], ignore_index=True)
                                aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                                is_aug = aug_df['img_path'].astype(str).str.contains(aug_tag)
                                def _ok_path(pth: str) -> bool:
                                    try:
                                        return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                    except Exception:
                                        return False
                                ok_mask = (~is_aug) | aug_df['img_path'].astype(str).apply(_ok_path)
                                aug_df = aug_df[ok_mask].reset_index(drop=True)
                                aug_csv = os.path.join(fold_out, 'final_train_aug.csv')
                                aug_df.to_csv(aug_csv, index=False)
                                full_train_ds = CadicaDataset(
                                    aug_csv, metadata, tf_train,
                                    selected_features=sel_nc,
                                    text_features=text_cols,
                                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                    stats=full_train_ds.stats,
                                )
                                if args.weighted:
                                    cnt_pos = int((aug_df['label'] == 1).sum())
                                    cnt_neg = int((aug_df['label'] == 0).sum())
                                    tot = float(cnt_pos + cnt_neg)
                                    w0 = tot / (2.0 * max(1.0, float(cnt_neg)))
                                    w1 = tot / (2.0 * max(1.0, float(cnt_pos)))
                                    class_weights_full = torch.tensor([w0, w1], dtype=torch.float32)
                    test_ds = CadicaDataset(
                        args.test_csv, metadata, get_val_tf(img_size),
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )

                    tr_full = DataLoader(full_train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    # non-shuffled train loader for metric curves
                    tr_full_eval = DataLoader(full_train_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=args.num_workers)
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

                    # Helpers for curves/figures
                    def _compute_train_ce_loss(model, loader, device, mode="late", clin_mlp=None, late_head=None, class_weights=None):
                        ce = torch.nn.CrossEntropyLoss(weight=(class_weights.to(device) if isinstance(class_weights, torch.Tensor) else class_weights))
                        total, n = 0.0, 0
                        model.eval()
                        if clin_mlp: clin_mlp.eval()
                        if late_head: late_head.eval()
                        with torch.no_grad():
                            for batch in loader:
                                x = batch["x"].to(device)
                                y = batch["y"].to(device)
                                clin = batch["clin"].to(device)
                                if mode == "late":
                                    feats, _ = model(x)
                                    logits = late_head(feats, clin_mlp(clin))
                                elif mode == "early":
                                    _, logits = model(x, clin)
                                else:
                                    _, logits = model(x)
                                loss = ce(logits, y)
                                bs = int(y.shape[0])
                                total += float(loss.item()) * bs
                                n += bs
                        return float(total / max(1, n))

                    # train full schedule and record curves
                    tr_acc_hist, tr_acc_video_hist, tr_loss_hist = [], [], []
                    best_loss = float('inf')
                    best_state = {"img": None, "clin": None, "late": None}
                    best_epoch = None
                    no_improve = 0
                    for epoch in range(1, epochs + 1):
                        print(f"[final-train] {backbone}/{fusion} | Epoch {epoch}/{epochs}")
                        train_one_epoch(
                            img_model_f, optimizer_f, tr_full, device,
                            mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f,
                            use_amp=(args.amp=="on"), class_weights=class_weights_full,
                        )
                        # training curves (frame acc + CE loss)
                        try:
                            fm_tr, y_tr, p_tr = evaluate(
                                img_model_f, tr_full_eval, device,
                                mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f, also_video=False,
                            )
                            tr_acc_hist.append(float(fm_tr.get("acc", float("nan"))))
                        except Exception:
                            tr_acc_hist.append(float("nan"))
                        # training video-level acc with tsmooth_max aggregator
                        try:
                            _, _, _, v_tr, vdf_tr = evaluate(
                                img_model_f, tr_full_eval, device,
                                mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f, also_video=True, video_reduce="tsmooth_max",
                            )
                            acc_v = None
                            if isinstance(v_tr, dict):
                                try:
                                    acc_v = float(v_tr.get("acc", float("nan")))
                                except Exception:
                                    acc_v = float("nan")
                            tr_acc_video_hist.append(acc_v if acc_v is not None else float("nan"))
                        except Exception:
                            tr_acc_video_hist.append(float("nan"))
                        try:
                            tr_loss = _compute_train_ce_loss(img_model_f, tr_full_eval, device, mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f, class_weights=class_weights_full)
                            tr_loss_hist.append(float(tr_loss))
                        except Exception:
                            tr_loss_hist.append(float("nan"))

                        # Early stopping on training loss (final_test_eval path)
                        if args.es_final:
                            cur_loss = tr_loss_hist[-1]
                            if np.isfinite(cur_loss) and (cur_loss + 1e-12) < best_loss:
                                best_loss = float(cur_loss)
                                best_epoch = int(epoch)
                                best_state["img"] = deepcopy(img_model_f.state_dict())
                                best_state["clin"] = deepcopy(clin_mlp_f.state_dict()) if clin_mlp_f else None
                                best_state["late"] = deepcopy(late_head_f.state_dict()) if late_head_f else None
                                no_improve = 0
                            else:
                                no_improve += 1
                                if no_improve >= int(args.es_final_patience):
                                    print(f"[final-train/es] early stop at epoch {epoch}; best_loss={best_loss:.6f} @ epoch {best_epoch}")
                                    break

                    # If ES used and a best state was captured, restore it before test evaluation
                    if args.es_final and best_state["img"] is not None:
                        try:
                            img_model_f.load_state_dict(best_state["img"])
                            if clin_mlp_f and best_state["clin"] is not None:
                                clin_mlp_f.load_state_dict(best_state["clin"])
                            if late_head_f and best_state["late"] is not None:
                                late_head_f.load_state_dict(best_state["late"])
                        except Exception:
                            pass

                    # evaluate on pristine test
                    fmet, yf_test, pf_test, vmet, vdf = evaluate(
                        img_model_f, te, device,
                        mode=fusion, clin_mlp=clin_mlp_f, late_head=late_head_f,
                        also_video=True, video_reduce=args.video_reduce,
                    )
                    # Cast video metrics excluding nested aggregator map ('_multi')
                    final_video = {}
                    for k, v in (vmet.items() if isinstance(vmet, dict) else []):
                        if k == "_multi":
                            continue
                        try:
                            final_video[k] = float(v)
                        except Exception:
                            pass
                    final_test_metrics = {
                        "frame": {k: float(v) for k, v in fmet.items()},
                        "video": final_video,
                    }
                    print("[final-test/frame]", {k: round(float(v), 4) for k, v in fmet.items()})
                    print("[final-test/video]", {k: round(float(v), 4) for k, v in final_video.items()})
                    # ---- Save final figures (training curves + test curves) ----
                    try:
                        PALETTE = ["#2078B3", "#FA9B9A", "#B2DF8A", "#FEBE70", "#FA9B9A"]
                        figs_dir = os.path.join(run_root, "final_full", "figs")
                        os.makedirs(figs_dir, exist_ok=True)

                        # Training curves
                        if plt is not None:
                            try:
                                epochs_axis = list(range(1, len(tr_acc_hist)+1))
                                plt.rcParams.update({
                                    'font.family': 'DejaVu Sans',
                                    'font.size': 12,
                                    'axes.titlesize': 12,
                                    'axes.labelsize': 12,
                                    'xtick.labelsize': 11,
                                    'ytick.labelsize': 11,
                                    'legend.fontsize': 11,
                                    'figure.titlesize': 12
                                })
                                fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
                                ax1.set_xlabel("Epoch")
                                ax1.set_ylabel("Train CE loss", color=PALETTE[1])
                                l_loss, = ax1.plot(epochs_axis, tr_loss_hist, color=PALETTE[1], lw=2.0, label="Loss")
                                ax1.tick_params(axis='y', labelcolor=PALETTE[1])
                                ax1.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)
                                ax1.minorticks_on(); ax1.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)
                                ax2 = ax1.twinx()
                                ax2.set_ylabel("Train acc", color=PALETTE[0])
                                l_acc_f, = ax2.plot(epochs_axis, tr_acc_hist, color=PALETTE[0], lw=2.2, label="Acc (frame)")
                                l_acc_v, = ax2.plot(epochs_axis, tr_acc_video_hist, color=PALETTE[2], lw=2.0, ls='--', label="Acc (video tsmooth_max)")
                                ax2.tick_params(axis='y', labelcolor=PALETTE[0])
                                # combined legend
                                lines = [l_loss, l_acc_f, l_acc_v]
                                labels = [ln.get_label() for ln in lines]
                                # Place legend outside plot area to avoid overlap
                                fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
                                # Leave top margin for legend
                                fig.tight_layout(rect=[0, 0, 1, 0.92])
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"training_curves.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] failed to save training curves: {_e}")

                        # Test curves: ROC, PR, sens-spec, confusion matrix
                        if plt is not None:
                            from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
                            import numpy as _np
                            try:
                                # ROC
                                fpr, tpr, _ = roc_curve(yf_test, pf_test)
                                plt.figure(figsize=(5,4), dpi=150)
                                plt.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"ROC (AUC={float(fmet.get('auc', 0.0)):.3f})")
                                plt.plot([0,1],[0,1], color="#888", ls="--", lw=1)
                                plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)"); plt.title("ROC (test)")
                                plt.grid(alpha=0.25); plt.legend(loc="lower right"); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"roc_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] ROC plot failed: {_e}")
                            try:
                                # PR
                                prec, rec, _ = precision_recall_curve(yf_test, pf_test)
                                plt.figure(figsize=(5,4), dpi=150)
                                plt.plot(rec, prec, color=PALETTE[3], lw=2, label=f"PR (AP={float(fmet.get('ap', 0.0)):.3f})")
                                plt.xlabel("Recall (Sensitivity)"); plt.ylabel("Precision (PPV)"); plt.title("PR (test)")
                                plt.grid(alpha=0.25); plt.legend(loc="lower left"); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"pr_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] PR plot failed: {_e}")
                            try:
                                # Calibration curve (confidence)
                                uniq = _np.unique(pf_test)
                                cand = _np.unique(_np.clip(_np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
                                sens_list, spec_list = [], []
                                for t in cand:
                                    md = metrics_from_probs(yf_test, pf_test, thr=float(t))
                                    sens_list.append(float(md.get("sensitivity", 0.0)))
                                    spec_list.append(float(md.get("specificity", 0.0)))
                                plt.figure(figsize=(5,4), dpi=150)
                                plt.plot(spec_list, sens_list, color=PALETTE[2], lw=2)
                                plt.xlabel("Specificity"); plt.ylabel("Sensitivity"); plt.title("Sensitivity–Specificity (test)")
                                plt.grid(alpha=0.25); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"sens_spec_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] Sens–Spec plot failed: {_e}")
                            try:
                                # DET curve (FNR vs FPR)
                                fpr, tpr, _ = roc_curve(yf_test, pf_test)
                                fnr = 1.0 - tpr
                                plt.figure(figsize=(5,4), dpi=150)
                                plt.plot(fpr, fnr, color=PALETTE[1], lw=2)
                                plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("False Negative Rate (FNR)"); plt.title("DET (test)")
                                plt.grid(alpha=0.25); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"det_frame_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] DET plot failed: {_e}")
                            try:
                                # Threshold sweeps (frame): sens, spec, bal_acc, f1, ppv, npv
                                uniq = _np.unique(pf_test)
                                cand = _np.unique(_np.clip(_np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
                                sens_l, spec_l, bal_l, f1_l, ppv_l, npv_l = [], [], [], [], [], []
                                for t in cand:
                                    md = metrics_from_probs(yf_test, pf_test, thr=float(t))
                                    sens_l.append(float(md.get("sensitivity", 0.0)))
                                    spec_l.append(float(md.get("specificity", 0.0)))
                                    bal_l.append(float(md.get("bal_acc", 0.0)))
                                    f1_l.append(float(md.get("f1", 0.0)))
                                    ppv_l.append(float(md.get("ppv", 0.0)))
                                    npv_l.append(float(md.get("npv", 0.0)))
                                plt.figure(figsize=(6,4), dpi=150)
                                plt.plot(cand, sens_l, color=PALETTE[0], lw=2, label="Sens")
                                plt.plot(cand, spec_l, color=PALETTE[1], lw=2, label="Spec")
                                plt.plot(cand, bal_l, color=PALETTE[2], lw=2, label="BalAcc")
                                plt.plot(cand, f1_l,  color=PALETTE[3], lw=2, label="F1")
                                plt.plot(cand, ppv_l, color=PALETTE[4], lw=2, label="PPV")
                                plt.xlabel("Threshold"); plt.ylabel("Metric value"); plt.title("Threshold sweep (frame)")
                                plt.grid(alpha=0.25); plt.legend(loc="lower left"); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"thr_sweep_frame.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] Threshold sweep (frame) failed: {_e}")
                            try:
                                # Calibration (frame) if available
                                if calibration_curve is not None:
                                    prob_true, prob_pred = calibration_curve(yf_test, pf_test, n_bins=10, strategy="uniform")
                                    plt.figure(figsize=(5,4), dpi=150)
                                    plt.plot([0,1],[0,1], color="#888", ls="--", lw=1)
                                    plt.plot(prob_pred, prob_true, color=PALETTE[0], lw=2)
                                    plt.xlabel("Predicted probability"); plt.ylabel("Observed prevalence"); plt.title("Calibration (frame)")
                                    plt.grid(alpha=0.25); plt.tight_layout()
                                    for ext in ("png","pdf","svg"):
                                        plt.savefig(os.path.join(figs_dir, f"calibration_frame_test.{ext}"))
                                    plt.close()
                            except Exception as _e:
                                print(f"[warn] Calibration (frame) failed: {_e}")
                            try:
                                # Score distributions (frame)
                                plt.figure(figsize=(5,4), dpi=150)
                                pos_scores = pf_test[yf_test==1]
                                neg_scores = pf_test[yf_test==0]
                                plt.hist(neg_scores, bins=30, range=(0,1), density=True, alpha=0.6, color=PALETTE[0], label="neg")
                                plt.hist(pos_scores, bins=30, range=(0,1), density=True, alpha=0.6, color=PALETTE[1], label="pos")
                                plt.axvline(0.5, color="#333", ls="--", lw=1, label="thr=0.5")
                                plt.xlabel("Predicted probability"); plt.ylabel("Density"); plt.title("Score distributions (frame)")
                                plt.legend(); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"score_dist_frame_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] Score dist (frame) failed: {_e}")

                            # ----- Video-level extended figures -----
                            try:
                                # Aggregator comparison (bar): auc, ap, bal_acc
                                vmulti = vmet.get("_multi") if isinstance(vmet, dict) else None
                                if isinstance(vmulti, dict):
                                    reducers = sorted(list(vmulti.keys()))
                                    aucs = [float(vmulti[r].get("auc", float("nan"))) for r in reducers]
                                    aps  = [float(vmulti[r].get("ap", float("nan"))) for r in reducers]
                                    bals = [float(vmulti[r].get("bal_acc", float("nan"))) for r in reducers]
                                    width = 0.25
                                    x = _np.arange(len(reducers))
                                    plt.figure(figsize=(8,4), dpi=150)
                                    plt.bar(x - width, aucs, width, color=PALETTE[0], label="AUC")
                                    plt.bar(x,         aps,  width, color=PALETTE[3], label="AP")
                                    plt.bar(x + width, bals, width, color=PALETTE[2], label="BalAcc")
                                    plt.xticks(x, reducers, rotation=30, ha='right')
                                    plt.ylabel("Score"); plt.title("Reducers comparison (video)")
                                    plt.grid(axis='y', alpha=0.25)
                                    plt.legend(); plt.tight_layout()
                                    for ext in ("png","pdf","svg"):
                                        plt.savefig(os.path.join(figs_dir, f"reducers_compare_video.{ext}"))
                                    plt.close()
                            except Exception as _e:
                                print(f"[warn] Reducers compare (video) failed: {_e}")
                            try:
                                # Threshold sweep (video) on selected reducer
                                yv = vdf["y_true"].values; pv = vdf["y_prob"].values
                                uniq = _np.unique(pv)
                                cand = _np.unique(_np.clip(_np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
                                sens_l, spec_l, bal_l, f1_l, ppv_l, npv_l = [], [], [], [], [], []
                                for t in cand:
                                    md = metrics_from_probs(yv, pv, thr=float(t))
                                    sens_l.append(float(md.get("sensitivity", 0.0)))
                                    spec_l.append(float(md.get("specificity", 0.0)))
                                    bal_l.append(float(md.get("bal_acc", 0.0)))
                                    f1_l.append(float(md.get("f1", 0.0)))
                                    ppv_l.append(float(md.get("ppv", 0.0)))
                                    npv_l.append(float(md.get("npv", 0.0)))
                                plt.figure(figsize=(6,4), dpi=150)
                                plt.plot(cand, sens_l, color=PALETTE[0], lw=2, label="Sens")
                                plt.plot(cand, spec_l, color=PALETTE[1], lw=2, label="Spec")
                                plt.plot(cand, bal_l, color=PALETTE[2], lw=2, label="BalAcc")
                                plt.plot(cand, f1_l,  color=PALETTE[3], lw=2, label="F1")
                                plt.plot(cand, ppv_l, color=PALETTE[4], lw=2, label="PPV")
                                plt.xlabel("Threshold"); plt.ylabel("Metric value"); plt.title("Threshold sweep (video)")
                                plt.grid(alpha=0.25); plt.legend(loc="lower left"); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"thr_sweep_video.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] Threshold sweep (video) failed: {_e}")
                            try:
                                # DET (video)
                                yv = vdf["y_true"].values; pv = vdf["y_prob"].values
                                fpr_v, tpr_v, _ = roc_curve(yv, pv)
                                fnr_v = 1.0 - tpr_v
                                plt.figure(figsize=(5,4), dpi=150)
                                plt.plot(fpr_v, fnr_v, color=PALETTE[1], lw=2)
                                plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("False Negative Rate (FNR)"); plt.title("DET (video)")
                                plt.grid(alpha=0.25); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"det_video_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] DET (video) failed: {_e}")
                            try:
                                # Calibration (video)
                                if calibration_curve is not None:
                                    yv = vdf["y_true"].values; pv = vdf["y_prob"].values
                                    prob_true, prob_pred = calibration_curve(yv, pv, n_bins=10, strategy="uniform")
                                    plt.figure(figsize=(5,4), dpi=150)
                                    plt.plot([0,1],[0,1], color="#888", ls="--", lw=1)
                                    plt.plot(prob_pred, prob_true, color=PALETTE[0], lw=2)
                                    plt.xlabel("Predicted probability"); plt.ylabel("Observed prevalence"); plt.title("Calibration (video)")
                                    plt.grid(alpha=0.25); plt.tight_layout()
                                    for ext in ("png","pdf","svg"):
                                        plt.savefig(os.path.join(figs_dir, f"calibration_video_test.{ext}"))
                                    plt.close()
                            except Exception as _e:
                                print(f"[warn] Calibration (video) failed: {_e}")
                            try:
                                # Confusion matrix at 0.5
                                yhat = (pf_test >= 0.5).astype(int)
                                cm = confusion_matrix(yf_test, yhat, labels=[0,1])
                                plt.figure(figsize=(4.5,4.0), dpi=150)
                                plt.imshow(cm, cmap="Blues")
                                plt.xticks([0,1],["Pred 0","Pred 1"]); plt.yticks([0,1],["True 0","True 1"])
                                for i in range(2):
                                    for j in range(2):
                                        plt.text(j, i, str(int(cm[i,j])), ha='center', va='center', color="#111", fontsize=12)
                                plt.title("Confusion Matrix (thr=0.5)"); plt.tight_layout()
                                for ext in ("png","pdf","svg"):
                                    plt.savefig(os.path.join(figs_dir, f"confusion_test.{ext}"))
                                plt.close()
                            except Exception as _e:
                                print(f"[warn] Confusion matrix plot failed: {_e}")
                    except Exception as _e:
                        print(f"[warn] failed to save final figures: {_e}")
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
