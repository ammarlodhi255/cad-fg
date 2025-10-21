import torch, pandas as pd, numpy as np, yaml, os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader
from src.dataio.transforms import get_val_tf
from src.dataio.dataset import CadicaDataset
from src.models.resnet_backbone import ResNetBackbone
from src.models.late_fusion import TabularMLP, LateFusionClassifier
from src.models.early_fusion_patient import ResNetEarlyFusionPatient
from src.trainloop.engine import evaluate
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

def main(ckpt):
    import torch, pandas as pd, numpy as np
    from torch.utils.data import DataLoader
    from src.dataio.transforms import get_val_tf
    from src.dataio.dataset import CadicaDataset
    from src.models.resnet_backbone import ResNetBackbone
    from src.models.late_fusion import TabularMLP, LateFusionClassifier
    from src.models.early_fusion_patient import ResNetEarlyFusionPatient
    from src.trainloop.engine import evaluate
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    ck = torch.load(ckpt, map_location="cpu")
    fusion = ck["fusion"]
    tab_dim = ck["tab_dim"]
    cfg = ck["cfg"]  # this contains the exact metadata_features used for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- dataset: use the SAME metadata subset as training ---
    # te_csv = "outputs/index/test.csv"
    # metadata_path = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    # selected = cfg.get("metadata_features", None)  # <- ablation subset used in training

    # ds = CadicaDataset(
    #     te_csv,
    #     metadata_path,
    #     get_val_tf(cfg["img_size"]),
    #     selected_features=selected
    # )


    # print("[meta] (TEST) Using features:",
    #   ds.get_feature_groups(),
    #   f"tab_dim={ds.tab_dim}")
    # print("[meta] (TEST) Expanded order:", ds.get_feature_names())

    # # you can lower workers if your system warned earlier
    # ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    te_csv = "outputs/index/test.csv"
    metadata_path = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")

    # Use exactly the same subsets as training (from cfg)
    sel = cfg.get("metadata_features", {})
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = None
    # Prefer the explicit “*_used” block if you saved it; else fall back to cfg:
    mf_used = cfg.get("metadata_features_used", None)
    if mf_used is not None:
        text_cols = mf_used.get("text", None)
    else:
        text_cols = sel.get("text", None)

    ds = CadicaDataset(
        te_csv,
        metadata_path,
        get_val_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )

    print("[meta] (TEST) Using features:", ds.get_feature_groups(), f"tab_dim={ds.tab_dim}")
    print("[meta] (TEST) Expanded order:", ds.get_feature_names())

    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    backbone = cfg.get("backbone", "resnet50").lower()

    if fusion == "late":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            model = ResNetBackbone(pretrained=False).to(device)
            img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            model = ResNet18Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 512
            img_feat_dim = model.feat_dim  # 1536
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            model = EfficientNetB0Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1280
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            model = InceptionV3Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 2048
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone
            model = ConvNeXtBackbone(variant="tiny", pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 768
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            model = ConvNeXtBackbone(variant="base", pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1024
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone
            model = MobileNetV3LargeBackbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 960
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone
            model = MobileNetV2Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1280
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone
            model = DenseNet121Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1024
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone
            model = AlexNetBackbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 9216
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone
            model = ViTBackbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 768
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7Backbone
            model = EfficientNetB7Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 2560
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
            model = EfficientNetB0Backbone(pretrained=False, cbam=True).to(device)
            img_feat_dim = model.feat_dim  # 1280
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM
            model = ConvNeXtBaseBackboneCBAM(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1024
        elif backbone == "metafusionnet_base":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            model = MetaFusionConvNeXtBackbone(variant="base", pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1024
        elif backbone == "metafusionnet_tiny":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            model = MetaFusionConvNeXtBackbone(variant="tiny", pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        model.load_state_dict(ck["model"]); clin_mlp.load_state_dict(ck["clin_mlp"]); head.load_state_dict(ck["head"])

    elif fusion == "early":
        if backbone == "resnet50":
            from src.models.early_fusion_patient import ResNetEarlyFusionPatient
            model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18EarlyFusionPatient
            model = ResNet18EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            from src.models.effnet_b3_backbone import EfficientNetB3EarlyFusionPatient
            model = EfficientNetB3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"],
                                                 pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0EarlyFusionPatient
            model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3EarlyFusionPatient
            model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            model = ConvNeXtEarlyFusionPatient(variant="tiny", tab_dim=tab_dim, c_pat=cfg["pat_channels"],
                                           pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            model = ConvNeXtEarlyFusionPatient(variant="base", tab_dim=tab_dim, c_pat=cfg["pat_channels"],
                                           pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3LargeEarlyFusionPatient
            model = MobileNetV3LargeEarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2EarlyFusionPatient
            model = MobileNetV2EarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "densenet201":
            from src.models.densenet_backbone import DenseNet201EarlyFusionPatient
            model = DenseNet201EarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121EarlyFusionPatient
            model = DenseNet121EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"],
                                              pretrained=False, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetEarlyFusionPatient
            model = AlexNetEarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTEarlyFusionPatient
            model = ViTEarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7EarlyFusionPatient
            model = EfficientNetB7EarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0EarlyFusionPatient
            model = EfficientNetB0EarlyFusionPatient(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2, cbam=True
            ).to(device)
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseEarlyFusionPatientCBAM
            model = ConvNeXtBaseEarlyFusionPatientCBAM(
                tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2
            ).to(device)
        elif backbone == "metafusionnet_base":
            from src.models.metafusionnet_backbone import MetaFusionNetEarlyFusionPatient
            model = MetaFusionNetEarlyFusionPatient(
                variant="base",
                tab_dim=tab_dim,
                pretrained=False,
                out_dim=2
            ).to(device)
        elif backbone == "metafusionnet_tiny":
            from src.models.metafusionnet_backbone import MetaFusionNetEarlyFusionPatient
            model = MetaFusionNetEarlyFusionPatient(
                variant="tiny",
                tab_dim=tab_dim,
                pretrained=False,
                out_dim=2
            ).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        clin_mlp = head = None
        model.load_state_dict(ck["model"])

    elif fusion == "imageonly":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            model = ResNet18Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            from src.models.effnet_b3_backbone import EfficientNetB3Backbone
            model = EfficientNetB3Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            model = EfficientNetB0Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            model = InceptionV3Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone
            model = ConvNeXtBackbone(variant="tiny", pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            model = ConvNeXtBackbone(variant="base", pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone
            model = MobileNetV3LargeBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone
            model = MobileNetV2Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet201":
            from src.models.densenet_backbone import DenseNet201Backbone
            model = DenseNet201Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone
            model = DenseNet121Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone
            model = AlexNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone
            model = ViTBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7Backbone
            model = EfficientNetB7Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
            model = EfficientNetB0Backbone(pretrained=False, out_dim=2, cbam=True).to(device)
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM
            model = ConvNeXtBaseBackboneCBAM(pretrained=False, out_dim=2).to(device)
        elif backbone == "metafusionnet_base":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            model = MetaFusionConvNeXtBackbone(variant="base", pretrained=False, out_dim=2).to(device)
        elif backbone == "metafusionnet_tiny":
            from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
            model = MetaFusionConvNeXtBackbone(variant="tiny", pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        clin_mlp = head = None
        model.load_state_dict(ck["model"])

    else:
        raise ValueError("Unknown fusion mode in checkpoint")


    frame_metrics, y_frame, p_frame, video_metrics, video_df = evaluate(
    model, ld, device, mode=fusion, clin_mlp=clin_mlp, late_head=head,
    also_video=True, video_reduce="max"   # options: max, mean, median, topk2, topk3, logit_mean, ew_mean, tsmooth_mean, tsmooth_max, all
)

    print("=== Frame-level metrics ===")
    for k, v in frame_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\n=== Video-level metrics ===")
    for k, v in video_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    # optional: save the per-video probabilities/labels
    video_df.to_csv("outputs/test_video_probs.csv", index=False)


    # helpful print so you know exactly what subset was used
    if sel:
        n_num = len(sel.get("numeric", []))
        n_cat = len(sel.get("categorical", []))
        print(f"[info] evaluated with metadata subset: numeric={n_num}, categorical={n_cat}, tab_dim(checkpoint)={tab_dim}")
    else:
        print(f"[info] evaluated with ALL metadata features, tab_dim(checkpoint)={tab_dim}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.ckpt)
