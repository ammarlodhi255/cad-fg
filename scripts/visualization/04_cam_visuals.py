import os, sys, cv2, torch, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.viz.gradcam import GradCAM
from src.viz.overlays import overlay_heatmap
from src.dataio.dataset import load_metadata_any, build_meta_vectors, build_text_embed_map
from src.models.resnet_backbone import ResNetBackbone
from src.models.early_fusion_patient import ResNetEarlyFusionPatient
from src.models.effnet_backbone import EfficientNetB0Backbone, EfficientNetB0EarlyFusionPatient
from src.models.inception_backbone import InceptionV3Backbone, InceptionV3EarlyFusionPatient

# new backbones
from src.models.resnet18_backbone import ResNet18Backbone, ResNet18EarlyFusionPatient
from src.models.effnet_b3_backbone import EfficientNetB3Backbone, EfficientNetB3EarlyFusionPatient
from src.models.convnext_backbone import ConvNeXtBackbone, ConvNeXtEarlyFusionPatient
from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone, MobileNetV3LargeEarlyFusionPatient
from src.models.mobilenetv2_backbone import MobileNetV2Backbone, MobileNetV2EarlyFusionPatient
from src.models.densenet_backbone import DenseNet201Backbone, DenseNet201EarlyFusionPatient
from src.models.densenet121_backbone import DenseNet121Backbone, DenseNet121EarlyFusionPatient
from src.models.alexnet_backbone import AlexNetBackbone, AlexNetEarlyFusionPatient
from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone, EfficientNetB0EarlyFusionPatient
from src.models.effnet_b7_backbone import EfficientNetB7Backbone, EfficientNetB7EarlyFusionPatient
from src.models.vit_backbone import ViTBackbone, ViTEarlyFusionPatient
from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone, MetaFusionNetEarlyFusionPatient

def get_gradcam_target_name(backbone_name: str) -> str:
    b = backbone_name.lower()
    if b == "resnet50": return "backbone.layer4.2"
    if b == "resnet18": return "backbone.layer4.1"
    if b == "effnet_b0": return "backbone.features.6.0.block.2"
    if b == "effnet_b3": return "backbone.features.8.0.block.2"
    if b == "inception_v3": return "backbone.Mixed_7c.branch3x3dbl_3.conv"
    if b in ("convnext_tiny","convnext_base","convnext_base_cbam",
             "metafusionnet_base","metafusionnet_tiny"):
        return "backbone.features.7.0.dwconv"
    if b == "mobilenet_v3_large": return "backbone.features.16.0"
    if b == "mobilenet_v2": return "backbone.features.18.0"
    if b == "densenet201": return "backbone.features.denseblock4.denselayer32.conv2"
    if b == "densenet121": return "backbone.features.denseblock4.denselayer16.conv2"
    if b == "alexnet": return "backbone.features.10"
    if b == "vit_b16": return "backbone.conv_proj"
    if b == "effnet_b7": return "backbone.features.8.0.block.2"
    if b == "effnet_b0_cbam": return "backbone.features.6.0.block.2"
    raise ValueError(f"No CAM target defined for backbone={backbone_name}")

def main(ckpt, outdir="outputs/figures", n=12):
    os.makedirs(outdir, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fusion, tab_dim, cfg = ck["fusion"], ck["tab_dim"], ck["cfg"]
    backbone = cfg.get("backbone", "resnet50").lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sample N test frames
    df = pd.read_csv("outputs/index/test.csv").sample(n, random_state=0)

    # ----- build metadata EXACTLY as in training (numeric + categorical + optional TEXT) -----
    meta_df = load_metadata_any(cfg.get("metadata_path", "data/CADICA/metadata.xlsx"))

    used = cfg.get("metadata_features_used", None)
    if used is not None:
        sel_nc    = {"numeric": used.get("numeric"), "categorical": used.get("categorical")}
        text_cols = used.get("text") or []        # ← coerce None→[]
    else:
        sel       = cfg.get("metadata_features", {})
        sel_nc    = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
        text_cols = sel.get("text") or []         # ← coerce None→[]

    text_map = None
    if len(text_cols) > 0:
        text_map = build_text_embed_map(
            meta_df,
            patient_col="patient",
            text_cols=text_cols,
            model_name=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
        )

    meta_map = build_meta_vectors(
        meta_df,
        selected_features=sel_nc,
        text_embed_map=text_map,   # None if no text → numeric+categorical only
    )

    # sanity check to fail fast if mismatch with checkpoint
    _pid0 = next(iter(meta_map))
    _dim0 = len(meta_map[_pid0])
    assert _dim0 == tab_dim, (
        f"Metadata vector length {_dim0} != checkpoint tab_dim {tab_dim}. "
        "Make sure features/text match training."
    )

    # ----- build model skeleton -----
    if fusion == "imageonly":
        if backbone == "resnet50":
            model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            model = ResNet18Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            model = EfficientNetB0Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            model = EfficientNetB3Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone in ("convnext_tiny","convnext_base"):
            model = ConvNeXtBackbone(variant="tiny" if backbone.endswith("tiny") else "base",
                                     pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            model = InceptionV3Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            model = MobileNetV3LargeBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            model = MobileNetV2Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet201":
            model = DenseNet201Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet121":
            model = DenseNet121Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "alexnet":
            model = AlexNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "vit_b16":
            model = ViTBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            model = EfficientNetB7Backbone(pretrained=False).to(device)
        elif backbone == "effnet_b0_cbam":
            model = EfficientNetB0Backbone(pretrained=False, cbam=True).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        model.load_state_dict(ck["model"])

        def logits_fn(x, clin):  # metadata ignored
            _, logits = model(x); return logits

    elif fusion == "late":
        if backbone == "resnet50":
            model = ResNetBackbone(pretrained=False).to(device); img_dim = 2048
        elif backbone == "resnet18":
            model = ResNet18Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "effnet_b0":
            model = EfficientNetB0Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "effnet_b3":
            model = EfficientNetB3Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone in ("convnext_tiny","convnext_base"):
            model = ConvNeXtBackbone(variant="tiny" if backbone.endswith("tiny") else "base",
                                     pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "inception_v3":
            model = InceptionV3Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "mobilenet_v3_large":
            model = MobileNetV3LargeBackbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "mobilenet_v2":
            model = MobileNetV2Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "densenet201":
            model = DenseNet201Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "densenet121":
            model = DenseNet121Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "alexnet":
            model = AlexNetBackbone(pretrained=False, out_dim=2).to(device); img_dim = model.feat_dim
        elif backbone == "vit_b16":
            model = ViTBackbone(pretrained=False, out_dim=2).to(device); img_dim = model.feat_dim
        elif backbone == "effnet_b7":
            model = EfficientNetB7Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "effnet_b0_cbam":
            model = EfficientNetB0Backbone(pretrained=False, cbam=True).to(device); img_dim = model.feat_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        model.load_state_dict(ck["model"])

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        clin_mlp.load_state_dict(ck["clin_mlp"])
        late_head = LateFusionClassifier(img_dim=img_dim, clin_dim=32, out_dim=2).to(device)
        late_head.load_state_dict(ck["head"])

        def logits_fn(x, clin):
            feats, _ = model(x)
            z = clin_mlp(clin)
            return late_head(feats, z)

    elif fusion == "early":
        if backbone == "resnet50":
            model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            model = ResNet18EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            model = EfficientNetB3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone in ("convnext_tiny","convnext_base"):
            model = ConvNeXtEarlyFusionPatient(variant="tiny" if backbone.endswith("tiny") else "base",
                                               tab_dim=tab_dim, c_pat=cfg["pat_channels"],
                                               pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            model = MobileNetV3LargeEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            model = MobileNetV2EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet201":
            model = DenseNet201EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet121":
            model = DenseNet121EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "alexnet":
            model = AlexNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "vit_b16":
            model = ViTEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            model = EfficientNetB7EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2, cbam=True).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        model.load_state_dict(ck["model"])

        def logits_fn(x, clin):
            _, logits = model(x, clin); return logits
    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")

    model.eval()

    # CAM target + runner
    target_name = get_gradcam_target_name(backbone)
    cam = GradCAM(model, target_layer_name=target_name)

    for _, r in df.iterrows():
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]

        # Resize to model's expected size (ViT: 224)
        backbone_obj = getattr(model, "backbone", model)
        target_size = getattr(backbone_obj, "image_size", 224)
        img_in = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        x = torch.tensor(
            np.stack([img_in, img_in, img_in], axis=0)[None].astype(np.float32) / 255.0
        ).to(device)

        # Build clinical vector from the precomputed map
        pid = int(r.patient)
        if pid not in meta_map:
            # rare, but avoid crash if patient missing
            clin_np = np.zeros((tab_dim,), dtype=np.float32)
        else:
            clin_np = np.asarray(meta_map[pid], dtype=np.float32)
        assert clin_np.shape[0] == tab_dim, f"clin dim {clin_np.shape[0]} != tab_dim {tab_dim}"
        clin = torch.tensor(clin_np)[None].to(device)

        def forward_logits(): return logits_fn(x, clin)

        logits, cam_t = cam(forward_logits, class_idx=1)

        # upsample CAM to original image size for overlay
        cam_up = torch.nn.functional.interpolate(
            cam_t, size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].detach().cpu().numpy()

        out = overlay_heatmap(img, cam_up)
        cv2.imwrite(os.path.join(outdir, f"p{r.patient}_v{int(r.video)}_{int(r.frame):05d}.png"), out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    main(args.ckpt, n=args.n)