# import os, sys, cv2, torch, numpy as np, pandas as pd
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.viz.gradcam import GradCAM
# from src.viz.overlays import overlay_heatmap
# from src.dataio.dataset import load_metadata_any, build_meta_vectors
# from src.dataio.transforms import get_val_tf  # not strictly needed here, but handy
# from src.models.resnet_backbone import ResNetBackbone
# from src.models.early_fusion_patient import ResNetEarlyFusionPatient
# from src.models.effnet_backbone import EfficientNetB0Backbone, EfficientNetB0EarlyFusionPatient
# from src.models.late_fusion import TabularMLP, LateFusionClassifier
# from src.models.inception_backbone import InceptionV3Backbone, InceptionV3EarlyFusionPatient

# def get_gradcam_target_name(backbone_name: str) -> str:
#     """
#     Return a string path to a good last conv layer for Grad-CAM.
#     This matches src.viz.gradcam.GradCAM which expects target_layer_name (string).
#     """
#     b = backbone_name.lower()
#     if b == "resnet50":
#         # final bottleneck conv; torchvision names it layer4.2.conv3
#         return "backbone.layer4.2"
#     elif b == "resnet18":
#         # BasicBlock (no conv3), so last conv is conv2 in the last block
#         return "backbone.layer4.1"       # safe block to hook; your GradCAM resolves module by string
#     elif b == "effnet_b3":
#         # robust pick across torchvision versions: last stage's first block last conv
#         return "backbone.features.-1.0.block.2"   # if your resolver supports -1 indexing
#         # If not, safer hardcode:
#         # return "backbone.features.8.0.block.2"
#     elif b == "effnet_b0":
#         # last MBConv's last conv: backbone.features.6.0.block.2
#         return "backbone.features.6.0.block.2"
#     elif b == "inception_v3":
#         # last bottleneck conv; torchvision names it auxlogits
#         return "backbone.auxlogits"
#     else:
#         raise ValueError(f"No CAM target defined for backbone={backbone_name}")

# def main(ckpt, outdir="outputs/figures", n=12):
#     os.makedirs(outdir, exist_ok=True)
#     # NOTE: we keep weights_only=False because our checkpoint includes more than raw tensors
#     ck = torch.load(ckpt, map_location="cpu", weights_only=False)
#     fusion, tab_dim, cfg = ck["fusion"], ck["tab_dim"], ck["cfg"]
#     backbone = cfg.get("backbone", "resnet50")
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # sample N test frames
#     df = pd.read_csv("outputs/index/test.csv").sample(n, random_state=0)
#     meta_df = load_metadata_any(cfg.get("metadata_path","data/CADICA/metadata.xlsx"))
#     # respect the same feature subset as training (if any)
#     selected = cfg.get("metadata_features", None)
#     meta_map = build_meta_vectors(meta_df, selected_features=selected)

#     # ---- rebuild model(s) exactly like eval ----
#     target_name = get_gradcam_target_name(backbone)

#     if fusion == "imageonly":
#         if backbone == "resnet50":
#             model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
#         elif backbone == "effnet_b0":
#             model = EfficientNetB0Backbone(pretrained=False, out_dim=2).to(device)
#         else:
#             raise ValueError(f"Unknown backbone: {backbone}")
#         model.load_state_dict(ck["model"])

#         def logits_fn(x, clin):
#             # metadata ignored for imageonly
#             _, logits = model(x)
#             return logits

#     elif fusion == "late":
#         # backbone only (no classifier inside), separate MLP + fusion head
#         if backbone == "resnet50":
#             model = ResNetBackbone(pretrained=False).to(device)
#             img_feat_dim = 2048
#         elif backbone == "effnet_b0":
#             model = EfficientNetB0Backbone(pretrained=False).to(device)
#             img_feat_dim = model.feat_dim  # 1280
#         else:
#             raise ValueError(f"Unknown backbone: {backbone}")
#         model.load_state_dict(ck["model"])

#         clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
#         clin_mlp.load_state_dict(ck["clin_mlp"])
#         late_head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
#         late_head.load_state_dict(ck["head"])

#         def logits_fn(x, clin):
#             feats, _ = model(x)            # (B, img_feat_dim)
#             z = clin_mlp(clin)             # (B, 32)
#             return late_head(feats, z)     # logits

#     elif fusion == "early":
#         if backbone == "resnet50":
#             model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
#         elif backbone == "effnet_b0":
#             model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
#         elif backbone == "inception_v3":
#             model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
#         else:
#             raise ValueError(f"Unknown backbone: {backbone}")
#         model.load_state_dict(ck["model"])

#         def logits_fn(x, clin):
#             _, logits = model(x, clin)
#             return logits
#     else:
#         raise ValueError(f"Unknown fusion mode: {fusion}")

#     model.eval()
#     cam = GradCAM(model, target_layer_name=target_name)

#     for _, r in df.iterrows():
#         # load grayscale, expand to 3ch
#         img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
#         x = torch.tensor(np.stack([img, img, img], axis=0)[None].astype(np.float32) / 255.0).to(device)
#         clin = torch.tensor(meta_map[int(r.patient)], dtype=torch.float32)[None].to(device)

#         # forward + CAM for class 1 (positive)
#         def forward_logits():
#             return logits_fn(x, clin)

#         logits, cam_t = cam(forward_logits, class_idx=1)
#         # upsample CAM to input size
#         cam_up = torch.nn.functional.interpolate(
#             cam_t, size=x.shape[2:], mode="bilinear", align_corners=False
#         )[0, 0].detach().cpu().numpy()   # <-- detach before numpy()

#         out = overlay_heatmap(img, cam_up)
#         cv2.imwrite(os.path.join(outdir, f"p{r.patient}_v{r.video}_{int(r.frame):05d}.png"), out)

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--n", type=int, default=12)
#     args = ap.parse_args()
#     main(args.ckpt, n=args.n)


import os, sys, cv2, torch, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.viz.gradcam import GradCAM
from src.viz.overlays import overlay_heatmap
from src.dataio.dataset import load_metadata_any, build_meta_vectors
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
# ---------------- CAM target selection ----------------

def get_gradcam_target_name(backbone_name: str) -> str:
    """
    Return a string path to a good deep conv layer for Grad-CAM.
    These string paths are resolved by your GradCAM helper.
    """
    b = backbone_name.lower()

    # ResNets
    if b == "resnet50":
        return "backbone.layer4.2"         # last bottleneck block
    if b == "resnet18":
        return "backbone.layer4.1"         # last basic block (has conv2 inside)

    # EfficientNet B0 / B3 (indices depend on torchvision; these are stable picks)
    if b == "effnet_b0":
        return "backbone.features.6.0.block.2"   # last MBConv's last conv
    if b == "effnet_b3":
        # final stage index is 8 in most torchvision builds; first block, last conv:
        return "backbone.features.8.0.block.2"

    # Inception v3: deep mixed module, 3x3dbl branch final conv
    if b == "inception_v3":
        return "backbone.Mixed_7c.branch3x3dbl_3.conv"

    # ConvNeXt Tiny/Base: use the last stage, last block's depthwise conv
    if b == "convnext_tiny":
        # features.7 is last stage; pick first block's dwconv (stable across versions)
        return "backbone.features.7.0.dwconv"
    if b == "convnext_base":
        return "backbone.features.7.0.dwconv"
    if b == "mobilenet_v3_large":
        # final ConvBNActivation before avgpool; first module is conv
        return "backbone.features.16.0"  # if your resolver expects the Conv module
        # If your resolver wants deeper: "backbone.features.16.0" usually points to Conv
    if b == "mobilenet_v2":
        # final conv before avgpool (1x1 conv). Works across most torchvision builds.
        return "backbone.features.18.0"
    if b == "densenet201":
        # Last dense block, last dense layer, 3x3 conv
        return "backbone.features.denseblock4.denselayer32.conv2"
    if b == "densenet121":
        return "backbone.features.denseblock4.denselayer16.conv2"
    if b == "alexnet":
        return "backbone.features.10"   # last Conv2d before ReLU/MaxPool
    if b == "vit_b16":
        return "backbone.conv_proj"
    if b == "effnet_b7":
        # final stage (index 8), first block (0), last conv in block
        return "backbone.features.8.0.block.2"
    if b == "effnet_b0_cbam":
        return "backbone.features.6.0.block.2"
    if b == "convnext_base_cbam":
        return "backbone.features.7.0.dwconv"
    raise ValueError(f"No CAM target defined for backbone={backbone_name}")

# optional: fallback if your GradCAM can accept a module handle
def get_last_conv_module(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found for CAM")
    return last

# ---------------- main ----------------

def main(ckpt, outdir="outputs/figures", n=12):
    os.makedirs(outdir, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fusion, tab_dim, cfg = ck["fusion"], ck["tab_dim"], ck["cfg"]
    backbone = cfg.get("backbone", "resnet50").lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sample N test frames
    df = pd.read_csv("outputs/index/test.csv").sample(n, random_state=0)

    # metadata (respect feature subset used at train time)
    meta_df = load_metadata_any(cfg.get("metadata_path", "data/CADICA/metadata.xlsx"))
    selected = cfg.get("metadata_features", None)
    meta_map = build_meta_vectors(meta_df, selected_features=selected)

    # build model skeleton
    if fusion == "imageonly":
        if backbone == "resnet50":
            model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            model = ResNet18Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            model = EfficientNetB0Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            model = EfficientNetB3Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone in ("convnext_tiny", "convnext_base"):
            variant = "tiny" if backbone.endswith("tiny") else "base"
            model = ConvNeXtBackbone(variant=variant, pretrained=False, out_dim=2).to(device)
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
            _, logits = model(x)
            return logits

    elif fusion == "late":
        # backbone w/o classifier
        if backbone == "resnet50":
            model = ResNetBackbone(pretrained=False).to(device); img_dim = 2048
        elif backbone == "resnet18":
            model = ResNet18Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "effnet_b0":
            model = EfficientNetB0Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone == "effnet_b3":
            model = EfficientNetB3Backbone(pretrained=False).to(device); img_dim = model.feat_dim
        elif backbone in ("convnext_tiny", "convnext_base"):
            variant = "tiny" if backbone.endswith("tiny") else "base"
            model = ConvNeXtBackbone(variant=variant, pretrained=False).to(device); img_dim = model.feat_dim
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
        elif backbone in ("convnext_tiny", "convnext_base"):
            variant = "tiny" if backbone.endswith("tiny") else "base"
            model = ConvNeXtEarlyFusionPatient(variant=variant, tab_dim=tab_dim, c_pat=cfg["pat_channels"],
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
            _, logits = model(x, clin)
            return logits
    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")

    model.eval()

    # choose CAM target
    target_name = get_gradcam_target_name(backbone)
    cam = GradCAM(model, target_layer_name=target_name)

    for _, r in df.iterrows():
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        x = torch.tensor(np.stack([img, img, img], axis=0)[None].astype(np.float32) / 255.0).to(device)
        clin = torch.tensor(meta_map[int(r.patient)], dtype=torch.float32)[None].to(device)

        def forward_logits():
            return logits_fn(x, clin)

        logits, cam_t = cam(forward_logits, class_idx=1)
        cam_up = torch.nn.functional.interpolate(
            cam_t, size=x.shape[2:], mode="bilinear", align_corners=False
        )[0, 0].detach().cpu().numpy()

        out = overlay_heatmap(img, cam_up)
        cv2.imwrite(os.path.join(outdir, f"p{r.patient}_v{r.video}_{int(r.frame):05d}.png"), out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    main(args.ckpt, n=args.n)