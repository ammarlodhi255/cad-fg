import os, sys, cv2, torch, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.viz.gradcam import GradCAM
from src.viz.overlays import overlay_heatmap
from src.dataio.dataset import load_metadata_any, build_meta_vectors
from src.dataio.transforms import get_val_tf  # not strictly needed here, but handy
from src.models.resnet_backbone import ResNetBackbone
from src.models.early_fusion_patient import ResNetEarlyFusionPatient
from src.models.effnet_backbone import EfficientNetB0Backbone, EfficientNetB0EarlyFusionPatient
from src.models.late_fusion import TabularMLP, LateFusionClassifier
from src.models.inception_backbone import InceptionV3Backbone, InceptionV3EarlyFusionPatient

def get_gradcam_target_name(backbone_name: str) -> str:
    """
    Return a string path to a good last conv layer for Grad-CAM.
    This matches src.viz.gradcam.GradCAM which expects target_layer_name (string).
    """
    b = backbone_name.lower()
    if b == "resnet50":
        # final bottleneck conv; torchvision names it layer4.2.conv3
        return "backbone.layer4.2"
    elif b == "resnet18":
        # BasicBlock (no conv3), so last conv is conv2 in the last block
        return "backbone.layer4.1"       # safe block to hook; your GradCAM resolves module by string
    elif b == "effnet_b0":
        # last MBConv's last conv: backbone.features.6.0.block.2
        return "backbone.features.6.0.block.2"
    elif b == "inception_v3":
        # last bottleneck conv; torchvision names it auxlogits
        return "backbone.auxlogits"
    else:
        raise ValueError(f"No CAM target defined for backbone={backbone_name}")

def main(ckpt, outdir="outputs/figures", n=12):
    os.makedirs(outdir, exist_ok=True)
    # NOTE: we keep weights_only=False because our checkpoint includes more than raw tensors
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fusion, tab_dim, cfg = ck["fusion"], ck["tab_dim"], ck["cfg"]
    backbone = cfg.get("backbone", "resnet50")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sample N test frames
    df = pd.read_csv("outputs/index/test.csv").sample(n, random_state=0)
    meta_df = load_metadata_any(cfg.get("metadata_path","data/CADICA/metadata.xlsx"))
    # respect the same feature subset as training (if any)
    selected = cfg.get("metadata_features", None)
    meta_map = build_meta_vectors(meta_df, selected_features=selected)

    # ---- rebuild model(s) exactly like eval ----
    target_name = get_gradcam_target_name(backbone)

    if fusion == "imageonly":
        if backbone == "resnet50":
            model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            model = EfficientNetB0Backbone(pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        model.load_state_dict(ck["model"])

        def logits_fn(x, clin):
            # metadata ignored for imageonly
            _, logits = model(x)
            return logits

    elif fusion == "late":
        # backbone only (no classifier inside), separate MLP + fusion head
        if backbone == "resnet50":
            model = ResNetBackbone(pretrained=False).to(device)
            img_feat_dim = 2048
        elif backbone == "effnet_b0":
            model = EfficientNetB0Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        model.load_state_dict(ck["model"])

        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        clin_mlp.load_state_dict(ck["clin_mlp"])
        late_head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        late_head.load_state_dict(ck["head"])

        def logits_fn(x, clin):
            feats, _ = model(x)            # (B, img_feat_dim)
            z = clin_mlp(clin)             # (B, 32)
            return late_head(feats, z)     # logits

    elif fusion == "early":
        if backbone == "resnet50":
            model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        model.load_state_dict(ck["model"])

        def logits_fn(x, clin):
            _, logits = model(x, clin)
            return logits
    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")

    model.eval()
    cam = GradCAM(model, target_layer_name=target_name)

    for _, r in df.iterrows():
        # load grayscale, expand to 3ch
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        x = torch.tensor(np.stack([img, img, img], axis=0)[None].astype(np.float32) / 255.0).to(device)
        clin = torch.tensor(meta_map[int(r.patient)], dtype=torch.float32)[None].to(device)

        # forward + CAM for class 1 (positive)
        def forward_logits():
            return logits_fn(x, clin)

        logits, cam_t = cam(forward_logits, class_idx=1)
        # upsample CAM to input size
        cam_up = torch.nn.functional.interpolate(
            cam_t, size=x.shape[2:], mode="bilinear", align_corners=False
        )[0, 0].detach().cpu().numpy()   # <-- detach before numpy()

        out = overlay_heatmap(img, cam_up)
        cv2.imwrite(os.path.join(outdir, f"p{r.patient}_v{r.video}_{int(r.frame):05d}.png"), out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    main(args.ckpt, n=args.n)