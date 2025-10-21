#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visual CAMs for ViT-B/16 checkpoints (imageonly or late-fusion).
- Supports EigenCAM (class-agnostic) and class-specific: GradCAM, GradCAM++, XGradCAM, LayerCAM.
- Rebuilds metadata exactly like training (numeric+categorical and optional TEXT).
- Normalizes inputs like ImageNet and resizes to ViT image_size (usually 224).
- Overlays the CAM on the original-resolution grayscale frame.

Usage:
  python cam_vit.py --ckpt outputs/checkpoints/all/all-vit_b16/best_vit_b16_late.pt \
                    --n 12 --outdir outputs/figures_vit --method layercam
"""

import os, sys, cv2, torch, numpy as np, pandas as pd
from typing import Optional

# repo imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.viz.overlays import overlay_heatmap
from src.dataio.dataset import load_metadata_any, build_meta_vectors, build_text_embed_map
from src.models.vit_backbone import ViTBackbone  # imageonly / late

# pytorch-grad-cam
from pytorch_grad_cam import (
    EigenCAM, GradCAM, GradCAMPlusPlus, XGradCAM, LayerCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------- Normalization ----------------
def _normalize_img_imsnet(x_chw: np.ndarray) -> np.ndarray:
    """x_chw: (3,H,W) in [0,1] -> normalize with ImageNet stats."""
    x = x_chw.transpose(1, 2, 0)   # -> (H,W,3)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)

# ---------------- ViT helpers ----------------
def _resolve_vit_target_layers(model_for_cam: torch.nn.Module):
    """Pick a suitable encoder norm layer inside torchvision ViT for CAM."""
    base = getattr(model_for_cam, "backbone", model_for_cam)
    enc = getattr(base, "encoder", None)
    if enc is None or not hasattr(enc, "layers"):
        raise RuntimeError("Could not find ViT encoder layers in model.backbone")
    last = enc.layers[-1]
    for name in ("ln_1", "norm1"):
        if hasattr(last, name):
            return [getattr(last, name)]
    for m in last.modules():
        if m.__class__.__name__.lower().startswith("layernorm"):
            return [m]
    raise RuntimeError("Could not resolve a target layer in last ViT block")

def make_vit_reshape_transform(model_for_cam: torch.nn.Module):
    """Return a reshape_transform function bound to this model."""
    def _vit_reshape_transform(activations):
        # activations: [B, tokens, C] (norm output from last encoder block)
        B, T, C = activations.shape
        base = getattr(model_for_cam, "backbone", model_for_cam)
        image_size = int(getattr(base, "image_size", 224))
        patch = getattr(base, "patch_size", 16)
        if isinstance(patch, (tuple, list)):
            patch = patch[0]
        h = w = image_size // int(patch)
        # drop CLS token
        activations_wo_cls = activations[:, 1:, :]  # [B, H*W, C]
        return activations_wo_cls.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
    return _vit_reshape_transform

# ---------------- Metadata ----------------
def _build_meta_map_like_training(cfg: dict, tab_dim: int):
    """Recreate the clinical vectors exactly like training (with/without TEXT)."""
    meta_df = load_metadata_any(cfg.get("metadata_path", "data/CADICA/metadata.xlsx"))

    used = cfg.get("metadata_features_used", None)
    if used is not None:
        sel_nc    = {"numeric": used.get("numeric"),
                     "categorical": used.get("categorical") or used.get("categororical")}
        text_cols = used.get("text") or []
    else:
        sel       = cfg.get("metadata_features", {})
        sel_nc    = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
        text_cols = sel.get("text") or []

    text_map = None
    if len(text_cols) > 0:
        text_map = build_text_embed_map(
            meta_df,
            patient_col="patient",
            text_cols=text_cols,
            model_name=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
        )

    meta_map = build_meta_vectors(meta_df, selected_features=sel_nc, text_embed_map=text_map)

    # sanity
    any_pid = next(iter(meta_map))
    dim0 = len(meta_map[any_pid])
    assert dim0 == tab_dim, f"Metadata dim {dim0} != ckpt tab_dim {tab_dim}"
    return meta_map

# ---------------- Wrapper so CAM sees logits (not (feats, logits)) ----------------
class ViTForCAM(torch.nn.Module):
    """
    Wraps your ViTBackbone so forward(x) returns logits tensor only.
    For late fusion, call .set_clin(clin) before forward.
    Exposes .backbone to keep target_layers resolution intact.
    """
    def __init__(self, vit: ViTBackbone, fusion: str, clin_mlp=None, late_head=None):
        super().__init__()
        self.model = vit
        self.fusion = fusion
        self.clin_mlp = clin_mlp
        self.late_head = late_head
        self._clin = None
        # Expose attributes used by helpers:
        self.backbone = vit.backbone
        self.feat_dim = vit.feat_dim

    def set_clin(self, clin: torch.Tensor):
        self._clin = clin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion == "imageonly":
            _, logits = self.model(x)
            return logits
        else:
            assert self._clin is not None, "Call set_clin(clin) before forward in late fusion."
            feats, _ = self.model(x)
            z = self.clin_mlp(self._clin)
            logits = self.late_head(feats, z)
            return logits

# ---------------- CAM factory ----------------
def make_cam(method: str, model, target_layers, reshape_transform):
    method = method.lower()
    if method == "eigen":    return EigenCAM(model, target_layers, reshape_transform=reshape_transform), False
    if method == "grad":     return GradCAM(model, target_layers, reshape_transform=reshape_transform), True
    if method == "grad++":   return GradCAMPlusPlus(model, target_layers, reshape_transform=reshape_transform), True
    if method == "xgrad":    return XGradCAM(model, target_layers, reshape_transform=reshape_transform), True
    if method == "layercam": return LayerCAM(model, target_layers, reshape_transform=reshape_transform), True
    raise ValueError("Unknown --method")

# ---------------- Main ----------------
def main(ckpt: str, outdir: str = "outputs/figures_vit", n: int = 12, method: str = "layercam",
         override_class: Optional[int] = None):
    os.makedirs(outdir, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fusion, tab_dim, cfg = ck["fusion"], ck["tab_dim"], ck["cfg"]
    backbone = cfg.get("backbone", "vit_b16").lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if backbone != "vit_b16":
        raise ValueError(f"This script is for ViT-B/16 only, got backbone={backbone}")
    if fusion not in ("imageonly", "late"):
        raise ValueError(f"Supports only fusion=['imageonly','late'], got {fusion}")

    # sample test frames
    df = pd.read_csv("outputs/index/test.csv").sample(n, random_state=0)

    # metadata
    meta_map = _build_meta_map_like_training(cfg, tab_dim)

    # build model (raw)
    if fusion == "imageonly":
        vit = ViTBackbone(pretrained=False, out_dim=2).to(device)
    else:  # late
        vit = ViTBackbone(pretrained=False).to(device)
    vit.load_state_dict(ck["model"])
    vit.eval()

    # late fusion heads
    clin_mlp = late_head = None
    if fusion == "late":
        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        img_dim = vit.feat_dim
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        clin_mlp.load_state_dict(ck["clin_mlp"])
        late_head = LateFusionClassifier(img_dim=img_dim, clin_dim=32, out_dim=2).to(device)
        late_head.load_state_dict(ck["head"])
        clin_mlp.eval(); late_head.eval()

    # Wrap so grad-cam gets logits tensor
    cam_model = ViTForCAM(vit, fusion, clin_mlp, late_head).to(device)
    cam_model.eval()

    # CAM setup
    target_layers = _resolve_vit_target_layers(cam_model)
    reshape_transform = make_vit_reshape_transform(cam_model)
    cam, need_targets = make_cam(method, cam_model, target_layers, reshape_transform)

    # ViT input size
    target_size = getattr(cam_model.backbone, "image_size", 224)

    with cam:
        for _, r in df.iterrows():
            img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[warn] could not read {r.img_path}; skipping"); continue
            H, W = img.shape[:2]

            # preprocess
            img_in = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            x_np = np.stack([img_in, img_in, img_in], axis=0).astype(np.float32) / 255.0
            x_np = _normalize_img_imsnet(x_np)
            x = torch.from_numpy(x_np[None]).to(device)

            # clinical vector (late fusion only)
            pid = int(r.patient)
            clin_np = np.asarray(meta_map.get(pid, np.zeros((tab_dim,), dtype=np.float32)), dtype=np.float32)
            clin = torch.from_numpy(clin_np[None]).to(device)
            cam_model.set_clin(clin)  # no-op for imageonly

            # targets for class-specific methods
            targets = None
            if need_targets:
                if override_class is not None:
                    targets = [ClassifierOutputTarget(int(override_class))]
                else:
                    with torch.no_grad():
                        pred_logits = cam_model(x)
                        pred_cls = int(pred_logits.argmax(dim=1).item())
                    targets = [ClassifierOutputTarget(pred_cls)]

            # CAM (returns H_in x W_in float in [0,1]); enable smoothing
            grayscale_cam = cam(
                input_tensor=x,
                targets=targets,
                eigen_smooth=(method.lower() == "eigen"),
                aug_smooth=True
            )[0]

            # normalize CAM to [0,1] and overlay at original size
            cam_map = torch.tensor(grayscale_cam).clamp(min=0)
            cam_map = cam_map / (cam_map.max() + 1e-6)
            cam_resized = cv2.resize(cam_map.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)

            out = overlay_heatmap(img, cam_resized)
            out_path = os.path.join(outdir, f"p{int(r.patient)}_v{int(r.video)}_{int(r.frame):05d}.png")
            cv2.imwrite(out_path, out)
            print(f"[saved] {out_path}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ViT-B/16 checkpoint (.pt)")
    ap.add_argument("--outdir", default="outputs/figures_vit")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--method", default="layercam",
                    choices=["eigen", "grad", "grad++", "xgrad", "layercam"],
                    help="ViT often looks better with layercam / grad++ / xgrad.")
    ap.add_argument("--class", dest="override_class", type=int, default=None,
                    help="If using class-specific methods, optional target class index (default: prediction).")
    args = ap.parse_args()
    main(args.ckpt, args.outdir, args.n, args.method, args.override_class)