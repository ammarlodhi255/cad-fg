from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.vit_early_fusion_variants import (
    ViTEarlyFusionMetaToken,
    ViTEarlyFusionFiLM,
    ViTEarlyFusionPrompt,
    ViTEarlyFusionCLSCrossAttn,
    ViTMidFusionDeepPrompt,
)
from src.models.vit_backbone import ViTEarlyFusionPatient
try:
    from src.models.vit384_backbone import ViTEarlyFusionPatient384  # optional if timm installed
except Exception:  # pragma: no cover
    ViTEarlyFusionPatient384 = None
from temporal_reasoning.model_vit_temporal import TemporalAggregator
try:
    from src.models.vit_mid_fusion_variants import (
        ViTMidFusionCLSPeriodicXAttn,
        ViTMidFusionResidualFiLM,
        ViTMidFusionMetaAdapter,
    )
except Exception:
    ViTMidFusionCLSPeriodicXAttn = None
    ViTMidFusionResidualFiLM = None
    ViTMidFusionMetaAdapter = None


_VARIANTS = {
    "meta_token": ViTEarlyFusionMetaToken,
    "film": ViTEarlyFusionFiLM,
    "prompt": ViTEarlyFusionPrompt,
    "cls_cross": ViTEarlyFusionCLSCrossAttn,
    "deep_prompt": ViTMidFusionDeepPrompt,
    # patient channels early-fusion (concat channels before patch embedding)
    "patient": ViTEarlyFusionPatient,
    "patient_channels": ViTEarlyFusionPatient,
    "channels": ViTEarlyFusionPatient,
}


class FrameEncoderViTEarlyFusion(nn.Module):
    """Wraps ViT early fusion variants to encode per-frame CLS features.

    forward(x, clin):
      x: (B,T,C,H,W), clin: (B, D_tab)
      returns: (B,T,D_vit)
    """
    def __init__(self, variant: str, tab_dim: int, *, c_pat: int = 8, img_size: int = 224,
                 prompt_K: int = 4, cls_cross_heads: int = 8, mid_M: int = 2, mid_adapter_r: int = 64):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown ViT early fusion variant: {variant}")
        # Select implementation
        if variant in ("patient", "patient_channels", "channels"):
            if img_size and img_size >= 384 and ViTEarlyFusionPatient384 is not None:
                self.enc = ViTEarlyFusionPatient384(tab_dim=tab_dim, c_pat=c_pat, pretrained=True, out_dim=2)
            else:
                self.enc = ViTEarlyFusionPatient(tab_dim=tab_dim, c_pat=c_pat, pretrained=True, out_dim=2)
        else:
            if variant == "prompt":
                self.enc = ViTEarlyFusionPrompt(tab_dim=tab_dim, K=int(prompt_K), pretrained=True, out_dim=2)
            elif variant == "cls_cross":
                self.enc = ViTEarlyFusionCLSCrossAttn(tab_dim=tab_dim, pretrained=True, out_dim=2, nheads=int(cls_cross_heads))
            elif variant == "deep_prompt":
                self.enc = ViTMidFusionDeepPrompt(tab_dim=tab_dim, K=int(prompt_K), pretrained=True, out_dim=2)
            elif variant in ("mid_cls_xattn", "mid_periodic_xattn") and ViTMidFusionCLSPeriodicXAttn is not None:
                self.enc = ViTMidFusionCLSPeriodicXAttn(tab_dim=tab_dim, M=int(mid_M), nheads=int(cls_cross_heads), pretrained=True, out_dim=2)
            elif variant in ("mid_film_res", "mid_residual_film") and ViTMidFusionResidualFiLM is not None:
                self.enc = ViTMidFusionResidualFiLM(tab_dim=tab_dim, pretrained=True, out_dim=2)
            elif variant in ("mid_adapter", "mid_meta_adapter") and ViTMidFusionMetaAdapter is not None:
                self.enc = ViTMidFusionMetaAdapter(tab_dim=tab_dim, r=int(mid_adapter_r), pretrained=True, out_dim=2)
            else:
                self.enc = _VARIANTS[variant](tab_dim=tab_dim, pretrained=True, out_dim=2)
        # Robustly determine embedding dimension
        d = getattr(self.enc, 'd_model', None)
        if d is None:
            d = getattr(self.enc, 'feat_dim', None)
        if d is None:
            try:
                with torch.no_grad():
                    dummy_x = torch.zeros(1, 3, int(img_size), int(img_size))
                    dummy_c = torch.zeros(1, int(tab_dim))
                    feats, _ = self.enc(dummy_x, dummy_c)
                    d = int(feats.shape[-1])
            except Exception:
                d = 768  # sane default for ViT-B/16
        self.d_model = int(d)

    def forward(self, x: torch.Tensor, clin: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            feats_or_cls, _ = self.enc(x[:, t, :, :, :], clin)
            cls_out = feats_or_cls
            feats.append(cls_out)
        feats = torch.stack(feats, dim=1)  # (B,T,D)
        return feats


class TemporalViTEarlyFusionClassifier(nn.Module):
    """Early-fusion ViT per-frame encoder + temporal transformer head.

    forward(x, clin):
      x: (B,T,C,H,W), clin: (B, D_tab)
      returns: (feats, logits) where feats is pooled video feature and logits is (B, 2)
    """
    def __init__(self, variant: str, tab_dim: int, n_layers: int = 2, n_heads: int = 8, pool: str = 'cls', out_dim: int = 2, *,
                 c_pat: int = 8, img_size: int = 224, prompt_K: int = 4, cls_cross_heads: int = 8, mid_M: int = 2, mid_adapter_r: int = 64):
        super().__init__()
        self.frame_enc = FrameEncoderViTEarlyFusion(variant=variant, tab_dim=tab_dim, c_pat=c_pat, img_size=img_size,
                                                   prompt_K=prompt_K, cls_cross_heads=cls_cross_heads,
                                                   mid_M=mid_M, mid_adapter_r=mid_adapter_r)
        d = self.frame_enc.d_model
        self.temporal = TemporalAggregator(d_model=d, n_layers=n_layers, n_heads=n_heads, pool=pool)
        self.head = nn.Linear(d, out_dim)
        self.requires_clin = True

    def forward(self, x: torch.Tensor, clin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.frame_enc(x, clin)  # (B,T,D)
        v = self.temporal(f)         # (B,D)
        logits = self.head(v)        # (B,2)
        return v, logits
