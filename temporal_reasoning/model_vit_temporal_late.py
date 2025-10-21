from typing import Tuple

import torch
import torch.nn as nn

from temporal_reasoning.model_vit_temporal import FrameEncoderViT, TemporalAggregator


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, out_dim: int = 32, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, out_dim), nn.ReLU(), nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionHead(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int, out_dim: int = 2, hid: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(img_dim + clin_dim, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, out_dim)
        )

    def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
        z = torch.cat([img_feat, clin_feat], dim=-1)
        return self.proj(z)


class TemporalViTLateFusionClassifier(nn.Module):
    """Image feature temporalization + late fusion with tabular MLP.

    forward(x, clin):
      x: (B,T,C,H,W), clin: (B, D_tab)
      returns: (feats, logits)
    """
    def __init__(self, tab_dim: int, n_layers: int = 2, n_heads: int = 8, pool: str = 'cls', out_dim: int = 2,
                 clin_hid: int = 64, clin_out: int = 32):
        super().__init__()
        self.frame_enc = FrameEncoderViT(pretrained=True)
        d = self.frame_enc.d_model
        self.temporal = TemporalAggregator(d_model=d, n_layers=n_layers, n_heads=n_heads, pool=pool)
        self.clin_mlp = TabularMLP(in_dim=tab_dim, hid=clin_hid, out_dim=clin_out)
        self.late_head = LateFusionHead(img_dim=d, clin_dim=clin_out, out_dim=out_dim)
        self.requires_clin = True

    def forward(self, x: torch.Tensor, clin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.frame_enc(x)    # (B,T,D)
        v = self.temporal(f)     # (B,D)
        c = self.clin_mlp(clin)  # (B,clin_out)
        logits = self.late_head(v, c)
        return v, logits

