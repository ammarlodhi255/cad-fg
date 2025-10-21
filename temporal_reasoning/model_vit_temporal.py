import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class FrameEncoderViT(nn.Module):
    """Encode frames with ViT and return per-frame CLS embeddings.

    Input:  x (B, T, C, H, W)
    Output: feats (B, T, D)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vit_b_16(weights=w)
        self.d_model = base.heads.head.in_features  # 768
        base.heads = nn.Identity()
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        # TorchVision ViT: forward returns (B*T, D) pooled CLS embedding
        feats = self.backbone(x)  # (B*T, D)
        feats = feats.view(B, T, -1)
        return feats


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TemporalAggregator(nn.Module):
    """Temporal transformer encoder over per-frame features.

    Returns pooled clip feature (B, D).
    """
    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1, pool: str = 'cls'):
        super().__init__()
        self.pool = pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == 'cls' else None
        self.pos = TemporalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        nn.init.trunc_normal_(self.cls_token, std=0.02) if self.cls_token is not None else None

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        if self.cls_token is not None:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, 1+T, D)
        x = self.pos(x)
        z = self.encoder(x, src_key_padding_mask=None)  # we assume dense clips, no mask
        if self.cls_token is not None:
            pooled = z[:, 0, :]
        else:
            pooled = z.mean(dim=1)
        return pooled


class TemporalViTClassifier(nn.Module):
    """
    ViT frame encoder + temporal transformer head for video classification.

    forward(x):
      x: (B, T, C, H, W)
      returns: (feats, logits) where feats is pooled video feature and logits is (B, 2)
    """
    def __init__(self, pretrained_vit: bool = True, n_layers: int = 2, n_heads: int = 8, pool: str = 'cls', out_dim: int = 2):
        super().__init__()
        self.frame_enc = FrameEncoderViT(pretrained=pretrained_vit)
        d = self.frame_enc.d_model
        self.temporal = TemporalAggregator(d_model=d, n_layers=n_layers, n_heads=n_heads, pool=pool)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,C,H,W)
        f = self.frame_enc(x)  # (B,T,D)
        v = self.temporal(f)   # (B,D)
        logits = self.head(v)  # (B,2)
        return v, logits

