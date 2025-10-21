import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


def _split_positional_embeddings(base, num_patches: int):
    pos = base.encoder.pos_embedding  # (1, 1+Nref, D)
    cls_pe, patch_pe = pos[:, :1, :], pos[:, 1:, :]
    if patch_pe.size(1) == num_patches:
        return cls_pe, patch_pe
    d = patch_pe.shape[-1]
    n_ref = patch_pe.shape[1]
    h_ref = w_ref = int(math.sqrt(n_ref))
    h_new = w_new = int(math.sqrt(num_patches))
    if h_ref * w_ref != n_ref or h_new * w_new != num_patches:
        if num_patches < n_ref:
            return cls_pe, patch_pe[:, :num_patches, :]
        else:
            repeat = patch_pe[:, -1:, :].repeat(1, num_patches - n_ref, 1)
            return cls_pe, torch.cat([patch_pe, repeat], dim=1)
    patch_2d = patch_pe[0].transpose(0, 1).reshape(1, d, h_ref, w_ref)
    patch_2d = F.interpolate(patch_2d, size=(h_new, w_new), mode="bicubic", align_corners=False)
    patch_new = patch_2d.reshape(1, d, h_new * w_new).transpose(1, 2)
    return cls_pe, patch_new


class _BaseViTMid(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vit_b_16(weights=w)
        d = base.heads.head.in_features
        base.heads = nn.Identity()
        self.backbone = base
        self.d_model = d
        self.head = nn.Linear(d, out_dim)

    def _patch_embeddings(self, x_img: torch.Tensor):
        return self.backbone._process_input(x_img)  # (B, N, D)


class ViTMidFusionCLSPeriodicXAttn(_BaseViTMid):
    """
    Periodically update CLS via cross-attention to a metadata token every M blocks.
    """
    def __init__(self, tab_dim: int, M: int = 2, nheads: int = 8, dropout: float = 0.1,
                 pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        self.meta_proj = nn.Sequential(
            nn.Linear(tab_dim, self.d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.d_model, self.d_model)
        )
        self.cross_attn = nn.MultiheadAttention(self.d_model, num_heads=nheads, batch_first=True)
        self.cls_gate = nn.Parameter(torch.tensor(1.0))
        self.M = max(1, int(M))

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)
        N = patches.size(1)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, patches], dim=1)
        x = x + torch.cat([cls_pe, patch_pe], dim=1)

        meta = self.meta_proj(clin_vec).unsqueeze(1)  # (B,1,D)

        x = self.backbone.encoder.dropout(x)
        for i, blk in enumerate(self.backbone.encoder.layers):
            if i % self.M == 0:
                cls = x[:, :1, :]
                cls_upd, _ = self.cross_attn(query=cls, key=meta, value=meta)
                x = torch.cat([cls + self.cls_gate * cls_upd, x[:, 1:, :]], dim=1)
            x = blk(x)
        x = self.backbone.encoder.ln(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits


class ViTMidFusionResidualFiLM(_BaseViTMid):
    """
    Residual-stream FiLM applied pre-block at every layer: x = x * (1+gamma) + beta.
    """
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        self.film = nn.Sequential(nn.Linear(tab_dim, self.d_model * 2))

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)
        N = patches.size(1)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, patches], dim=1)
        x = x + torch.cat([cls_pe, patch_pe], dim=1)

        gamma, beta = self.film(clin_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        x = self.backbone.encoder.dropout(x)
        for blk in self.backbone.encoder.layers:
            x = x * (1 + gamma) + beta
            x = blk(x)
        x = self.backbone.encoder.ln(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits


class _Adapter(nn.Module):
    def __init__(self, d_model: int, r: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, r)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d_model)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class ViTMidFusionMetaAdapter(_BaseViTMid):
    """
    Adapter after each transformer block, scaled by metadata-conditioned factors.
    x <- x + alpha_i * Adapter(x)
    """
    def __init__(self, tab_dim: int, r: int = 64, pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        self.adapter = _Adapter(self.d_model, r)
        L = len(self.backbone.encoder.layers)
        # Produce per-layer scaling coefficients from metadata
        self.alpha_mlp = nn.Linear(tab_dim, L)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)
        N = patches.size(1)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, patches], dim=1)
        x = x + torch.cat([cls_pe, patch_pe], dim=1)

        alphas = torch.tanh(self.alpha_mlp(clin_vec))  # (B, L) in [-1,1]

        x = self.backbone.encoder.dropout(x)
        for i, blk in enumerate(self.backbone.encoder.layers):
            x = blk(x)
            a = alphas[:, i].view(B, 1, 1)
            x = x + a * self.adapter(x)
        x = self.backbone.encoder.ln(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits

