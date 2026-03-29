import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtTinyMidFusionResidualFiLM(nn.Module):
    """
    Mid-fusion Residual FiLM for ConvNeXt-Tiny.
    - Apply FiLM modulation to the last feature map before global pooling:
      x <- x * (1 + gamma) + beta, where gamma,beta are derived from metadata.
    """
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        self.feat_dim = 768
        self.film = nn.Linear(tab_dim, 2 * self.feat_dim)
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        x = self.backbone.features(x_img)
        gamma, beta = self.film(clin_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        x = x * (1 + gamma) + beta
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits


class _Adapter(nn.Module):
    def __init__(self, d_model: int, r: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, r)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d_model)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class ConvNeXtTinyMidFusionMetaAdapter(nn.Module):
    """
    Mid-fusion Meta-Adapter for ConvNeXt-Tiny.
    - After global pooling, apply an adapter on the pooled feature vector:
      feats <- feats + alpha * Adapter(feats), where alpha is metadata-derived.
    """
    def __init__(self, tab_dim: int, r: int = 64, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        self.feat_dim = 768
        self.adapter = _Adapter(self.feat_dim, r)
        self.alpha_mlp = nn.Linear(tab_dim, 1)
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        x = self.backbone.features(x_img)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        alpha = torch.tanh(self.alpha_mlp(clin_vec)).view(-1, 1)
        feats = feats + alpha * self.adapter(feats)
        logits = self.head(feats)
        return feats, logits


class ConvNeXtTinyMidFusionCLSCrossAttn(nn.Module):
    """
    Mid-fusion CLS-like cross-attention for ConvNeXt-Tiny.
    - Pool spatial map to a CLS-like vector, cross-attend to a meta token,
      broadcast the update back to the feature map via a gated residual.
    """
    def __init__(self, tab_dim: int, nheads: int = 8, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        self.feat_dim = 768
        self.meta_proj = nn.Linear(tab_dim, self.feat_dim)
        self.cross_attn = nn.MultiheadAttention(self.feat_dim, num_heads=nheads, batch_first=True)
        self.cls_gate = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        x = self.backbone.features(x_img)  # (B,C,H,W)
        B, C, H, W = x.shape
        cls = x.mean(dim=(2,3), keepdim=False).unsqueeze(1)  # (B,1,C)
        meta = self.meta_proj(clin_vec).unsqueeze(1)         # (B,1,C)
        cls_upd, _ = self.cross_attn(query=cls, key=meta, value=meta)
        delta = (cls_upd - cls) * self.cls_gate              # (B,1,C)
        x = x + delta.transpose(1,2).unsqueeze(-1)           # broadcast to (B,C,1,1)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits
