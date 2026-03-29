import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNeXtBaseMidFusionResidualFiLM(nn.Module):
    """
    Mid-fusion Residual FiLM for ConvNeXt-Base (applied before pooling).
    """
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_base(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        self.feat_dim = 1024
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


class ConvNeXtBaseMidFusionMetaAdapter(nn.Module):
    """
    Mid-fusion Meta-Adapter for ConvNeXt-Base (post-pooling adapter with metadata gating).
    """
    def __init__(self, tab_dim: int, r: int = 64, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_base(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        self.feat_dim = 1024
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


class ConvNeXtBaseMidFusionCLSCrossAttn(nn.Module):
    """
    Mid-fusion CLS-like cross-attention for ConvNeXt-Base.
    """
    def __init__(self, tab_dim: int, nheads: int = 8, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_base(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        self.feat_dim = 1024
        self.meta_proj = nn.Linear(tab_dim, self.feat_dim)
        self.cross_attn = nn.MultiheadAttention(self.feat_dim, num_heads=nheads, batch_first=True)
        self.cls_gate = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        x = self.backbone.features(x_img)
        B, C, H, W = x.shape
        cls = x.mean(dim=(2,3), keepdim=False).unsqueeze(1)
        meta = self.meta_proj(clin_vec).unsqueeze(1)
        cls_upd, _ = self.cross_attn(query=cls, key=meta, value=meta)
        delta = (cls_upd - cls) * self.cls_gate
        x = x + delta.transpose(1,2).unsqueeze(-1)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits
