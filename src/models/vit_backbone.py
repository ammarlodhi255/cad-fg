import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for image-only / late fusion.
    Returns (feats, logits).
    Feat dim = 768 for vit_b_16.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vit_b_16(weights=w)

        # vit_b_16 has: encoder -> heads (linear classifier)
        in_feats = base.heads.head.in_features
        self.feat_dim = in_feats

        base.heads = nn.Identity()  # strip classifier
        self.backbone = base
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x):
        # forward until penultimate
        return self.backbone(x)

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class ViTEarlyFusionPatient(nn.Module):
    """
    Early fusion: concat metadata channels to RGB before patch embedding.
    """
    def __init__(self, tab_dim: int, c_pat: int = 8, pretrained: bool = True,
                 out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vit_b_16(weights=w)

        # PatchEmbed is base.conv_proj (Conv2d)
        old = base.conv_proj
        new_in = old.in_channels + c_pat
        new_conv = nn.Conv2d(
            new_in, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :old.in_channels] = old.weight
        nn.init.kaiming_normal_(new_conv.weight[:, old.in_channels:], nonlinearity="linear")
        base.conv_proj = new_conv

        in_feats = base.heads.head.in_features
        self.feat_dim = in_feats
        base.heads = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img, clin_vec):
        B, _, H, W = x_img.shape
        ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)
        feats = self.backbone(x)    # (B, 768)
        logits = self.head(feats)
        return feats, logits