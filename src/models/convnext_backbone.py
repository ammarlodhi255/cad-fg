import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, convnext_base,
    ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
)

# --------------------------- Utilities ---------------------------

def _load_convnext(variant: str, pretrained: bool):
    variant = variant.lower()
    if variant == "tiny":
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = convnext_tiny(weights=w)
        feat_dim = 768
    elif variant == "base":
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        m = convnext_base(weights=w)
        feat_dim = 1024
    else:
        raise ValueError(f"Unknown ConvNeXt variant: {variant}")
    return m, feat_dim

def _first_conv_module(m: nn.Module) -> nn.Conv2d:
    """
    Torchvision ConvNeXt stem lives at m.features[0][0] and is a Conv2d(3, C, k=4, s=4)
    """
    stem = m.features[0]
    conv = stem[0]
    if not isinstance(conv, nn.Conv2d):
        # very defensive; unlikely to trigger for torchvision
        for x in stem.modules():
            if isinstance(x, nn.Conv2d):
                return x
        raise RuntimeError("Could not find first Conv2d in ConvNeXt stem")
    return conv

# --------------------------- Image-only / Late ---------------------------

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt (Tiny/Base) backbone returning pooled features and logits.
    - variant: "tiny" or "base"
    - out_dim: #classes (2 for binary)
    """
    def __init__(self, variant: str = "tiny", pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        m, feat_dim = _load_convnext(variant, pretrained)
        self.backbone = m
        self.feat_dim = feat_dim
        # remove classifier (keep avgpool)
        self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # (B, feat_dim)
        return x

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

# --------------------------- Early fusion ---------------------------

class ConvNeXtEarlyFusionPatient(nn.Module):
    """
    Early fusion:
      metadata vector (B, tab_dim) --MLP--> (B, c_pat) --broadcast--> (B, c_pat, H, W)
      then concat with RGB before the first ConvNeXt stem conv.
    """
    def __init__(self, variant: str, tab_dim: int, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2,
                 pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # meta -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        m, feat_dim = _load_convnext(variant, pretrained)

        # patch first conv (RGB -> RGB + c_pat)
        first = _first_conv_module(m)
        assert isinstance(first, nn.Conv2d) and first.in_channels == 3
        new_first = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False
        )
        with torch.no_grad():
            new_first.weight[:, :3, ...] = first.weight
        nn.init.kaiming_normal_(new_first.weight[:, 3:, ...], nonlinearity="relu")

        # assign back
        m.features[0][0] = new_first

        # strip classifier, keep avgpool
        m.classifier = nn.Identity()

        self.backbone = m
        self.feat_dim = feat_dim
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img, clin_vec):
        """
        x_img:   (B, 3, H, W)
        clin_vec:(B, tab_dim)
        """
        B, C, H, W = x_img.shape
        assert C == 3, f"expected RGB input, got {C} channels"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)                               # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)  # (B,c_pat,H,W)
        x = torch.cat([x_img, ch], dim=1)                      # (B, 3+c_pat, H, W)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits