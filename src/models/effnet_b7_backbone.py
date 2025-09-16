import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


# ---------------- Image-only / Late-fusion backbone ----------------

class EfficientNetB7Backbone(nn.Module):
    """
    EfficientNet-B7 backbone that returns (feats, logits).
    Feature dim for late fusion: 2560.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b7(weights=w)

        in_feats = m.classifier[1].in_features  # 2560 for B7
        self.feat_dim = in_feats

        # strip classifier, keep features + avgpool
        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)         # AdaptiveAvgPool2d(1)
        x = torch.flatten(x, 1)              # (B, 2560)
        return x

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


# ---------------- Early fusion (metadata → channels at input) ----------------

class EfficientNetB7EarlyFusionPatient(nn.Module):
    """
    Early fusion:
      metadata vector (B, tab_dim) --MLP--> (B, c_pat) --broadcast--> (B,c_pat,H,W)
      concat with RGB (3) -> first conv sees 3+c_pat channels.
    """
    def __init__(self, tab_dim: int, c_pat: int = 8, pretrained: bool = True,
                 out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # metadata -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b7(weights=w)

        # First conv is at m.features[0][0] in torchvision EfficientNet
        first = m.features[0][0]
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
            new_first.weight[:, :3, ...] = first.weight          # keep pretrained RGB
        nn.init.kaiming_normal_(new_first.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0][0] = new_first

        in_feats = m.classifier[1].in_features  # 2560
        self.feat_dim = in_feats
        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img, clin_vec):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected 3-channel input, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)                                 # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)                        # (B, 3+c_pat, H, W)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)                               # (B, 2560)
        logits = self.head(feats)
        return feats, logits