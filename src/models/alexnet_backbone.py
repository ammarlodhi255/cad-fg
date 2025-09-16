import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights


class AlexNetBackbone(nn.Module):
    """
    Image backbone for image-only / late fusion.
    Returns (feats, logits).
    Feature dim (after features + AdaptiveAvgPool2d(6,6) + flatten): 9216.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        m = alexnet(weights=w)

        # AlexNet has: features -> avgpool((6,6)) -> classifier(Sequential)
        # We remove classifier and add our head.
        self.backbone = m
        self.backbone.classifier = nn.Identity()

        self.feat_dim = 256 * 6 * 6  # 9216
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)          # (B,256,6,6)
        x = torch.flatten(x, 1)               # (B,9216)
        return x

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class AlexNetEarlyFusionPatient(nn.Module):
    """
    Early fusion: metadata (B, tab_dim) -> c_pat channels -> concat with RGB before first conv.
    """
    def __init__(self, tab_dim: int, c_pat: int = 8, pretrained: bool = True,
                 out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.tab_dim = tab_dim
        self.c_pat = c_pat

        # metadata -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        m = alexnet(weights=w)

        # First conv is features[0]: Conv2d(3,64,k=11,s=4,p=2)
        old = m.features[0]
        assert isinstance(old, nn.Conv2d) and old.in_channels == 3
        new_conv = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3, ...] = old.weight
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0] = new_conv

        # remove classifier; keep avgpool to (6,6)
        m.classifier = nn.Identity()
        self.backbone = m
        self.feat_dim = 256 * 6 * 6
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img, clin_vec):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected RGB, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)                               # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)                      # (B,3+c_pat,H,W)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)                            # (B,9216)
        logits = self.head(feats)
        return feats, logits