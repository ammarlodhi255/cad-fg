import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121Backbone(nn.Module):
    """
    Image backbone for image-only / late fusion.
    Returns (feats, logits). Feature dim = 1024.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet121(weights=w)

        in_feats = m.classifier.in_features  # 1024
        self.feat_dim = in_feats

        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x  # (B, 1024)

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class DenseNet121EarlyFusionPatient(nn.Module):
    """
    Early fusion: metadata (B, tab_dim) -> c_pat channels -> concat with RGB.
    """
    def __init__(self, tab_dim: int, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2,
                 pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.tab_dim = tab_dim
        self.c_pat = c_pat

        # metadata -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet121(weights=w)

        # patch first conv (3 -> 3+c_pat)
        old = m.features.conv0
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
        m.features.conv0 = new_conv

        in_feats = m.classifier.in_features  # 1024
        self.feat_dim = in_feats
        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img, clin_vec):
        B, _, H, W = x_img.shape
        # project metadata -> channels
        pm = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, pm], dim=1)

        x = self.backbone.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        feats = x
        logits = self.head(feats)
        return feats, logits