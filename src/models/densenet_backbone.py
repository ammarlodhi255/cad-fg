import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201, DenseNet201_Weights


class DenseNet201Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet201(weights=w)

        in_feats = m.classifier.in_features  # 1920
        self.feat_dim = in_feats

        # remove classifier head
        m.classifier = nn.Identity()
        self.backbone = m

        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x):
        # follow torchvision densenet forward minus classifier
        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return feats

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class DenseNet201EarlyFusionPatient(nn.Module):
    """
    Early fusion: concat RGB with patient-channel map at input.
    Metadata vector (B, tab_dim) -> pmap -> (B, c_pat, H, W).
    """
    def __init__(self, tab_dim: int, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2,
                 pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.tab_dim = tab_dim
        self.c_pat = c_pat

        # project metadata vector -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet201(weights=w)

        # patch first conv
        old = m.features.conv0
        assert isinstance(old, nn.Conv2d) and old.in_channels == 3
        new_in = 3 + c_pat
        new_conv = nn.Conv2d(
            new_in, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3, ...] = old.weight
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")
        m.features.conv0 = new_conv

        in_feats = m.classifier.in_features  # 1920
        self.feat_dim = in_feats
        m.classifier = nn.Identity()
        self.backbone = m

        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img, clin_vec):
        B, _, H, W = x_img.shape
        ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)

        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

        logits = self.head(feats)
        return feats, logits