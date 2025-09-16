import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ---------- Image-only / Late-fusion backbone ----------
class EfficientNetB3Backbone(nn.Module):
    """
    Returns pooled features (1536-d) and logits via a small head.
    Use out_dim=2 for binary classification.
    """
    def __init__(self, pretrained=True, out_dim=2):
        super().__init__()
        w = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b3(weights=w)
        in_feats = m.classifier[1].in_features  # 1536 for B3
        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats  # 1536

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # 1536

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


# ---------- Early fusion: add patient channels to the first conv ----------
class EfficientNetB3EarlyFusionPatient(nn.Module):
    """
    Metadata vector (B, tab_dim) -> small MLP -> c_pat channels,
    broadcast to HxW and concatenated with RGB before the first conv.
    """
    def __init__(self, tab_dim, c_pat=8, pretrained=True, out_dim=2,
                 pmap_hidden=64, pmap_drop=0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # metadata -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b3(weights=w)

        # First conv lives at m.features[0][0] in torchvision EfficientNet
        first = m.features[0][0]
        assert isinstance(first, nn.Conv2d) and first.in_channels == 3
        new_first = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        with torch.no_grad():
            new_first.weight[:, :3, ...] = first.weight  # keep RGB pretrained weights
        nn.init.kaiming_normal_(new_first.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0][0] = new_first

        in_feats = m.classifier[1].in_features  # 1536
        m.classifier = nn.Identity()

        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats

    def forward(self, x_img, clin_vec):
        B, _, H, W = x_img.shape
        ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits