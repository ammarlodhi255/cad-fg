import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# ---------------- Image-only / Late-fusion backbone ----------------

class MobileNetV3LargeBackbone(nn.Module):
    """
    Image backbone returning pooled features and logits.
    feat_dim (for late fusion): 960
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_large(weights=w)

        # torchvision head: avgpool -> classifier (Linear 960->1280->#cls)
        in_feats = m.classifier[0].in_features  # 960
        self.feat_dim = in_feats

        # keep avgpool, strip classifier, add our head
        self.backbone = m
        self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)   # AdaptiveAvgPool2d(1)
        x = torch.flatten(x, 1)        # (B, 960)
        return x

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


# ---------------- Early fusion (metadata → channels at input) ----------------

class MobileNetV3LargeEarlyFusionPatient(nn.Module):
    """
    Metadata vector (B, tab_dim) -> small MLP -> c_pat channels,
    broadcast to HxW and concatenated with RGB before the first conv.
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

        w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_large(weights=w)

        # First conv is features[0][0] (Conv2d)
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
            new_first.weight[:, :3, ...] = first.weight
        nn.init.kaiming_normal_(new_first.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0][0] = new_first

        in_feats = m.classifier[0].in_features  # 960
        self.feat_dim = in_feats
        m.classifier = nn.Identity()

        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img, clin_vec):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected 3-channel image, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)                                # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)                       # (B, 3+c_pat, H, W)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)                              # (B, 960)
        logits = self.head(feats)
        return feats, logits