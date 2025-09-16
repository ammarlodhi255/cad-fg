import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .attn.cbam import CBAM


# ---------- Late / Image-only: EfficientNet-B0 backbone (optional CBAM) ----------
class EfficientNetB0Backbone(nn.Module):
    """
    Returns pooled features (1280-d) and logits.
    Set cbam=True to apply CBAM on the final feature map (C=1280) before pooling.
    """
    def __init__(self, pretrained=True, out_dim=2, cbam: bool = False, cbam_reduction: int = 16):
        super().__init__()
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b0(weights=w)
        in_feats = m.classifier[1].in_features  # 1280

        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats

        self.use_cbam = cbam
        self.cbam = CBAM(channels=in_feats, reduction=cbam_reduction) if cbam else nn.Identity()

    def forward_feats(self, x):
        x = self.backbone.features(x)   # (B, 1280, H/32, W/32)
        x = self.cbam(x)                # <-- CBAM here (no-op if disabled)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # 1280

    def forward(self, x):
        feats = self.forward_feats(x)
        return feats, self.head(feats)


# ---------- Early fusion: add patient channels to the first conv (optional CBAM) ----------
class EfficientNetB0EarlyFusionPatient(nn.Module):
    """
    EfficientNet-B0 with extra input channels (3 + c_pat).
    Set cbam=True to apply CBAM on the final feature map (C=1280) before pooling.
    """
    def __init__(self, tab_dim, c_pat=8, pretrained=True, out_dim=2,
                 pmap_hidden=64, pmap_drop=0.1, cbam: bool = False, cbam_reduction: int = 16):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # metadata -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b0(weights=w)

        # First conv lives at m.features[0][0] in torchvision EfficientNet
        first = m.features[0][0]
        assert isinstance(first, nn.Conv2d) and first.in_channels == 3, "Unexpected EfficientNet structure"
        new_first = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        with torch.no_grad():
            new_first.weight[:, :3, ...] = first.weight
        nn.init.kaiming_normal_(new_first.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0][0] = new_first

        in_feats = m.classifier[1].in_features  # 1280
        m.classifier = nn.Identity()

        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats

        self.use_cbam = cbam
        self.cbam = CBAM(channels=in_feats, reduction=cbam_reduction) if cbam else nn.Identity()

    def forward(self, x_img, clin_vec):
        B, _, H, W = x_img.shape
        # map metadata -> (B, c_pat, H, W)
        ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)

        x = self.backbone.features(x)   # (B, 1280, ...)
        x = self.cbam(x)                # <-- CBAM here (no-op if disabled)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits