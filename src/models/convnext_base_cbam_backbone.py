import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from .attn.cbam import CBAM


def _first_conv_module(m: nn.Module) -> nn.Conv2d:
    """
    Torchvision ConvNeXt stem: m.features[0][0] is Conv2d(3, C, k=4, s=4).
    """
    stem = m.features[0]
    conv = stem[0]
    if not isinstance(conv, nn.Conv2d):
        for x in stem.modules():
            if isinstance(x, nn.Conv2d):
                return x
        raise RuntimeError("Could not find first Conv2d in ConvNeXt stem")
    return conv


# ---------------- Image-only / Late-fusion backbone with CBAM ----------------

class ConvNeXtBaseBackboneCBAM(nn.Module):
    """
    ConvNeXt-Base backbone that inserts CBAM on the FINAL feature map (C=1024)
    before avg-pooling. This is robust across torchvision versions.

    Returns (feats, logits).  feat_dim = 1024.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2,
                 cbam_reduction: int = 16, cbam_spatial_kernel: int = 7):
        super().__init__()
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        m = convnext_base(weights=w)

        # strip classifier, keep features + avgpool
        m.classifier = nn.Identity()
        self.backbone = m
        self.feat_dim = 1024

        # CBAM on the final feature map
        self.cbam = CBAM(
            channels=self.feat_dim,
            reduction=cbam_reduction,
            spatial_kernel=cbam_spatial_kernel
        )

        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)   # (B, 1024, H/32, W/32)
        x = self.cbam(x)                # <-- CBAM here
        x = self.backbone.avgpool(x)    # AdaptiveAvgPool2d(1)
        x = torch.flatten(x, 1)         # (B, 1024)
        return x

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


# ---------------- Early fusion (metadata → channels) + CBAM ----------------

class ConvNeXtBaseEarlyFusionPatientCBAM(nn.Module):
    """
    Early fusion:
      metadata (B, tab_dim) --MLP--> (B, c_pat) --broadcast--> (B,c_pat,H,W)
      concat with RGB before first stem conv.
    CBAM is applied on the final feature map before pooling (C=1024).
    """
    def __init__(self, tab_dim: int, c_pat: int = 8, pretrained: bool = True,
                 out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1,
                 cbam_reduction: int = 16, cbam_spatial_kernel: int = 7):
        super().__init__()
        self.tab_dim = tab_dim
        self.c_pat = c_pat

        # metadata -> channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        m = convnext_base(weights=w)

        # Patch the first stem conv to accept 3 + c_pat channels
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
        m.features[0][0] = new_first  # assign back

        # Remove classifier; keep features + avgpool
        m.classifier = nn.Identity()
        self.backbone = m
        self.feat_dim = 1024

        # CBAM on final map
        self.cbam = CBAM(
            channels=self.feat_dim,
            reduction=cbam_reduction,
            spatial_kernel=cbam_spatial_kernel
        )

        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img, clin_vec):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected 3-channel RGB, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)                                # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)                       # (B, 3+c_pat, H, W)

        x = self.backbone.features(x)       # (B, 1024, ...)
        x = self.cbam(x)                    # <-- CBAM here
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)         # (B, 1024)
        logits = self.head(feats)
        return feats, logits