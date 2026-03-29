import torch
import torch.nn as nn
from torchvision.models import (
    vgg16, vgg19,
    VGG16_Weights, VGG19_Weights,
)


class VGG16Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vgg16(weights=w)
        # keep first two FC layers (dim=4096) and drop the final classifier layer
        in_feats = m.classifier[-1].in_features  # 4096
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        self.backbone = m
        self.feat_dim = in_feats
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x  # (B, 4096)

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class VGG19Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        m = vgg19(weights=w)
        in_feats = m.classifier[-1].in_features  # 4096
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        self.backbone = m
        self.feat_dim = in_feats
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

class VGG19EarlyFusionPatient(nn.Module):
    """
    Early fusion for VGG19: project tabular vector to c_pat channels and
    concatenate with RGB at the input, then run standard VGG19 up to the penultimate
    FC layer (4096-dim) and classify.
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

        w = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        m = vgg19(weights=w)

        # Patch first conv in the features stem (index 0) to accept (3 + c_pat)
        old = m.features[0]
        assert isinstance(old, nn.Conv2d) and old.in_channels == 3
        new_conv = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :3, ...] = old.weight
            if old.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old.bias)
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0] = new_conv

        # Keep penultimate FC (4096-d) as features and add a new head
        in_feats = m.classifier[-1].in_features  # 4096
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        self.backbone = m
        self.feat_dim = in_feats
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B, _, H, W = x_img.shape
        pm = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, pm], dim=1)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        feats = self.backbone.classifier(x)
        logits = self.head(feats)
        return feats, logits
