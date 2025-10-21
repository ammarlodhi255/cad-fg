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

