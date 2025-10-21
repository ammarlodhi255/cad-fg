import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
)


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 backbone (large/small) returning pooled features and logits.
    """
    def __init__(self, variant: str = "large", pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        v = variant.lower()
        if v == "large":
            w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            m = mobilenet_v3_large(weights=w)
        elif v == "small":
            w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            m = mobilenet_v3_small(weights=w)
        else:
            raise ValueError(f"Unknown MobileNetV3 variant: {variant}")

        # After features + avgpool + flatten, MobileNetV3 outputs 960 (large) / 576 (small)
        # The first classifier Linear takes this pooled feature dim as input.
        in_feats = m.classifier[0].in_features
        self.feat_dim = in_feats
        m.classifier = nn.Identity()

        self.backbone = m
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits
