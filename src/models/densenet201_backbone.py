import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201, DenseNet201_Weights


class DenseNet201Backbone(nn.Module):
    """
    Image-only / late-fusion DenseNet-201 backbone.
    Returns (feats, logits). Feature dim = 1920.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet201(weights=w)

        in_feats = m.classifier.in_features  # 1920
        self.feat_dim = in_feats

        m.classifier = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x):
        x = self.backbone.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x  # (B, 1920)

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

