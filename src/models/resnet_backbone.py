import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, out_dim=2):
        super().__init__()
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=w)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(2048, out_dim)

    def forward_feats(self, x):
        return self.backbone(x)

    def forward(self, x):
        feats = self.forward_feats(x)
        return feats, self.head(feats)