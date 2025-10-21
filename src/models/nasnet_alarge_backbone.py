import torch
import torch.nn as nn

try:
    import timm
except Exception:
    timm = None


class NASNetALargeBackbone(nn.Module):
    """
    NASNet-A-Large backbone via timm. Returns pooled features and logits.
    Model name: 'nasnetalarge'. Heavier than Mobile variant.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for NASNet-A-Large backbone. Install timm>=1.0.")
        m = timm.create_model("nasnetalarge", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.backbone = m
        self.feat_dim = getattr(m, "num_features", None) or m(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

