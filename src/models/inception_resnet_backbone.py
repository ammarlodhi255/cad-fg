import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None


class InceptionResNetV2Backbone(nn.Module):
    """
    Inception-ResNet-v2 backbone via timm.
    Returns global pooled features and logits.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for Inception-ResNet-v2 backbone. Install timm>=1.0.")

        # Try common timm aliases across versions
        last_err = None
        model = None
        for name in [
            "inception_resnet_v2",
            "inceptionresnetv2",
        ]:
            try:
                model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
                break
            except Exception as err:  # pragma: no cover - only hit if model not present
                last_err = err
                model = None
        if model is None:
            raise RuntimeError(f"Could not create Inception-ResNet-v2 via timm: {last_err}")

        self.backbone = model
        self.feat_dim = getattr(model, "num_features", None) or getattr(model, "feature_info", None)
        if self.feat_dim is None:
            # Fallback: attempt a dummy forward to infer dim
            self.feat_dim = model(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, feat_dim)

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

