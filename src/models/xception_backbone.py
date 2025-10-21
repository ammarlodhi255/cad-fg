import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None


class XceptionBackbone(nn.Module):
    """
    Xception backbone via timm. Model names may vary by version:
      'xception', 'xception41', 'xception65', 'xception71'
    Default tries a preferred list and picks the first available.
    """
    def __init__(self, model_name: str = None, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for Xception backbone. Install timm>=1.0.")
        names = [model_name] if model_name else ["xception", "xception41", "xception65", "xception71"]
        last_err = None
        m = None
        for n in names:
            try:
                m = timm.create_model(n, pretrained=pretrained, num_classes=0, global_pool="avg")
                break
            except Exception as err:  # pragma: no cover
                last_err = err
                m = None
        if m is None:
            raise RuntimeError(f"Could not instantiate Xception model via timm: {last_err}")
        self.backbone = m
        self.feat_dim = getattr(m, "num_features", None) or m(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

