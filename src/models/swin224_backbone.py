import torch
import torch.nn as nn

try:
    import timm  # Preferred for feature extraction API
except Exception:
    timm = None

try:
    from torchvision.models import swin_t, Swin_T_Weights
except Exception:
    swin_t = None
    Swin_T_Weights = None


class SwinTiny224Backbone(nn.Module):
    """
    Swin Transformer Tiny (224x224) backbone for image-only use.
    Returns (feats, logits) where feats is a pooled feature vector.

    Uses timm if available (num_classes=0, global_pool="avg"), otherwise
    falls back to torchvision's `swin_t` with the classifier head removed.
    """

    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        self._uses_timm = False

        if timm is not None:
            # timm model returns pooled features when num_classes=0
            m = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.backbone = m
            self._uses_timm = True
            # Prefer provided attribute, fallback to inference via dummy pass
            self.feat_dim = getattr(m, "num_features", None)
            if self.feat_dim is None:
                with torch.no_grad():
                    self.feat_dim = m(torch.zeros(1, 3, 224, 224)).shape[1]
        elif swin_t is not None:
            # Torchvision fallback
            w = Swin_T_Weights.IMAGENET1K_V1 if (pretrained and Swin_T_Weights is not None) else None
            m = swin_t(weights=w)
            # Get feature dim from the classifier before stripping it
            in_feats = getattr(m.head, "in_features", None)
            if in_feats is None:
                # As a last resort, infer with a dummy forward
                tmp = m
                tmp_head = tmp.head
                tmp.head = nn.Identity()
                with torch.no_grad():
                    in_feats = tmp(torch.zeros(1, 3, 224, 224)).shape[1]
                tmp.head = tmp_head
            # Remove classifier head so forward returns pooled features
            m.head = nn.Identity()
            self.backbone = m
            self.feat_dim = in_feats
        else:
            raise ImportError(
                "Neither timm nor torchvision swin_t is available. Install timm>=1.0 or torchvision>=0.13."
            )

        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        # timm: model returns pooled features directly (num_classes=0, global_pool="avg")
        # torchvision: we set head=Identity so forward returns pooled features
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits

