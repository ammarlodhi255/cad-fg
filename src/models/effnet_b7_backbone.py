import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


class EfficientNetB7Backbone(nn.Module):
    """
    Image-only EfficientNet-B7 backbone returning pooled features and logits.
    Tip: use img_size=600 in config for B7 pretraining resolution.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b7(weights=w)
        in_feats = m.classifier[1].in_features  # 2560
        m.classifier = nn.Identity()

        self.backbone = m
        self.feat_dim = in_feats
        self.head = nn.Linear(in_feats, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class EfficientNetB7EarlyFusionPatient(nn.Module):
    """
    EfficientNet-B7 with early fusion of patient metadata.
    Adds c_pat channels to RGB by projecting the tabular vector and concatenating
    before the first conv, then forwards through EfficientNet-B7.

    Recommend using img_size ~600 to align with B7 pretraining resolution.
    """
    def __init__(self, tab_dim: int, c_pat: int = 8, pretrained: bool = True,
                 out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b7(weights=w)

        # Patch first conv to accept 3 + c_pat channels
        first = m.features[0][0]
        assert isinstance(first, nn.Conv2d) and first.in_channels == 3, "Unexpected EfficientNet-B7 stem"
        new_first = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        with torch.no_grad():
            new_first.weight[:, :3, ...] = first.weight
        nn.init.kaiming_normal_(new_first.weight[:, 3:, ...], nonlinearity="relu")
        m.features[0][0] = new_first

        in_feats = m.classifier[1].in_features  # 2560
        m.classifier = nn.Identity()

        self.backbone = m
        self.feat_dim = in_feats
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected RGB input, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits
