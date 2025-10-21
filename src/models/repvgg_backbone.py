import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None


class RepVGGBackbone(nn.Module):
    """
    RepVGG backbone via timm. Example model names:
      repvgg_b0, repvgg_b1, repvgg_b1g4, repvgg_b2, repvgg_b3
    """
    def __init__(self, model_name: str = "repvgg_b1", pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for RepVGG backbone. Install timm>=1.0.")
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.backbone = m
        self.feat_dim = getattr(m, "num_features", None)
        if self.feat_dim is None:
            self.feat_dim = m(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class RepVGGEarlyFusionPatient(nn.Module):
    """
    Early fusion for RepVGG via timm: patch first Conv2d to accept (3 + c_pat) input channels.
    """
    def __init__(self, model_name: str = "repvgg_b1", tab_dim: int = 0, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2,
                 pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for RepVGG backbone. Install timm>=1.0.")
        self.c_pat = c_pat
        self.tab_dim = tab_dim
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        m = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")

        # Find all Conv2d layers in the stem expecting 3 input channels (some stems have parallel convs).
        stem = getattr(m, 'stem', None)
        search_modules = list(stem.modules()) if stem is not None else list(m.modules())
        convs_to_patch = []
        for mod in search_modules:
            if isinstance(mod, nn.Conv2d) and getattr(mod, 'in_channels', None) == 3:
                convs_to_patch.append(mod)
        if not convs_to_patch:
            raise RuntimeError("Could not find any Conv2d(in=3, ...) in RepVGG stem to patch for early fusion")

        def _replace_module(parent: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
            for name, child in parent.named_children():
                if child is target:
                    setattr(parent, name, replacement)
                    return True
                if _replace_module(child, target, replacement):
                    return True
            return False

        # Patch each 3-channel conv in the stem
        for old in convs_to_patch:
            new_conv = nn.Conv2d(
                in_channels=3 + c_pat,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=(old.bias is not None),
                padding_mode=old.padding_mode,
            )
            with torch.no_grad():
                new_conv.weight[:, :3, ...] = old.weight
                if old.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old.bias)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")
            if not _replace_module(m, old, new_conv):
                raise RuntimeError("Failed to replace a stem conv in RepVGG model for early fusion")

        self.backbone = m
        self.feat_dim = getattr(m, "num_features", None) or m(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected RGB input, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim
        pm = self.pmap(clin_vec)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)
        feats = self.backbone(x)
        logits = self.head(feats)
        return feats, logits
