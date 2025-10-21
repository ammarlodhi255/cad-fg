import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _guess_regnet_weights_enum_name(name: str) -> str:
    # Convert e.g. 'regnet_y_400mf' -> 'RegNet_Y_400MF_Weights'
    parts = name.split("_")
    cap = "_".join(p.upper() if any(c.isdigit() for c in p) else p.capitalize() for p in parts)
    return f"{cap}_Weights"


class RegNetBackbone(nn.Module):
    """
    Torchvision RegNet backbone. Example variants:
      - regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
      - regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf
    """
    def __init__(self, variant: str = "regnet_y_400mf", pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        name = variant.lower()
        if not hasattr(tvm, name):
            raise ValueError(f"Unknown torchvision RegNet function: {variant}")
        fn = getattr(tvm, name)

        weights = None
        if pretrained:
            enum_name = _guess_regnet_weights_enum_name(name)
            weights_enum = getattr(tvm, enum_name, None)
            if weights_enum is not None:
                # Prefer IMAGENET1K_V2 if available, else V1
                weights = getattr(weights_enum, "IMAGENET1K_V2", None) or getattr(weights_enum, "IMAGENET1K_V1", None)

        m = fn(weights=weights)
        in_feats = m.fc.in_features
        self.feat_dim = in_feats
        m.fc = nn.Identity()

        self.backbone = m
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit ResNet-like forward
        m = self.backbone
        x = m.stem(x) if hasattr(m, "stem") else m.conv1(x)
        if hasattr(m, "bn1"): x = m.bn1(x)
        if hasattr(m, "relu"): x = m.relu(x)
        if hasattr(m, "maxpool"): x = m.maxpool(x)
        # RegNet blocks live in trunk_output or sequential layers
        if hasattr(m, "trunk_output"):
            x = m.trunk_output(x)
        else:
            for layer_name in ("layer1", "layer2", "layer3", "layer4"):
                if hasattr(m, layer_name):
                    x = getattr(m, layer_name)(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


class RegNetEarlyFusionPatient(nn.Module):
    """
    Early fusion for torchvision RegNet.
    Adds c_pat metadata channels to RGB at input by replacing the first Conv2d in the stem.
    """
    def __init__(self, variant: str, tab_dim: int, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2,
                 pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # metadata -> channel projection
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        name = variant.lower()
        if not hasattr(tvm, name):
            raise ValueError(f"Unknown torchvision RegNet function: {variant}")
        fn = getattr(tvm, name)

        weights = None
        if pretrained:
            enum_name = _guess_regnet_weights_enum_name(name)
            weights_enum = getattr(tvm, enum_name, None)
            if weights_enum is not None:
                weights = getattr(weights_enum, "IMAGENET1K_V2", None) or getattr(weights_enum, "IMAGENET1K_V1", None)

        m = fn(weights=weights)

        # find and replace first conv in the stem
        stem = getattr(m, "stem", None)
        if stem is None:
            # fallback: search whole model for first Conv2d with 3 in-ch
            modules = list(m.modules())
        else:
            modules = list(stem.modules())

        first_conv = None
        for mod in modules:
            if isinstance(mod, nn.Conv2d) and getattr(mod, "in_channels", None) == 3:
                first_conv = mod
                break
        if first_conv is None:
            raise RuntimeError("Could not locate first Conv2d with 3 input channels in RegNet stem")

        new_conv = nn.Conv2d(
            in_channels=3 + c_pat,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=(first_conv.bias is not None),
            padding_mode=first_conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, ...] = first_conv.weight
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")

        # assign back into the model by navigating attributes
        def _replace_module(parent: nn.Module, target: nn.Module, replacement: nn.Module):
            for name, child in parent.named_children():
                if child is target:
                    setattr(parent, name, replacement)
                    return True
                if _replace_module(child, target, replacement):
                    return True
            return False

        if not _replace_module(m, first_conv, new_conv):
            raise RuntimeError("Failed to replace first conv in RegNet model")

        in_feats = m.fc.in_features
        m.fc = nn.Identity()

        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected RGB input, got {C} channels"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        x = torch.cat([x_img, ch], dim=1)

        # run standard RegNet forward to pooled features
        m = self.backbone
        if hasattr(m, "stem"):
            x = m.stem(x)
        else:
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        if hasattr(m, "trunk_output"):
            x = m.trunk_output(x)
        else:
            for lname in ("layer1", "layer2", "layer3", "layer4"):
                if hasattr(m, lname): x = getattr(m, lname)(x)
        x = m.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits
