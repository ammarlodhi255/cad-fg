import os
import torch
import torch.nn as nn
import types


class YOLOv5ClsBackbone(nn.Module):
    """
    Image-only wrapper for YOLOv5 classification via torch.hub (e.g., 'yolov5s-cls').
    - Requires torch.hub access to 'ultralytics/yolov5' or a locally cached copy.
    - Uses pre-logits (penultimate features) by replacing the final Linear with Identity,
      then maps to out_dim via a Linear head.
    """
    def __init__(self, model_name: str = "yolov5s-cls", pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        # Accept names with or without '.pt' extension
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        # If weights file is not present, try to fetch from a known release URL
        if not os.path.isfile(name):
            base = "https://github.com/ultralytics/yolov5/releases/download/v7.0/"
            known = {"yolov5s-cls.pt", "yolov5m-cls.pt", "yolov5l-cls.pt", "yolov5x-cls.pt"}
            if os.path.basename(name) in known:
                url = base + os.path.basename(name)
                try:
                    # Prefer torch.hub downloader if available
                    if hasattr(torch, 'hub') and hasattr(torch.hub, 'download_url_to_file'):
                        torch.hub.download_url_to_file(url, name, progress=True)
                    else:
                        import urllib.request
                        urllib.request.urlretrieve(url, name)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Could not find weights '{name}' locally and failed to download from {url}. "
                        f"Place the weights file in the working directory or provide a full path. Error: {e}"
                    )

        # Load YOLOv5 classification model via torch.hub 'custom' entrypoint.
        # Use trust_repo=True to allow loading the external repo code.
        if not hasattr(torch, 'hub') or not isinstance(torch.hub, types.ModuleType):
            raise ImportError("torch.hub is required to load YOLOv5 classification models (ultralytics/yolov5).")
        self.base = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=name,
            autoshape=False,
            trust_repo=True,
            verbose=False,
        )  # type: ignore

        # Replace the last Linear with Identity to expose pre-logits as features.
        in_feats = self._replace_last_linear_with_identity(self.base)
        if in_feats is None:
            # Fallback: run a dummy forward to infer dimensionality
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.base(dummy)
                in_feats = int(out.shape[1])

        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        if isinstance(self.base, nn.Module):
            try:
                nn.Module.train(self.base, mode)
            except Exception:
                self.base.eval()
        nn.Module.train(self.head, mode)
        return self

    @staticmethod
    def _to_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for it in list(obj)[::-1]:
                t = YOLOv5ClsBackbone._to_tensor(it)
                if t is not None:
                    return t
        for attr in ("logits", "pred", "output"):
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if torch.is_tensor(v):
                    return v
        if hasattr(obj, "probs"):
            probs = getattr(obj, "probs")
            if hasattr(probs, "data") and torch.is_tensor(probs.data):
                return probs.data
        return None

    def forward(self, x: torch.Tensor):
        out = self.base(x)
        feats = self._to_tensor(out)
        if feats is None:
            raise RuntimeError("YOLOv5 classification forward returned unsupported output; cannot extract logits tensor.")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        logits = self.head(feats)
        return feats, logits

    @staticmethod
    def _replace_last_linear_with_identity(model: nn.Module):
        """Find and replace the last nn.Linear in `model` with nn.Identity.
        Returns the in_features of the replaced layer, or None if not found.
        """
        last_name = None
        last_fc = None
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                last_name, last_fc = name, m
        if last_fc is None or last_name is None:
            return None
        # Walk to parent and replace
        parent = model
        parts = last_name.split('.')
        for p in parts[:-1]:
            parent = parent._modules[p]
        parent._modules[parts[-1]] = nn.Identity()
        return int(last_fc.in_features)


class YOLOv5EarlyFusionPatient(nn.Module):
    """
    Early-fusion variant for YOLOv5 classification models loaded via torch.hub.
    - Adds metadata-derived channels (c_pat) to the input by patching the first Conv2d.
    - Uses pre-logits (penultimate features) by replacing the final Linear with Identity,
      then maps to `out_dim` with a Linear head.
    """
    def __init__(self, model_name: str = "yolov5s-cls", tab_dim: int = 0, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        # Prefer local weights; do not force download here (caller should place weights)
        if not os.path.isfile(name):
            raise FileNotFoundError(
                f"Weights '{name}' not found. Place the YOLOv5 classification weights in the working directory.")

        if not hasattr(torch, 'hub') or not isinstance(torch.hub, types.ModuleType):
            raise ImportError("torch.hub is required to load YOLOv5 classification models (ultralytics/yolov5).")
        self.base = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=name,
            autoshape=False,
            trust_repo=True,
            verbose=False,
        )  # type: ignore

        self.c_pat = int(c_pat)
        self.tab_dim = int(tab_dim)
        self.pmap = nn.Sequential(
            nn.Linear(self.tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, self.c_pat)
        ) if self.tab_dim > 0 else None

        self._patch_first_conv(self.base, add_in_channels=self.c_pat)

        # Replace the last Linear with Identity to expose pre-logits.
        in_feats = YOLOv5ClsBackbone._replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3 + self.c_pat, 224, 224)
                out = self.base(dummy)
                in_feats = int(out.shape[1])
        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    @staticmethod
    def _find_first_conv_wrapper(model: nn.Module):
        for m in model.modules():
            if hasattr(m, 'conv') and isinstance(getattr(m, 'conv'), nn.Conv2d):
                conv = getattr(m, 'conv')
                if int(getattr(conv, 'in_channels', -1)) == 3:
                    return m, conv
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and int(m.in_channels) == 3:
                return None, m
        return None, None

    @staticmethod
    def _patch_first_conv(model: nn.Module, add_in_channels: int):
        if add_in_channels <= 0:
            return
        wrapper, conv = YOLOv5EarlyFusionPatient._find_first_conv_wrapper(model)
        if conv is None:
            raise RuntimeError("Could not locate first Conv2d to patch for YOLOv5 early fusion")
        new_conv = nn.Conv2d(
            in_channels=int(conv.in_channels) + add_in_channels,
            out_channels=int(conv.out_channels),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, ...] = conv.weight
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")
        if conv.bias is not None:
            with torch.no_grad():
                new_conv.bias.copy_(conv.bias)
        if wrapper is not None and hasattr(wrapper, 'conv'):
            setattr(wrapper, 'conv', new_conv)
        else:
            for name, module in model.named_modules():
                if module is conv:
                    path = name.split('.')
                    parent = model
                    for p in path[:-1]:
                        parent = parent._modules[p]
                    parent._modules[path[-1]] = new_conv
                    break

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        if isinstance(self.base, nn.Module):
            try:
                nn.Module.train(self.base, mode)
            except Exception:
                self.base.eval()
        nn.Module.train(self.head, mode)
        if self.pmap is not None:
            nn.Module.train(self.pmap, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B, C, H, W = x_img.shape
        if self.c_pat > 0:
            assert clin_vec is not None and clin_vec.dim() == 2 and clin_vec.size(0) == B
            ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
            x = torch.cat([x_img, ch], dim=1)
        else:
            x = x_img
        out = self.base(x)
        feats = YOLOv5ClsBackbone._to_tensor(out) if hasattr(YOLOv5ClsBackbone, '_to_tensor') else (out if torch.is_tensor(out) else None)
        if feats is None:
            raise RuntimeError("YOLOv5 classification forward returned unsupported output; cannot extract logits tensor.")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        logits = self.head(feats)
        return feats, logits
