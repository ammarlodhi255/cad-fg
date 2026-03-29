import torch
import torch.nn as nn

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # ultralytics not installed


class YOLOv8ClsBackbone(nn.Module):
    """
    Image-only wrapper for YOLOv8 classification models (e.g., 'yolov8n-cls', 'yolov8s-cls').
    - Requires 'ultralytics' to be installed and weights available (local or cache).
    - Uses pre-logits (penultimate features) by replacing the final Linear with Identity,
      then maps to out_dim via a Linear head.
    """
    def __init__(self, model_name: str = "yolov8n-cls", pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOv8 classification. Install via 'pip install ultralytics'.")

        # Accept names with or without '.pt'
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        self._yolo = YOLO(name) if pretrained else YOLO(name)
        # Use the underlying PyTorch model for forward to get logits
        self.base = self._yolo.model

        # Replace last Linear with Identity to expose pre-logits
        in_feats = self._replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.base(dummy)
                t = YOLOv8ClsBackbone._to_tensor(out)
                in_feats = int(t.shape[1])

        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    # Avoid triggering Ultralytics' high-level training engine when toggling mode.
    # We only need standard PyTorch train/eval behavior for our backbone wrapper.
    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        # put the torch model and head into appropriate mode, but do NOT call YOLO().train()
        if isinstance(self.base, nn.Module):
            try:
                nn.Module.train(self.base, mode)  # call base class impl directly
            except Exception:
                self.base.eval()
        nn.Module.train(self.head, mode)
        return self

    @staticmethod
    def _to_tensor(obj):
        # Return the first tensor found in nested structures
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            # prefer last element (common for (features, logits) style), then recurse
            for it in list(obj)[::-1]:
                t = YOLOv8ClsBackbone._to_tensor(it)
                if t is not None:
                    return t
        # try common attrs
        for attr in ("logits",):
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if torch.is_tensor(v):
                    return v
        # Results objects may have probs.data
        if hasattr(obj, "probs"):
            probs = getattr(obj, "probs")
            if hasattr(probs, "data") and torch.is_tensor(probs.data):
                return probs.data
        return None

    def forward(self, x: torch.Tensor):
        # Attempt raw forward on underlying torch model
        out = self.base(x)
        feats = self._to_tensor(out)
        if feats is None:
            # Fallback to Ultralytics high-level API (inference only), yielding Results
            with torch.no_grad():
                results = self._yolo(x, verbose=False)
            if isinstance(results, (list, tuple)) and len(results) > 0:
                t = [self._to_tensor(r) for r in results]
                if all(torch.is_tensor(z) for z in t):
                    feats = torch.stack(t, dim=0).squeeze(1)
        if feats is None:
            raise RuntimeError("YOLOv8 classification forward returned unsupported output; cannot extract logits tensor.")
        # Ensure feats are on the same device as the head
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
        parent = model
        parts = last_name.split('.')
        for p in parts[:-1]:
            parent = parent._modules[p]
        parent._modules[parts[-1]] = nn.Identity()
        return int(last_fc.in_features)


class YOLOv8EarlyFusionPatient(nn.Module):
    """
    Early-fusion variant for YOLOv8 classification models.
    - Adds metadata-derived channels (c_pat) to the RGB input and patches the
      first Conv2d to accept 3 + c_pat channels.
    - Uses the pretrained classification logits (e.g., 1000-d) as features and
      maps them to `out_dim` using a Linear head.
    """
    def __init__(self, model_name: str = "yolov8n-cls", tab_dim: int = 0, c_pat: int = 8,
                 pretrained: bool = True, out_dim: int = 2, pmap_hidden: int = 64, pmap_drop: float = 0.1):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOv8 classification. Install via 'pip install ultralytics'.")

        # Underlying Ultralytics model
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        self._yolo = YOLO(name) if pretrained else YOLO(name)
        self.base = self._yolo.model  # PyTorch model

        # Metadata → channels mapper
        self.c_pat = int(c_pat)
        self.tab_dim = int(tab_dim)
        self.pmap = nn.Sequential(
            nn.Linear(self.tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, self.c_pat)
        ) if self.tab_dim > 0 else None

        # Patch first Conv2d in the stem to accept 3 + c_pat input channels
        self._patch_first_conv(self.base, add_in_channels=self.c_pat)

        # Expose pre-logits by replacing the final classifier with Identity
        in_feats = YOLOv8ClsBackbone._replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3 + self.c_pat, 224, 224)
                out = self.base(dummy)
                t = YOLOv8ClsBackbone._to_tensor(out)
                in_feats = int(t.shape[1])
        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    @staticmethod
    def _find_first_conv_wrapper(model: nn.Module):
        """
        Find a module that has attribute `.conv` being an nn.Conv2d with 3 input channels.
        This matches Ultralytics' Conv wrapper commonly used as the first layer.
        Returns the wrapper module and the inner conv.
        """
        for m in model.modules():
            if hasattr(m, 'conv') and isinstance(getattr(m, 'conv'), nn.Conv2d):
                conv = getattr(m, 'conv')
                if int(getattr(conv, 'in_channels', -1)) == 3:
                    return m, conv
        # fallback: raw first Conv2d
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and int(m.in_channels) == 3:
                return None, m
        return None, None

    @staticmethod
    def _patch_first_conv(model: nn.Module, add_in_channels: int):
        if add_in_channels <= 0:
            return
        wrapper, conv = YOLOv8EarlyFusionPatient._find_first_conv_wrapper(model)
        if conv is None:
            raise RuntimeError("Could not locate first Conv2d to patch for YOLOv8 early fusion")
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
            # replace in the parent module by path
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
        feats = YOLOv8ClsBackbone._to_tensor(out)
        if feats is None:
            with torch.no_grad():
                results = self._yolo(x, verbose=False)
            if isinstance(results, (list, tuple)) and len(results) > 0:
                t = [YOLOv8ClsBackbone._to_tensor(r) for r in results]
                if all(torch.is_tensor(z) for z in t):
                    feats = torch.stack(t, dim=0).squeeze(1)
        if feats is None:
            raise RuntimeError("YOLOv8 classification forward returned unsupported output; cannot extract logits tensor.")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        logits = self.head(feats)
        return feats, logits
