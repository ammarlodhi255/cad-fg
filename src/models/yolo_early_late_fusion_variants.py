import os
import types
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


# ----------------------
# Common small utilities
# ----------------------

def _to_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for it in list(obj)[::-1]:
            t = _to_tensor(it)
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


def _replace_last_linear_with_identity(model: nn.Module):
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


def _find_first_conv_wrapper(model: nn.Module):
    """Return (wrapper_module, conv2d) for first 3->C Conv. Wrapper may be None."""
    for m in model.modules():
        if hasattr(m, 'conv') and isinstance(getattr(m, 'conv'), nn.Conv2d):
            conv = getattr(m, 'conv')
            if int(getattr(conv, 'in_channels', -1)) == 3:
                return m, conv
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and int(m.in_channels) == 3:
            return None, m
    return None, None


# -----------------------------
# Early FiLM-at-stem (YOLOv8/11)
# -----------------------------

class _EarlyFiLMStemBase(nn.Module):
    """Base for early FiLM on the first conv outputs; preserves identity at init."""
    def __init__(self, out_dim: int = 2, tab_dim: int = 0, hidden: int = 64, drop: float = 0.1,
                 gamma_scale: float = 0.1):
        super().__init__()
        self.tab_dim = int(tab_dim)
        self.gamma_scale = float(gamma_scale)
        self._clin = None
        self._hook = None

    def _attach_film_hook(self, model: nn.Module):
        wrapper, conv = _find_first_conv_wrapper(model)
        if conv is None:
            raise RuntimeError("EarlyFiLMStem: could not locate first Conv2d")
        C = int(conv.out_channels)
        self._film = nn.Sequential(
            nn.Linear(self.tab_dim, max(1, int(getattr(self, '_film_hidden', 64)))),
            nn.ReLU(), nn.Dropout(getattr(self, '_film_drop', 0.1)),
            nn.Linear(max(1, int(getattr(self, '_film_hidden', 64))), 2 * C),
        ) if self.tab_dim > 0 else None
        # identity-preserving init on last layer
        if self._film is not None:
            last = None
            for m in self._film.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

        def _hook_fn(_mod, _inputs, output):
            if self._film is None or self._clin is None:
                return output
            x = output
            if not torch.is_tensor(x) or x.dim() != 4:
                return output
            B, Co, H, W = x.shape
            gb = self._film(self._clin.to(x.device))
            g, b = gb.chunk(2, dim=1)
            g = torch.tanh(g) * self.gamma_scale
            return x * (1.0 + g.view(B, Co, 1, 1)) + b.view(B, Co, 1, 1)

        target = wrapper if (wrapper is not None and hasattr(wrapper, 'conv')) else conv
        try:
            self._hook = target.register_forward_hook(_hook_fn, with_kwargs=False)
        except TypeError:
            self._hook = target.register_forward_hook(_hook_fn)

    def _set_clin(self, clin):
        self._clin = clin

    def _clear_clin(self):
        self._clin = None


class YOLOv8EarlyFiLMStem(_EarlyFiLMStemBase):
    def __init__(self, model_name: str = "yolov8n-cls", tab_dim: int = 0, pretrained: bool = True,
                 out_dim: int = 2, hidden: int = 64, drop: float = 0.1, gamma_scale: float = 0.1):
        super().__init__(out_dim=out_dim, tab_dim=tab_dim, hidden=hidden, drop=drop, gamma_scale=gamma_scale)
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOv8")
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        self._yolo = YOLO(name) if pretrained else YOLO(name)
        self.base = self._yolo.model
        self._film_hidden = hidden
        self._film_drop = drop
        self._attach_film_hook(self.base)
        in_feats = _replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                t = _to_tensor(self.base(torch.zeros(1, 3, 224, 224)))
                in_feats = int(t.shape[1])
        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        try:
            nn.Module.train(self.base, mode)
        except Exception:
            if not mode:
                self.base.eval()
        nn.Module.train(self.head, mode)
        if getattr(self, '_film', None) is not None:
            nn.Module.train(self._film, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        self._clear_clin()
        feats = _to_tensor(out)
        if feats is None:
            raise RuntimeError("YOLOv8EarlyFiLMStem: cannot extract logits tensor")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        logits = self.head(feats)
        return feats, logits


class YOLO11EarlyFiLMStem(_EarlyFiLMStemBase):
    def __init__(self, model_name: str = "yolo11n-cls", tab_dim: int = 0, pretrained: bool = True,
                 out_dim: int = 2, hidden: int = 64, drop: float = 0.1, gamma_scale: float = 0.1):
        super().__init__(out_dim=out_dim, tab_dim=tab_dim, hidden=hidden, drop=drop, gamma_scale=gamma_scale)
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO11")
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        self._yolo = YOLO(name) if pretrained else YOLO(name)
        self.base = self._yolo.model
        self._film_hidden = hidden
        self._film_drop = drop
        self._attach_film_hook(self.base)
        in_feats = _replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                t = _to_tensor(self.base(torch.zeros(1, 3, 224, 224)))
                in_feats = int(t.shape[1])
        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        try:
            nn.Module.train(self.base, mode)
        except Exception:
            if not mode:
                self.base.eval()
        nn.Module.train(self.head, mode)
        if getattr(self, '_film', None) is not None:
            nn.Module.train(self._film, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        self._clear_clin()
        feats = _to_tensor(out)
        if feats is None:
            raise RuntimeError("YOLO11EarlyFiLMStem: cannot extract logits tensor")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        logits = self.head(feats)
        return feats, logits


class YOLOv5EarlyFiLMStem(_EarlyFiLMStemBase):
    def __init__(self, model_name: str = "yolov5s-cls", tab_dim: int = 0, pretrained: bool = True,
                 out_dim: int = 2, hidden: int = 64, drop: float = 0.1, gamma_scale: float = 0.1):
        super().__init__(out_dim=out_dim, tab_dim=tab_dim, hidden=hidden, drop=drop, gamma_scale=gamma_scale)
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        if not os.path.isfile(name):
            raise FileNotFoundError(f"Weights '{name}' not found for YOLOv5")
        if not hasattr(torch, 'hub') or not isinstance(torch.hub, types.ModuleType):
            raise ImportError("torch.hub is required for YOLOv5 classification (ultralytics/yolov5)")
        self.base = torch.hub.load('ultralytics/yolov5', 'custom', path=name, autoshape=False, trust_repo=True, verbose=False)  # type: ignore
        self._film_hidden = hidden
        self._film_drop = drop
        self._attach_film_hook(self.base)
        in_feats = _replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                t = self.base(torch.zeros(1, 3, 224, 224))
                in_feats = int(_to_tensor(t).shape[1])
        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        try:
            nn.Module.train(self.base, mode)
        except Exception:
            if not mode:
                self.base.eval()
        nn.Module.train(self.head, mode)
        if getattr(self, '_film', None) is not None:
            nn.Module.train(self._film, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        self._clear_clin()
        feats = _to_tensor(out)
        if feats is None:
            raise RuntimeError("YOLOv5EarlyFiLMStem: cannot extract logits tensor")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        logits = self.head(feats)
        return feats, logits


# -----------------
# Late FiLM-on-feat
# -----------------

class LateFiLMHead(nn.Module):
    """
    Late fusion head with FiLM on image features: z' = z * (1 + gamma(meta)) + beta(meta) -> classifier.
    Use with a separate tabular MLP that maps metadata to a compact embedding fed here.
    """
    def __init__(self, img_dim: int, clin_dim: int, out_dim: int = 2, gamma_scale: float = 0.1):
        super().__init__()
        self.img_dim = int(img_dim)
        self.gammabeta = nn.Linear(int(clin_dim), 2 * int(img_dim))
        # identity-ish init for stability
        nn.init.zeros_(self.gammabeta.weight)
        nn.init.zeros_(self.gammabeta.bias)
        self.gamma_scale = float(gamma_scale)
        self.classifier = nn.Sequential(
            nn.Linear(int(img_dim), 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, img_feats: torch.Tensor, clin_embed: torch.Tensor):
        z = nn.functional.normalize(img_feats, dim=1)
        gb = self.gammabeta(clin_embed)
        g, b = gb.chunk(2, dim=1)
        g = torch.tanh(g) * self.gamma_scale
        z_mod = z * (1.0 + g) + b
        return self.classifier(z_mod)

