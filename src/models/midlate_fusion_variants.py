import torch
import torch.nn as nn

# Reuse helpers from existing backbones/variants
from .convnext_backbone import _load_convnext
from .yolov8_cls_backbone import YOLOv8ClsBackbone

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

import types, os


# ---------- Common utilities ----------

def _find_prepool_module(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            return m
    # Fallback: avgpool by name
    for name, m in model.named_modules():
        if 'avgpool' in name.lower():
            return m
    raise RuntimeError("Could not locate AdaptiveAvgPool2d in model")


def _dummy_like_model(model: nn.Module, input_channels: int = 3, size: int = 224):
    try:
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
    except Exception:
        dev, dt = torch.device('cpu'), torch.float32
    return torch.zeros(1, input_channels, size, size, device=dev, dtype=dt)


def _infer_prepool_channels(base: nn.Module, input_channels: int = 3, size: int = 224) -> int:
    pre = _find_prepool_module(base)
    info = {'c': None}

    def _hook(_mod, inputs):
        x = inputs[0]
        info['c'] = int(x.shape[1])
        return None

    h = pre.register_forward_pre_hook(_hook, with_kwargs=False)
    try:
        base.eval()
        with torch.no_grad():
            dummy = _dummy_like_model(base, input_channels, size)
            _ = base(dummy)
        if info['c'] is None:
            raise RuntimeError("Failed to infer pre-pool channels")
        return int(info['c'])
    finally:
        h.remove()


class _LateHead(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int, out_dim: int = 2, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + clin_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, img_feats, clin_embed):
        z = nn.functional.normalize(img_feats, dim=1)
        return self.net(torch.cat([z, clin_embed], dim=1))


# ---------- YOLOv8 Mid+Late (Dual) ----------

class _YOLOv8DualBase(nn.Module):
    def __init__(self, model_name: str, tab_dim: int, out_dim: int = 2,
                 clin_hid: int = 64, clin_out: int = 32):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOv8 models.")
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        self._yolo = YOLO(name)
        self.base = self._yolo.model
        self.prepool = _find_prepool_module(self.base)
        self.prepool_C = _infer_prepool_channels(self.base, input_channels=3)
        # Expose pre-logits by replacing last Linear with Identity
        in_feats = YOLOv8ClsBackbone._replace_last_linear_with_identity(self.base) if hasattr(YOLOv8ClsBackbone, '_replace_last_linear_with_identity') else None
        if in_feats is None:
            with torch.no_grad():
                feats = YOLOv8ClsBackbone._to_tensor(self.base(_dummy_like_model(self.base)))
                in_feats = int(feats.shape[1])
        self.feat_dim = int(in_feats)
        # late fusion components
        self.clin_mlp = nn.Sequential(
            nn.Linear(int(tab_dim), clin_hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(clin_hid, clin_out), nn.ReLU()
        ) if tab_dim > 0 else None
        self.late_head = _LateHead(img_dim=self.feat_dim, clin_dim=clin_out, out_dim=out_dim)
        self._clin = None

    def _set_clin(self, clin): self._clin = clin
    def _clear_clin(self): self._clin = None

    def _extract_logits(self, out):
        # Now returns pre-logits features after Identity replacement
        feats = YOLOv8ClsBackbone._to_tensor(out)
        if feats is None:
            raise RuntimeError("YOLOv8Dual: unable to extract logits tensor")
        return feats

    def _late(self, feats, clin):
        emb = self.clin_mlp(clin) if self.clin_mlp is not None else torch.zeros(feats.size(0), 0, device=feats.device)
        return self.late_head(feats, emb)

    # Avoid triggering Ultralytics training engine; only set PyTorch modes
    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        if isinstance(self.base, nn.Module):
            try:
                nn.Module.train(self.base, mode)
            except Exception:
                # fallback: keep eval for safety
                if not mode:
                    try:
                        self.base.eval()
                    except Exception:
                        pass
        if isinstance(getattr(self, 'clin_mlp', None), nn.Module):
            nn.Module.train(self.clin_mlp, mode)
        if isinstance(getattr(self, 'late_head', None), nn.Module):
            nn.Module.train(self.late_head, mode)
        # best-effort for optional mid modules
        for name in ("film", "gate", "q_proj", "k_proj", "v_proj", "out_proj", "down", "to_beta", "to_gamma"):
            m = getattr(self, name, None)
            if isinstance(m, nn.Module):
                nn.Module.train(m, mode)
        return self


class YOLOv8MidLateFiLM(_YOLOv8DualBase):
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, out_dim: int = 2,
                 mid_hid: int = 64):
        super().__init__(model_name=model_name, tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.film = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, 2 * C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.film is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        gb = self.film(self._clin.to(x.device))
        gamma, beta = gb.chunk(2, dim=1)
        return (x * (1.0 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1),)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


class YOLOv8MidLateSEGate(_YOLOv8DualBase):
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, out_dim: int = 2,
                 mid_hid: int = 64):
        super().__init__(model_name=model_name, tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.gate = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.gate is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        g = torch.sigmoid(self.gate(self._clin.to(x.device))).view(B, C, 1, 1)
        return (x * g,)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self._extract_logits(self.base(x_img))
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


# ---------- YOLOv5 Mid+Late (Dual) ----------

class _YOLOv5DualBase(nn.Module):
    def __init__(self, model_name: str, tab_dim: int, out_dim: int = 2,
                 clin_hid: int = 64, clin_out: int = 32):
        super().__init__()
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        if not os.path.isfile(name):
            raise FileNotFoundError(f"Weights '{name}' not found for YOLOv5.")
        if not hasattr(torch, 'hub') or not isinstance(torch.hub, types.ModuleType):
            raise ImportError("torch.hub is required for YOLOv5 models.")
        self.base = torch.hub.load('ultralytics/yolov5', 'custom', path=name, autoshape=False, trust_repo=True, verbose=False)  # type: ignore
        self.prepool = _find_prepool_module(self.base)
        self.prepool_C = _infer_prepool_channels(self.base, input_channels=3)
        # Replace last Linear with Identity to expose pre-logits
        # Local helper to avoid cyclic import
        def _replace_last_linear_with_identity(m: nn.Module):
            last_name = None
            last_fc = None
            for nm, mod in m.named_modules():
                if isinstance(mod, nn.Linear):
                    last_name, last_fc = nm, mod
            if last_fc is None or last_name is None:
                return None
            parent = m
            parts = last_name.split('.')
            for p in parts[:-1]:
                parent = parent._modules[p]
            parent._modules[parts[-1]] = nn.Identity()
            return int(last_fc.in_features)

        in_feats = _replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                in_feats = int(self.base(_dummy_like_model(self.base)).shape[1])
        self.feat_dim = int(in_feats)
        # late
        self.clin_mlp = nn.Sequential(
            nn.Linear(int(tab_dim), clin_hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(clin_hid, clin_out), nn.ReLU()
        ) if tab_dim > 0 else None
        self.late_head = _LateHead(img_dim=self.feat_dim, clin_dim=clin_out, out_dim=out_dim)
        self._clin = None

    def _set_clin(self, clin): self._clin = clin
    def _clear_clin(self): self._clin = None
    def _late(self, feats, clin):
        emb = self.clin_mlp(clin) if self.clin_mlp is not None else torch.zeros(feats.size(0), 0, device=feats.device)
        return self.late_head(feats, emb)


class YOLOv5MidLateFiLM(_YOLOv5DualBase):
    def __init__(self, model_name: str = "yolov5s-cls", tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64):
        super().__init__(model_name=model_name, tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.film = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, 2 * C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.film is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        gb = self.film(self._clin.to(x.device))
        gamma, beta = gb.chunk(2, dim=1)
        return (x * (1.0 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1),)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self.base(x_img)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


class YOLOv5MidLateSEGate(_YOLOv5DualBase):
    def __init__(self, model_name: str = "yolov5s-cls", tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64):
        super().__init__(model_name=model_name, tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.gate = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.gate is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        g = torch.sigmoid(self.gate(self._clin.to(x.device))).view(B, C, 1, 1)
        return (x * g,)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self.base(x_img)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


# ---------- ConvNeXt Mid+Late (Dual) ----------

class _ConvNeXtDualBase(nn.Module):
    def __init__(self, variant: str, tab_dim: int, out_dim: int = 2, clin_hid: int = 64, clin_out: int = 32):
        super().__init__()
        m, feat_dim = _load_convnext(variant, pretrained=True)
        self.backbone = m
        self.backbone.classifier = nn.Identity()  # keep features and avgpool
        self.prepool = _find_prepool_module(self.backbone)
        self.prepool_C = _infer_prepool_channels(self.backbone, input_channels=3)
        self.feat_dim = feat_dim
        self.clin_mlp = nn.Sequential(
            nn.Linear(int(tab_dim), clin_hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(clin_hid, clin_out), nn.ReLU()
        ) if tab_dim > 0 else None
        self.late_head = _LateHead(img_dim=self.feat_dim, clin_dim=clin_out, out_dim=out_dim)
        self._clin = None

    def _set_clin(self, clin): self._clin = clin
    def _clear_clin(self): self._clin = None
    def _late(self, feats, clin):
        emb = self.clin_mlp(clin) if self.clin_mlp is not None else torch.zeros(feats.size(0), 0, device=feats.device)
        return self.late_head(feats, emb)

    def _forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        return feats


class ConvNeXtTinyMidLateFiLM(_ConvNeXtDualBase):
    def __init__(self, tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64):
        super().__init__(variant="tiny", tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.film = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, 2 * C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.film is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        gb = self.film(self._clin.to(x.device))
        g, b = gb.chunk(2, dim=1)
        return (x * (1.0 + g.view(B, C, 1, 1)) + b.view(B, C, 1, 1),)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self._forward_feats(x_img)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


class ConvNeXtBaseMidLateFiLM(_ConvNeXtDualBase):
    def __init__(self, tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64):
        super().__init__(variant="base", tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.film = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, 2 * C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.film is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        gb = self.film(self._clin.to(x.device))
        g, b = gb.chunk(2, dim=1)
        return (x * (1.0 + g.view(B, C, 1, 1)) + b.view(B, C, 1, 1),)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self._forward_feats(x_img)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


class ConvNeXtTinyMidLateGate(_ConvNeXtDualBase):
    def __init__(self, tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64):
        super().__init__(variant="tiny", tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.gate = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.gate is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        g = torch.sigmoid(self.gate(self._clin.to(x.device))).view(B, C, 1, 1)
        return (x * g,)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self._forward_feats(x_img)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


class ConvNeXtBaseMidLateGate(_ConvNeXtDualBase):
    def __init__(self, tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64):
        super().__init__(variant="base", tab_dim=tab_dim, out_dim=out_dim)
        C = int(self.prepool_C)
        self.gate = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, C)
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.gate is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        g = torch.sigmoid(self.gate(self._clin.to(x.device))).view(B, C, 1, 1)
        return (x * g,)

    def forward(self, x_img, clin_vec):
        self._set_clin(clin_vec)
        feats = self._forward_feats(x_img)
        self._clear_clin()
        logits = self._late(feats, clin_vec)
        return feats, logits


# ---------- VGG19 Mid+Late (Dual) ----------

class VGG19MidLateFiLM(nn.Module):
    def __init__(self, tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64, clin_hid: int = 64, clin_out: int = 32):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        m = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features  # 4096
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        self.backbone = m
        self.prepool = _find_prepool_module(self.backbone)
        self.prepool_C = _infer_prepool_channels(self.backbone, input_channels=3)
        self.feat_dim = in_feats
        self.film = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, 2 * int(self.prepool_C))
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)
        self.clin_mlp = nn.Sequential(
            nn.Linear(int(tab_dim), clin_hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(clin_hid, clin_out), nn.ReLU()
        ) if tab_dim > 0 else None
        self.late_head = _LateHead(img_dim=self.feat_dim, clin_dim=clin_out, out_dim=out_dim)
        self._clin = None

    def _hook_fn(self, _mod, inputs):
        if self.film is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        gb = self.film(self._clin.to(x.device))
        g, b = gb.chunk(2, dim=1)
        return (x * (1.0 + g.view(B, C, 1, 1)) + b.view(B, C, 1, 1),)

    def _forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        feats = self.backbone.classifier(x)
        return feats

    def forward(self, x_img, clin_vec):
        self._clin = clin_vec
        feats = self._forward_feats(x_img)
        self._clin = None
        emb = self.clin_mlp(clin_vec) if self.clin_mlp is not None else torch.zeros(feats.size(0), 0, device=feats.device)
        logits = self.late_head(feats, emb)
        return feats, logits


class VGG19MidLateGate(nn.Module):
    def __init__(self, tab_dim: int = 0, out_dim: int = 2, mid_hid: int = 64, clin_hid: int = 64, clin_out: int = 32):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        m = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features  # 4096
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        self.backbone = m
        self.prepool = _find_prepool_module(self.backbone)
        self.prepool_C = _infer_prepool_channels(self.backbone, input_channels=3)
        self.feat_dim = in_feats
        self.gate = nn.Sequential(
            nn.Linear(int(tab_dim), mid_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_hid, int(self.prepool_C))
        ) if tab_dim > 0 else None
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)
        self.clin_mlp = nn.Sequential(
            nn.Linear(int(tab_dim), clin_hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(clin_hid, clin_out), nn.ReLU()
        ) if tab_dim > 0 else None
        self.late_head = _LateHead(img_dim=self.feat_dim, clin_dim=clin_out, out_dim=out_dim)
        self._clin = None

    def _hook_fn(self, _mod, inputs):
        if self.gate is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        g = torch.sigmoid(self.gate(self._clin.to(x.device))).view(B, C, 1, 1)
        return (x * g,)

    def _forward_feats(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        feats = self.backbone.classifier(x)
        return feats

    def forward(self, x_img, clin_vec):
        self._clin = clin_vec
        feats = self._forward_feats(x_img)
        self._clin = None
        emb = self.clin_mlp(clin_vec) if self.clin_mlp is not None else torch.zeros(feats.size(0), 0, device=feats.device)
        logits = self.late_head(feats, emb)
        return feats, logits
