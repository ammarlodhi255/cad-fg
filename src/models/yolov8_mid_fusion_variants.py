import torch
import torch.nn as nn

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

from .yolov8_cls_backbone import YOLOv8ClsBackbone


def _find_prepool_module(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            return m
    # Fallback: look for attribute average pooling layers by name
    for name, m in model.named_modules():
        if 'avgpool' in name.lower() and isinstance(m, nn.Module):
            return m
    raise RuntimeError("Could not find AdaptiveAvgPool2d (pre-pool) in YOLOv8 classifier")


def _infer_prepool_channels(base: nn.Module, input_channels: int = 3, size: int = 224) -> int:
    m = _find_prepool_module(base)
    ch = {'c': None}

    def _hook(_mod, inputs):
        x = inputs[0]
        ch['c'] = int(x.shape[1])
        return None

    h = m.register_forward_pre_hook(_hook, with_kwargs=False)
    try:
        base.eval()
        with torch.no_grad():
            # Create dummy on same device/dtype as model parameters to avoid CPU/GPU mismatch
            try:
                p = next(base.parameters())
                dev, dt = p.device, p.dtype
            except Exception:
                dev, dt = torch.device('cpu'), torch.float32
            dummy = torch.zeros(1, input_channels, size, size, device=dev, dtype=dt)
            _ = base(dummy)
        if ch['c'] is None:
            raise RuntimeError("Pre-pool channels could not be inferred")
        return int(ch['c'])
    finally:
        h.remove()


def _replace_last_linear_with_identity(model: nn.Module):
    """Replace the last nn.Linear in `model` with nn.Identity.
    Returns the in_features of that layer, or None if not found.
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


class _YOLOv8MidBase(nn.Module):
    def __init__(self, model_name: str, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOv8 mid-fusion. Install via 'pip install ultralytics'.")
        name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        self._yolo = YOLO(name) if pretrained else YOLO(name)
        self.base = self._yolo.model
        self.tab_dim = int(tab_dim)

        # Locate pre-pool and infer its channel size
        self.prepool = _find_prepool_module(self.base)
        self.prepool_C = _infer_prepool_channels(self.base, input_channels=3)

        # Expose pre-logits feature width
        in_feats = _replace_last_linear_with_identity(self.base)
        if in_feats is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.base(dummy)
                feats = YOLOv8ClsBackbone._to_tensor(out)
                in_feats = int(feats.shape[1])
        self.feat_dim = int(in_feats)
        self.head = nn.Linear(self.feat_dim, out_dim)

        # Storage for current clinical vector used by hooks
        self._clin = None

    def _set_clin(self, clin: torch.Tensor):
        self._clin = clin

    def _clear_clin(self):
        self._clin = None

    def _extract_logits(self, out):
        # Now extracts pre-logits features after Identity replacement
        feats = YOLOv8ClsBackbone._to_tensor(out)
        if feats is None:
            with torch.no_grad():
                results = self._yolo(out, verbose=False)  # this path should not be used in practice
            if isinstance(results, (list, tuple)) and len(results) > 0:
                t = [YOLOv8ClsBackbone._to_tensor(r) for r in results]
                if all(torch.is_tensor(z) for z in t):
                    feats = torch.stack(t, dim=0).squeeze(1)
        if feats is None:
            raise RuntimeError("YOLOv8 mid-fusion: cannot extract feature tensor from model output")
        if feats.device != self.head.weight.device:
            feats = feats.to(self.head.weight.device)
        return feats


class YOLOv8MidFusionResidualFiLM(_YOLOv8MidBase):
    """
    Apply channel-wise FiLM (gamma, beta) to the pre-pool tensor.
    y = x * (1 + gamma) + beta, where gamma/beta are from an MLP(meta).
    """
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 hidden: int = 64, drop: float = 0.1):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        C = int(self.prepool_C)
        self.film = nn.Sequential(
            nn.Linear(self.tab_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 2 * C)
        ) if self.tab_dim > 0 else None
        # Identity-preserving init: last linear zeros -> gamma=0, beta=0
        if self.film is not None:
            last = None
            for m in self.film.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        self.gamma_scale = 0.1
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.film is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        gb = self.film(self._clin.to(x.device))  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * self.gamma_scale
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return (x * (1.0 + gamma) + beta,)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        if self.film is not None:
            nn.Module.train(self.film, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits


class YOLOv8MidFusionSEGate(_YOLOv8MidBase):
    """SE-style gating at pre-pool: x = x * sigmoid(g(meta))."""
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 hidden: int = 64, drop: float = 0.1):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        C = int(self.prepool_C)
        self.gate = nn.Sequential(
            nn.Linear(self.tab_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, C)
        ) if self.tab_dim > 0 else None
        # Identity-approx init: sigmoid(bias) ≈ 1
        if self.gate is not None:
            last = None
            for m in self.gate.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                with torch.no_grad():
                    last.bias.fill_(4.0)
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.gate is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        g = torch.sigmoid(self.gate(self._clin.to(x.device))).view(B, C, 1, 1)
        return (x * g,)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        if self.gate is not None:
            nn.Module.train(self.gate, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits


class YOLOv8MidFusionCLSCrossAttn(_YOLOv8MidBase):
    """
    Cross-attention from a meta-derived query to spatial tokens just before pooling.
    Produces a delta vector in channel space, added residually to the pre-pool tensor.
    """
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 attn_dim: int = 256, drop: float = 0.1):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        C = int(self.prepool_C)
        D = int(attn_dim)
        self.q_proj = nn.Sequential(
            nn.Linear(self.tab_dim, D), nn.ReLU(), nn.Dropout(drop)
        ) if self.tab_dim > 0 else None
        self.k_proj = nn.Linear(C, D, bias=False)
        self.v_proj = nn.Linear(C, D, bias=False)
        self.out_proj = nn.Linear(D, C, bias=False)
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.q_proj is None or self._clin is None:
            return None
        x = inputs[0]  # (B, C, H, W)
        B, C, H, W = x.shape
        X = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        K = self.k_proj(X)  # (B, N, D)
        V = self.v_proj(X)  # (B, N, D)
        Q = self.q_proj(self._clin.to(x.device)).unsqueeze(1)  # (B, 1, D)
        attn = torch.softmax((Q @ K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=-1)  # (B,1,N)
        ctx = attn @ V  # (B,1,D)
        delta = self.out_proj(ctx.squeeze(1))  # (B, C)
        delta = delta.view(B, C, 1, 1)
        return (x + delta,)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        if self.q_proj is not None:
            nn.Module.train(self.q_proj, mode)
        nn.Module.train(self.k_proj, mode)
        nn.Module.train(self.v_proj, mode)
        nn.Module.train(self.out_proj, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits


class YOLOv8MidFusionMetaAdapter(_YOLOv8MidBase):
    """
    Meta-adapter: bottleneck MLP on meta to produce per-channel bias (and optional gain).
    Uses reduction ratio r << C.
    """
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 r: int = 64, drop: float = 0.1, use_gain: bool = False):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        C = int(self.prepool_C)
        R = max(1, int(r))
        self.use_gain = bool(use_gain)
        # meta -> bottleneck -> (beta) [and (gamma) if enabled]
        self.down = nn.Sequential(
            nn.Linear(self.tab_dim, R), nn.ReLU(), nn.Dropout(drop)
        ) if self.tab_dim > 0 else None
        self.to_beta = nn.Linear(R, C)
        self.to_gamma = nn.Linear(R, C) if self.use_gain else None
        # Identity-preserving init
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)
        if self.to_gamma is not None:
            nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        self.gamma_scale = 0.1
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.down is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        z = self.down(self._clin.to(x.device))  # (B, R)
        beta = self.to_beta(z).view(B, C, 1, 1)
        if self.use_gain:
            gamma = self.to_gamma(z)
            gamma = torch.tanh(gamma) * self.gamma_scale
            gamma = gamma.view(B, C, 1, 1)
            return (x * (1.0 + gamma) + beta,)
        else:
            return (x + beta,)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        if self.down is not None:
            nn.Module.train(self.down, mode)
        nn.Module.train(self.to_beta, mode)
        if self.to_gamma is not None:
            nn.Module.train(self.to_gamma, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits


class YOLOv8MidFusionSpatialGate(_YOLOv8MidBase):
    """
    Spatial gating: meta -> per-channel weights w(meta) and scalar bias b(meta),
    mask m = sigmoid(sum_c w_c x_c + b), output = x * m.
    """
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 drop: float = 0.1):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        C = int(self.prepool_C)
        self.w_meta = nn.Linear(self.tab_dim, C) if self.tab_dim > 0 else None
        self.b_meta = nn.Sequential(nn.Dropout(drop), nn.Linear(self.tab_dim, 1)) if self.tab_dim > 0 else None
        if self.w_meta is not None:
            nn.init.zeros_(self.w_meta.weight); nn.init.zeros_(self.w_meta.bias)
        if self.b_meta is not None:
            lin = list(self.b_meta.modules())[-1]
            if isinstance(lin, nn.Linear):
                nn.init.zeros_(lin.weight); nn.init.constant_(lin.bias, 4.0)
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.w_meta is None or self.b_meta is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        w = self.w_meta(self._clin.to(x.device)).view(B, C, 1, 1)
        b = self.b_meta(self._clin.to(x.device)).view(B, 1, 1, 1)
        m = torch.sigmoid((x * w).sum(dim=1, keepdim=True) + b)
        return (x * m,)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        if self.w_meta is not None:
            nn.Module.train(self.w_meta, mode)
        if self.b_meta is not None:
            nn.Module.train(self.b_meta, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits


class YOLOv8MidFusionHyper1x1(_YOLOv8MidBase):
    """
    Hyper 1x1: meta produces r-dim scaling s(meta); apply low-rank 1x1 mixer
    delta = U( (V x) ⊙ tanh(s) ), output = x + delta.
    Identity at init via s=0.
    """
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 rank: int = None, drop: float = 0.1):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        C = int(self.prepool_C)
        r = rank if rank is not None else max(1, min(64, C // 8))
        self.v_conv = nn.Conv2d(C, r, kernel_size=1, bias=False)
        self.u_conv = nn.Conv2d(r, C, kernel_size=1, bias=False)
        self.s_meta = nn.Sequential(nn.Dropout(drop), nn.Linear(self.tab_dim, r)) if self.tab_dim > 0 else None
        if self.s_meta is not None:
            lin = list(self.s_meta.modules())[-1]
            if isinstance(lin, nn.Linear):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        nn.init.kaiming_normal_(self.v_conv.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.u_conv.weight, nonlinearity="linear")
        self._hook = self.prepool.register_forward_pre_hook(self._hook_fn, with_kwargs=False)

    def _hook_fn(self, _mod, inputs):
        if self.s_meta is None or self._clin is None:
            return None
        x = inputs[0]
        B, C, H, W = x.shape
        z1 = self.v_conv(x)
        s = self.s_meta(self._clin.to(x.device)).view(B, z1.size(1), 1, 1)
        s = torch.tanh(s)
        z1s = z1 * s
        delta = self.u_conv(z1s)
        return (x + delta,)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        nn.Module.train(self.v_conv, mode)
        nn.Module.train(self.u_conv, mode)
        if self.s_meta is not None:
            nn.Module.train(self.s_meta, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits


class YOLOv8MidFusionMultiSiteFiLM(_YOLOv8MidBase):
    """
    Multi-site FiLM applied at several 4D feature-producing modules before pooling.
    Sites are auto-selected by execution order during a dummy pass.
    """
    def __init__(self, model_name: str = "yolov8s-cls", tab_dim: int = 0, pretrained: bool = True, out_dim: int = 2,
                 nsites: int = 2, hidden: int = 64, drop: float = 0.1, gamma_scale: float = 0.1):
        super().__init__(model_name=model_name, tab_dim=tab_dim, pretrained=pretrained, out_dim=out_dim)
        self.nsites = int(max(1, nsites))
        self.hidden = int(hidden)
        self.drop = float(drop)
        self.gamma_scale = float(gamma_scale)
        self._site_hooks = []
        self._site_mlps = {}
        if self.tab_dim > 0:
            self._attach_sites()

    def _attach_sites(self):
        cands = []
        handles = []
        def hook(mod, inputs):
            try:
                x = inputs[0]
                if torch.is_tensor(x) and x.dim() == 4:
                    cands.append((mod, int(x.shape[1])))
            except Exception:
                pass
            return None
        for m in self.base.modules():
            try:
                h = m.register_forward_pre_hook(hook, with_kwargs=False)
                handles.append(h)
            except Exception:
                continue
        try:
            self.base.eval()
            with torch.no_grad():
                try:
                    p = next(self.base.parameters())
                    dev, dt = p.device, p.dtype
                except Exception:
                    dev, dt = torch.device('cpu'), torch.float32
                dummy = torch.zeros(1, 3, 224, 224, device=dev, dtype=dt)
                _ = self.base(dummy)
        finally:
            for h in handles:
                h.remove()

        if not cands:
            return
        idxs = []
        L = len(cands)
        fracs = [0.35, 0.6, 0.8]
        for i in range(min(self.nsites, len(fracs))):
            idxs.append(min(L - 1, max(0, int(fracs[i] * L))))
        idxs = sorted(set(idxs))

        for idx in idxs:
            mod, C = cands[idx]
            mlp = nn.Sequential(
                nn.Linear(self.tab_dim, self.hidden), nn.ReLU(), nn.Dropout(self.drop),
                nn.Linear(self.hidden, 2 * C)
            )
            last = None
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            self._site_mlps[mod] = mlp

            def _make_hook(mlp_ref):
                def _h(_m, inputs):
                    if self._clin is None:
                        return None
                    x = inputs[0]
                    if not torch.is_tensor(x) or x.dim() != 4:
                        return None
                    B, Cx, H, W = x.shape
                    gb = mlp_ref(self._clin.to(x.device))
                    # Robustness: adapt mlp output width to current channel count if it differs
                    need = 2 * Cx
                    have = int(gb.size(1))
                    if have != need:
                        if have > need:
                            gb = gb[:, :need]
                        else:
                            pad = torch.zeros(B, need - have, device=gb.device, dtype=gb.dtype)
                            gb = torch.cat([gb, pad], dim=1)
                    gamma, beta = gb.chunk(2, dim=1)
                    gamma = torch.tanh(gamma) * self.gamma_scale
                    gamma = gamma.view(B, Cx, 1, 1)
                    beta = beta.view(B, Cx, 1, 1)
                    return (x * (1.0 + gamma) + beta,)
                return _h

            try:
                hk = mod.register_forward_pre_hook(_make_hook(mlp), with_kwargs=False)
                self._site_hooks.append(hk)
            except Exception:
                continue

        self._mlp_list = nn.ModuleList(self._site_mlps.values())

    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        nn.Module.train(self.base, mode)
        nn.Module.train(self.head, mode)
        if hasattr(self, '_mlp_list'):
            nn.Module.train(self._mlp_list, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        self._set_clin(clin_vec)
        out = self.base(x_img)
        feats = self._extract_logits(out)
        self._clear_clin()
        logits = self.head(feats)
        return feats, logits
