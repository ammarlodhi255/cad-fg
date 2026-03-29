import os
import types
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _load_yolov5_detection(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Load a YOLOv5 detection model via torch.hub.

    If `model_name` points to an existing .pt file, use the 'custom' entrypoint.
    Otherwise, assume it is a standard YOLOv5 model name such as 'yolov5l'.
    """
    name = model_name
    if model_name.endswith(".pt") and os.path.isfile(model_name):
        # Explicit checkpoint path
        path = model_name
        entry = "custom"
    else:
        # Allow "yolov5l" -> "yolov5l.pt" local file first, then hub fallback.
        candidate = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        if os.path.isfile(candidate):
            path = candidate
            entry = "custom"
        else:
            path = model_name  # hub builtin name, e.g. 'yolov5l'
            entry = model_name

    if not hasattr(torch, "hub") or not isinstance(torch.hub, types.ModuleType):
        raise ImportError("torch.hub is required to load YOLOv5 detection models (ultralytics/yolov5).")

    if entry == "custom":
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=path,
            autoshape=False,
            trust_repo=True,
            verbose=False,
        )  # type: ignore
    else:
        model = torch.hub.load(
            "ultralytics/yolov5",
            entry,
            pretrained=pretrained,
            autoshape=False,
            trust_repo=True,
            verbose=False,
        )  # type: ignore
    return model


class _CrossAttnSite(nn.Module):
    """
    Per-site cross-attention block:
      - Treat spatial features as K/V tokens.
      - Treat metadata embedding as a single Q token.
      - Produce a per-channel delta added back to the feature map.
    """

    def __init__(self, in_channels: int, tab_dim: int, attn_dim: int = 256, drop: float = 0.1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.tab_dim = int(tab_dim)
        D = int(attn_dim)

        self.q_proj = (
            nn.Sequential(
                nn.Linear(self.tab_dim, D),
                nn.ReLU(),
                nn.Dropout(drop),
            )
            if self.tab_dim > 0
            else None
        )
        self.k_proj = nn.Linear(self.in_channels, D, bias=False)
        self.v_proj = nn.Linear(self.in_channels, D, bias=False)
        self.out_proj = nn.Linear(D, self.in_channels, bias=False)

    def forward(self, x: torch.Tensor, clin_vec: torch.Tensor) -> torch.Tensor:
        if self.q_proj is None or clin_vec is None:
            return x
        if not torch.is_tensor(x) or x.dim() != 4:
            return x
        B, C, H, W = x.shape
        X = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        K = self.k_proj(X)  # (B, N, D)
        V = self.v_proj(X)  # (B, N, D)
        Q = self.q_proj(clin_vec).unsqueeze(1)  # (B, 1, D)
        attn = torch.softmax((Q @ K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=-1)  # (B, 1, N)
        ctx = attn @ V  # (B, 1, D)
        delta = self.out_proj(ctx.squeeze(1))  # (B, C)
        delta = delta.view(B, C, 1, 1)
        return x + delta


class YOLOv5DetMidFusionCLSCrossAttn(nn.Module):
    """
    YOLOv5 detection wrapper with mid-level cross-attention fusion of tabular metadata.

    The wrapper:
      - Loads a YOLOv5 detection model (typically 'yolov5l').
      - Automatically probes the network to find several 4D feature map sites near the end.
      - At each selected site, applies a _CrossAttnSite block using metadata as the query.

    Forward:
      out = model(x_img, ...)   # standard YOLOv5 detection outputs
      Metadata is injected only via hooks; the public API remains detection-like:
        forward(x_img, clin_vec) -> out
    """

    def __init__(
        self,
        model_name: str = "yolov5l",
        tab_dim: int = 0,
        pretrained: bool = True,
        attn_dim: int = 256,
        drop: float = 0.1,
        nsites: int = 2,
        base_model: nn.Module | None = None,
    ):
        super().__init__()
        self.tab_dim = int(tab_dim)
        self.attn_dim = int(attn_dim)
        self.drop = float(drop)
        self.nsites = int(max(1, nsites))

        # Allow injecting an existing YOLOv5 model (e.g., from a local clone) to
        # keep loss code and architecture in sync. If not provided, fall back to
        # torch.hub loading.
        if base_model is not None:
            self.base = base_model
        else:
            self.base = _load_yolov5_detection(model_name=model_name, pretrained=pretrained)

        self._clin: torch.Tensor | None = None
        self._site_hooks: List[Any] = []
        self._site_blocks: Dict[nn.Module, _CrossAttnSite] = {}

        if self.tab_dim > 0:
            self._attach_sites()

    # ------------------
    # Clinical state API
    # ------------------
    def _set_clin(self, clin: torch.Tensor) -> None:
        self._clin = clin

    def _clear_clin(self) -> None:
        self._clin = None

    # ---------------
    # Site selection
    # ---------------
    def _attach_sites(self) -> None:
        """
        Automatically find several mid/late 4D feature map sites and attach cross-attention hooks.
        """
        cands: List[Tuple[nn.Module, int]] = []
        handles: List[Any] = []

        def hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            try:
                x = inputs[0]
                if torch.is_tensor(x) and x.dim() == 4:
                    cands.append((mod, int(x.shape[1])))
            except Exception:
                pass

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
                    dev, dt = torch.device("cpu"), torch.float32
                dummy = torch.zeros(1, 3, 640, 640, device=dev, dtype=dt)
                _ = self.base(dummy)
        finally:
            for h in handles:
                h.remove()

        if not cands:
            return

        idxs: List[int] = []
        L = len(cands)
        fracs = [0.6, 0.8, 0.9]
        for i in range(min(self.nsites, len(fracs))):
            idxs.append(min(L - 1, max(0, int(fracs[i] * L))))
        idxs = sorted(set(idxs))

        for idx in idxs:
            mod, C = cands[idx]

            # Heuristic: avoid attaching directly to the final Detect head if present.
            cls_name = mod.__class__.__name__.lower()
            if "detect" in cls_name:
                continue

            block = _CrossAttnSite(in_channels=C, tab_dim=self.tab_dim, attn_dim=self.attn_dim, drop=self.drop)
            self._site_blocks[mod] = block

            def _make_hook(site_ref: _CrossAttnSite):
                def _h(_m: nn.Module, inputs: Tuple[torch.Tensor, ...]):
                    if self._clin is None:
                        return None
                    x = inputs[0]
                    if not torch.is_tensor(x) or x.dim() != 4:
                        return None
                    return (site_ref(x, self._clin.to(x.device)),)

                return _h

            try:
                hk = mod.register_forward_pre_hook(_make_hook(block), with_kwargs=False)
                self._site_hooks.append(hk)
            except Exception:
                continue

        if self._site_blocks:
            self._site_block_list = nn.ModuleList(self._site_blocks.values())

    # ----------
    # Interface
    # ----------
    def train(self, mode: bool = True):  # type: ignore[override]
        self.training = mode
        try:
            nn.Module.train(self.base, mode)
        except Exception:
            if not mode:
                self.base.eval()
        if hasattr(self, "_site_block_list"):
            nn.Module.train(self._site_block_list, mode)
        return self

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor, *args: Any, **kwargs: Any):
        """
        Run YOLOv5 detection conditioned on metadata.

        Args:
            x_img: image tensor, shape (B,3,H,W)
            clin_vec: tabular/metadata tensor, shape (B, tab_dim)
            *args, **kwargs: forwarded to the underlying YOLOv5 model (e.g., augment flags).

        Returns:
            In training mode: list of feature maps [p0, p1, p2] where each is [B,C,H,W]
            In eval mode: detections tuple
        """
        self._set_clin(clin_vec)
        # YOLOv5 BaseModel.forward expects just (x, profile=False, visualize=False)
        # Filter kwargs to only pass valid ones
        valid_kwargs = {}
        if "profile" in kwargs:
            valid_kwargs["profile"] = kwargs["profile"]
        if "visualize" in kwargs:
            valid_kwargs["visualize"] = kwargs["visualize"]
        out = self.base(x_img, **valid_kwargs)
        self._clear_clin()
        return out


