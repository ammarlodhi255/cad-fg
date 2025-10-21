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
    - Uses the pretrained 1000-class logits as features, then maps to out_dim via a Linear head.
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

        # Infer base logits dim from the last Linear if possible; fallback to a dummy forward.
        base_dim = None
        for m in self.base.modules():
            if isinstance(m, nn.Linear):
                base_dim = m.out_features
        if base_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.base(dummy)
                base_dim = int(out.shape[1])

        self.feat_dim = base_dim
        self.head = nn.Linear(base_dim, out_dim)

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
