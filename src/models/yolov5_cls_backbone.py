import os
import torch
import torch.nn as nn
import types


class YOLOv5ClsBackbone(nn.Module):
    """
    Image-only wrapper for YOLOv5 classification via torch.hub (e.g., 'yolov5s-cls').
    - Requires torch.hub access to 'ultralytics/yolov5' or a locally cached copy.
    - Uses pretrained logits (typically 1000-d) as features, then maps to out_dim via a Linear head.
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
        # Use trust_repo='check' so it will prompt only on first use of an untrusted repo.
        if not hasattr(torch, 'hub') or not isinstance(torch.hub, types.ModuleType):
            raise ImportError("torch.hub is required to load YOLOv5 classification models (ultralytics/yolov5).")
        self.base = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=name,
            autoshape=False,
            trust_repo='check',
            verbose=False,
        )  # type: ignore

        # Infer base logits dim via last Linear or dummy forward
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
