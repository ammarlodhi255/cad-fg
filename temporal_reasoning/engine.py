import time
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.trainloop.metrics import metrics_from_probs


def train_one_epoch_temporal(model, optimizer, loader, device, use_amp: bool = False) -> float:
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    # AMP handling compatible across torch versions
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        def _autocast():
            return torch.amp.autocast('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        def _autocast():
            return torch.cuda.amp.autocast(enabled=use_amp)

    running = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in tqdm(loader, desc="train[temporal]"):
        x = batch["x"].to(device)  # (B,T,C,H,W)
        y = batch["y"].to(device)
        clin = batch.get("clin", None)
        if clin is not None:
            clin = clin.to(device)
        with _autocast():
            try:
                if clin is not None:
                    _, logits = model(x, clin)
                else:
                    _, logits = model(x)
            except TypeError:
                # Fallback if model signature differs
                _, logits = model(x)
            loss = ce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running += float(loss.detach().cpu().item()) * x.size(0)
        n += x.size(0)
    return running / max(1, n)


def evaluate_temporal(model, loader, device, video_reduce: str = 'mean', reduce_params: Optional[Dict] = None) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Evaluate clips and aggregate to videos.
    Returns: (video_metrics: dict, per_video_probs: dict, video_df: DataFrame)
    """
    model.eval()
    probs = []
    ys = []
    patients = []
    videos = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval[temporal]"):
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()  # video-level label
            clin = batch.get("clin", None)
            if clin is not None:
                clin = clin.to(device)
            try:
                if clin is not None:
                    _, logits = model(x, clin)
                else:
                    _, logits = model(x)
            except TypeError:
                _, logits = model(x)
            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            probs.append(p); ys.append(y)
            patients.append(batch["patient"].cpu().numpy())
            videos.append(batch["video"].cpu().numpy())

    y_clip = np.concatenate(ys)
    p_clip = np.concatenate(probs)
    pat = np.concatenate(patients)
    vid = np.concatenate(videos)

    df = pd.DataFrame({"patient": pat, "video": vid, "y_true": y_clip, "y_prob": p_clip})
    df = df.reset_index(drop=True)
    df['clip_order'] = df.groupby(["patient", "video"]).cumcount()

    # MIL-style reducer params
    rp = reduce_params or {}
    topk_frac = float(rp.get('topk_frac', 0.05))
    topk_cap = int(rp.get('topk_cap', 30))
    smooth_win = int(rp.get('smooth_win', 5))
    smooth_kind = str(rp.get('smooth_kind', 'median')).lower()
    lse_alpha = float(rp.get('lse_alpha', 8.0))

    def _auto_k(n: int) -> int:
        base = int(np.ceil(topk_frac * max(1, n)))
        return int(max(3, min(topk_cap, base)))

    def _topk_mean(a: np.ndarray, k: int) -> float:
        if a.size == 0:
            return float('nan')
        k = max(1, min(int(k), a.size))
        return float(np.mean(np.sort(a)[-k:]))

    def _smooth(a: np.ndarray) -> np.ndarray:
        if a.size == 0 or smooth_win <= 1:
            return a
        s = pd.Series(a)
        if smooth_kind == 'mean':
            return s.rolling(window=smooth_win, min_periods=1, center=True).mean().values
        return s.rolling(window=smooth_win, min_periods=1, center=True).median().values

    def _attn_pool(a: np.ndarray) -> float:
        if a.size == 0:
            return float('nan')
        w = np.exp(lse_alpha * a)
        w = w / (w.sum() + 1e-12)
        return float((w * a).sum())

    reducers = {
        "max": lambda g: float(np.max(g["y_prob"].values)),
        "mean": lambda g: float(np.mean(g["y_prob"].values)),
        "median": lambda g: float(np.median(g["y_prob"].values)),
        "topk2": lambda g: float(np.mean(np.sort(g["y_prob"].values)[-min(2, len(g)):])),
        "topk3": lambda g: float(np.mean(np.sort(g["y_prob"].values)[-min(3, len(g)):])),
        # Additional reducers to mirror frame pipeline
        "logit_mean": lambda g: float(1.0/(1.0+np.exp(-np.mean(np.log(np.clip(g["y_prob"].values,1e-6,1-1e-6)/np.clip(1-g["y_prob"].values,1e-6,1-1e-6)))))),
        "ew_mean": lambda g: float(np.average(g["y_prob"].values, weights=np.abs(g["y_prob"].values-0.5)*2 + 1e-6)),
        "tsmooth_mean": lambda g: float(np.mean(_smooth(g.sort_values('clip_order')["y_prob"].values))),
        "tsmooth_max": lambda g: float(np.max(_smooth(g.sort_values('clip_order')["y_prob"].values))),
        # MIL-inspired reducers
        "topk_auto": lambda g: _topk_mean(g["y_prob"].values, _auto_k(len(g))),
        "tsmooth_topk_auto": lambda g: _topk_mean(_smooth(g.sort_values('clip_order')["y_prob"].values), _auto_k(len(g))),
        "lse": lambda g: _attn_pool(g["y_prob"].values),
    }

    sel = video_reduce if video_reduce in reducers else 'mean'
    keys = []
    vals = []
    trues = []
    for (pid, vid), g in df.groupby(["patient", "video"], sort=False):
        keys.append((int(pid), int(vid)))
        trues.append(int(np.max(g["y_true"].values)))
        vals.append(reducers[sel](g))
    vdf = pd.DataFrame(keys, columns=["patient", "video"])  # one row per video
    vdf["y_true"] = trues
    vdf["y_prob"] = vals
    metrics = metrics_from_probs(vdf["y_true"].values, vdf["y_prob"].values, thr=0.5)
    return metrics, {"y_true": y_clip, "y_prob": p_clip}, vdf
