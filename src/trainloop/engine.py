import torch, numpy as np, pandas as pd
from contextlib import nullcontext
from tqdm import tqdm
from .metrics import metrics_from_probs
import numpy as np
import pandas as pd

def train_one_epoch(model, optimizer, loader, device, mode="late", clin_mlp=None, late_head=None,
                    pim_lambdas: dict | None = None, pim_update_freq: int = 1, use_amp: bool = False):
    model.train()
    if clin_mlp: clin_mlp.train()
    if late_head: late_head.train()
    ce = torch.nn.CrossEntropyLoss()
    # AMP: prefer torch.amp (new API), fall back to torch.cuda.amp for older torch
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        def _autocast():
            return torch.amp.autocast('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        def _autocast():
            return torch.cuda.amp.autocast(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for batch in tqdm(loader, desc=f"train[{mode}]"):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        clin = batch["clin"].to(device)

        with _autocast():
            if mode == "late":
                feats, _ = model(x)
                logits = late_head(feats, clin_mlp(clin))
            elif mode == "early":
                _, logits = model(x, clin)
            elif mode == "imageonly":
                _, logits = model(x)
            else:
                raise ValueError("mode must be 'late', 'early' or 'imageonly'")

            loss = None
            pim_outs = getattr(model, 'last_pim_outs', None)
            if pim_lambdas and isinstance(pim_outs, dict):
                lb = float(pim_lambdas.get('b', 0.0) if pim_lambdas else 0.0)
                ls = float(pim_lambdas.get('s', 0.0) if pim_lambdas else 0.0)
                ln = float(pim_lambdas.get('n', 0.0) if pim_lambdas else 0.0)
                lc = float(pim_lambdas.get('c', 0.0) if pim_lambdas else 0.0)
                loss_total = 0.0
                # selector positives
                if ls != 0.0:
                    for name, val in pim_outs.items():
                        if isinstance(name, str) and name.startswith('select_'):
                            S = val.size(1)
                            logit = val.view(-1, val.size(-1)).contiguous()
                            loss_total = loss_total + ls * ce(logit, y.unsqueeze(1).repeat(1, S).flatten(0))
                # selector negatives
                if ln != 0.0:
                    for name, val in pim_outs.items():
                        if isinstance(name, str) and name.startswith('drop_'):
                            S = val.size(1)
                            logit = val.view(-1, val.size(-1)).contiguous()
                            n_preds = torch.tanh(logit)
                            labels_0 = torch.zeros_like(n_preds) - 1.0
                            loss_total = loss_total + ln * torch.nn.functional.mse_loss(n_preds, labels_0)
                # per-layer background CE
                if lb != 0.0:
                    for name, val in pim_outs.items():
                        if isinstance(name, str) and name.startswith('layer'):
                            loss_total = loss_total + lb * ce(val.mean(1), y)
                # combiner CE (final)
                if lc != 0.0 and 'comb_outs' in pim_outs:
                    loss_total = loss_total + lc * ce(pim_outs['comb_outs'], y)
                # Fallback if no terms accumulated
                if mode == "late":
                    loss = loss_total + ce(logits, y)
                else:
                    loss = loss_total if loss_total != 0.0 else ce(logits, y)
            else:
                loss = ce(logits, y)

            loss = loss / max(1, int(pim_update_freq))

        scaler.scale(loss).backward()
        step += 1
        if step % max(1, int(pim_update_freq)) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

def evaluate(model, loader, device, mode="late", clin_mlp=None, late_head=None,
             also_video: bool=False, video_reduce: str="mean"):
    """
    Returns:
      if also_video=False:
        (frame_metrics: dict, y_frame: np.array, p_frame: np.array)
      if also_video=True:
        (frame_metrics: dict, y_frame: np.array, p_frame: np.array,
         video_metrics: dict, video_df: pd.DataFrame)
    """
    model.eval()
    if clin_mlp: clin_mlp.eval()
    if late_head: late_head.eval()

    probs, ys = [], []
    patients, videos, frames = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval[{mode}]"):
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            clin = batch["clin"].to(device)

            if mode == "late":
                feats, _ = model(x)
                logits = late_head(feats, clin_mlp(clin))
            elif mode == "early":
                _, logits = model(x, clin)
            elif mode == "imageonly":
                _, logits = model(x)
            else:
                raise ValueError("mode must be 'late', 'early' or 'imageonly'")

            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()

            probs.append(p); ys.append(y)
            patients.append(np.asarray(batch["patient"]))
            videos.append(np.asarray(batch["video"]))
            # Optional frame index for temporal strategies
            if "frame" in batch:
                frames.append(np.asarray(batch["frame"]))
            else:
                frames.append(np.arange(len(y)))

    y_frame = np.concatenate(ys)
    p_frame = np.concatenate(probs)

    frame_metrics = metrics_from_probs(y_frame, p_frame, thr=0.5)

    if not also_video:
        return frame_metrics, y_frame, p_frame

    # ----- video-level aggregation -----
    df = pd.DataFrame({
        "patient": np.concatenate(patients),
        "video":   np.concatenate(videos),
        "frame":   np.concatenate(frames),
        "y_true":  y_frame,
        "y_prob":  p_frame,
    })

    # ground-truth per video: positive if any frame positive (max)
    def _agg_true(g):
        return float(np.max(g["y_true"].values))

    reducers = {
        "max": lambda g: float(np.max(g["y_prob"].values)),
        "mean": lambda g: float(np.mean(g["y_prob"].values)),
        "median": lambda g: float(np.median(g["y_prob"].values)),
        "topk2": lambda g: float(np.mean(np.sort(g["y_prob"].values)[-min(2, len(g)):])),
        "topk3": lambda g: float(np.mean(np.sort(g["y_prob"].values)[-min(3, len(g)):])),
        "logit_mean": lambda g: float(1.0/(1.0+np.exp(-np.mean(np.log(np.clip(g["y_prob"].values,1e-6,1-1e-6)/np.clip(1-g["y_prob"].values,1e-6,1-1e-6)))))),
        "ew_mean": lambda g: float(np.average(g["y_prob"].values, weights=np.abs(g["y_prob"].values-0.5)*2 + 1e-6)),
        "tsmooth_mean": lambda g: float(np.mean(pd.Series(g.sort_values("frame")["y_prob"]).rolling(window=5, min_periods=1, center=True).mean().values)),
        "tsmooth_max": lambda g: float(np.max(pd.Series(g.sort_values("frame")["y_prob"]).rolling(window=5, min_periods=1, center=True).mean().values)),
    }

    # Compute all reducers
    dfs = {}
    metrics_map = {}
    base_cols = ["patient", "video", "y_true"]
    for name, fn in reducers.items():
        vals = []
        trues = []
        keys = []
        for (pid, vid), g in df.groupby(["patient", "video"], sort=False):
            keys.append((pid, vid))
            trues.append(_agg_true(g))
            try:
                vals.append(fn(g))
            except Exception:
                vals.append(float(np.mean(g["y_prob"].values)))
        vdf = pd.DataFrame(keys, columns=["patient","video"])
        vdf["y_true"] = trues
        vdf[f"y_prob_{name}"] = vals
        dfs[name] = vdf
        metrics_map[name] = metrics_from_probs(vdf["y_true"].values, vdf[f"y_prob_{name}"].values, thr=0.5)

    # Select requested reducer for compatibility
    sel = video_reduce if video_reduce in reducers else ("max" if video_reduce == "all" else "mean")
    video_df = dfs[sel].rename(columns={f"y_prob_{sel}": "y_prob"}).copy()
    # Merge all y_prob_* columns for analysis convenience
    for name, vdf in dfs.items():
        if name == sel:
            continue
        video_df = video_df.merge(vdf[["patient","video", f"y_prob_{name}"]], on=["patient","video"], how="left")

    video_metrics = metrics_map[sel]
    # Attach multi-metric map for downstream reporting (optional use)
    video_metrics["_multi"] = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics_map.items()}

    return frame_metrics, y_frame, p_frame, video_metrics, video_df
