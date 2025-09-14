import torch, numpy as np, pandas as pd
from tqdm import tqdm
from .metrics import metrics_from_probs

def train_one_epoch(model, optimizer, loader, device, mode="late", clin_mlp=None, late_head=None):
    model.train()
    if clin_mlp: clin_mlp.train()
    if late_head: late_head.train()
    ce = torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc=f"train[{mode}]"):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
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

        loss = ce(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

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
    patients, videos = [], []

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

    y_frame = np.concatenate(ys)
    p_frame = np.concatenate(probs)

    frame_metrics = metrics_from_probs(y_frame, p_frame, thr=0.5)

    if not also_video:
        return frame_metrics, y_frame, p_frame

    # ----- video-level aggregation -----
    df = pd.DataFrame({
        "patient": np.concatenate(patients),
        "video":   np.concatenate(videos),
        "y_true":  y_frame,
        "y_prob":  p_frame,
    })

    # ground-truth per video: positive if any frame positive (max)
    agg = {"y_true": ("y_true", "max")}
    # prediction reduce (choose mean or max)
    if video_reduce == "max":
        agg["y_prob"] = ("y_prob", "max")
    else:
        agg["y_prob"] = ("y_prob", "mean")

    video_df = df.groupby(["patient", "video"]).agg(**agg).reset_index()
    y_true_v = video_df["y_true"].values
    y_prob_v = video_df["y_prob"].values

    video_metrics = metrics_from_probs(y_true_v, y_prob_v, thr=0.5)
    return frame_metrics, y_frame, p_frame, video_metrics, video_df