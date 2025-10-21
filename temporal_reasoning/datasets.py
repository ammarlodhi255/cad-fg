import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

# Reuse project image transforms
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import (
    load_metadata_any,
    compute_meta_stats,
    build_meta_vectors_from_stats,
    build_text_embed_map,
)


def _read_img_rgb(path: str) -> np.ndarray:
    """Read as grayscale if needed and expand to 3 channels, or read as color if available."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:  # grayscale -> 3ch
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:  # RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class VideoClipsIndex:
    """
    Index frames by (patient, video), providing access to sorted frame rows.

    Expected CSV columns: img_path, label, patient, video, frame
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        required = {"img_path", "label", "patient", "video", "frame"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
        # Normalize dtypes
        df["patient"] = df["patient"].astype(int)
        df["video"] = df["video"].astype(int)
        df["frame"] = df["frame"].astype(int)
        # Sort within video by frame
        df = df.sort_values(["patient", "video", "frame"]).reset_index(drop=True)
        self.df = df

        # Build mapping
        self.keys: List[Tuple[int, int]] = []
        self.index: Dict[Tuple[int, int], pd.DataFrame] = {}
        self.video_labels: Dict[Tuple[int, int], int] = {}
        for (pid, vid), g in df.groupby(["patient", "video"], sort=False):
            g = g.sort_values("frame").reset_index(drop=True)
            key = (int(pid), int(vid))
            self.keys.append(key)
            self.index[key] = g
            try:
                self.video_labels[key] = int(np.max(g["label"].values))
            except Exception:
                self.video_labels[key] = int(g["label"].astype(int).max())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> Tuple[int, int, pd.DataFrame]:
        k = self.keys[i]
        return k[0], k[1], self.index[k]


def _label_reduce_max(frame_labels: np.ndarray) -> int:
    return int(np.max(frame_labels))


def _sample_clip_indices(n: int, T: int, stride: int, mode: str) -> List[np.ndarray]:
    """
    Produce one or more index arrays of length T into a sequence of length n.
    mode:
      - 'random': random contiguous window (with stride across elements); if n<T, pad by repeating last
      - 'center': centered contiguous window
      - 'uniform': T approximately uniform-spaced indices across [0..n-1]
      - 'chunked': return a list of non-overlapping contiguous windows of size T (last may be shorter -> pad)
    """
    if n <= 0:
        return [np.zeros((T,), dtype=int)]

    if mode == 'uniform':
        if n >= T:
            idxs = np.linspace(0, n - 1, T).round().astype(int)
        else:
            # repeat last to pad
            base = np.arange(n, dtype=int)
            pad = np.full((T - n,), n - 1, dtype=int)
            idxs = np.concatenate([base, pad], axis=0)
        return [idxs]

    if mode == 'center':
        if n >= T:
            mid = n // 2
            half = T // 2
            start = max(0, min(n - T, mid - half))
            idxs = np.arange(start, start + T)
        else:
            base = np.arange(n, dtype=int)
            pad = np.full((T - n,), n - 1, dtype=int)
            idxs = np.concatenate([base, pad], axis=0)
        return [idxs]

    if mode == 'chunked':
        clips = []
        if n <= T:
            base = np.arange(n, dtype=int)
            pad = np.full((T - n,), n - 1, dtype=int)
            clips.append(np.concatenate([base, pad], axis=0))
            return clips
        for start in range(0, n, T):
            end = min(start + T, n)
            chunk = np.arange(start, end, dtype=int)
            if len(chunk) < T:
                pad = np.full((T - len(chunk),), end - 1, dtype=int)
                chunk = np.concatenate([chunk, pad], axis=0)
            clips.append(chunk)
        return clips

    # default: random contiguous window with stride=1 across frames
    if n >= T:
        start_max = n - T
        start = np.random.randint(0, start_max + 1)
        idxs = np.arange(start, start + T, dtype=int)
    else:
        base = np.arange(n, dtype=int)
        pad = np.full((T - n,), n - 1, dtype=int)
        idxs = np.concatenate([base, pad], axis=0)
    return [idxs]


class TemporalClipDataset(Dataset):
    """
    Produce clips of T frames for temporal training. Labels are reduced to a video label by max(frame_labels).

    Returns a dict:
      - x: FloatTensor (T, C, H, W)
      - y: LongTensor () video-level label
      - patient: int
      - video: int
      - frames: LongTensor (T,) original frame indices
      - img_paths: list[str] (debugging)
    """
    def __init__(
        self,
        csv_path: str,
        img_size: int = 224,
        num_frames: int = 8,
        sample_mode: str = 'random',  # 'random'|'center'|'uniform'|'chunked'
        label_reduce: str = 'max',
        is_train: bool = True,
        # metadata options
        metadata_path: Optional[str] = None,
        selected_features: Optional[Dict[str, List[str]]] = None,
        fit_patients: Optional[List[int]] = None,
        stats: Optional[Dict] = None,
        # text features (match configs/default.yaml style)
        text_features: Optional[List[str]] = None,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_cache_dir: str = "outputs/text_embeds",
    ):
        super().__init__()
        self.index = VideoClipsIndex(csv_path)
        self.T = int(num_frames)
        self.sample_mode = sample_mode
        self.label_reduce = label_reduce
        self.is_train = bool(is_train)
        self.tfm = get_train_tf(img_size) if is_train else get_val_tf(img_size)

        # Optional metadata loading
        self.meta_map = None
        self.meta_stats = None
        self.feature_names = None
        if metadata_path is not None:
            meta_df = load_metadata_any(metadata_path)
            # compute stats (fit on provided patient subset to avoid leakage)
            if stats is None:
                self.meta_stats = compute_meta_stats(meta_df, selected_features=selected_features, fit_patients=fit_patients)
            else:
                self.meta_stats = stats
            text_map = None
            if text_features:
                text_map = build_text_embed_map(meta_df, patient_col="patient", text_cols=text_features, model_name=text_model, cache_dir=text_cache_dir)
            self.meta_map = build_meta_vectors_from_stats(meta_df, selected_features=selected_features, text_embed_map=text_map, stats=self.meta_stats)

        # Pre-build clip pointers
        self.ptrs: List[Tuple[int, np.ndarray]] = []  # (key_idx, idxs)
        if self.is_train and self.sample_mode in ('random', 'center', 'uniform'):
            # For training, sample per __getitem__ dynamically; here keep 1 ptr per video as placeholder
            self.ptrs = [(i, None) for i in range(len(self.index))]
        else:
            # For eval, expand all chunked clips deterministically
            for i in range(len(self.index)):
                _, _, g = self.index[i]
                clips = _sample_clip_indices(len(g), self.T, stride=1, mode=(self.sample_mode or 'chunked'))
                for idxs in clips:
                    self.ptrs.append((i, idxs))

    def __len__(self) -> int:
        return len(self.ptrs)

    def __getitem__(self, i: int) -> Dict:
        key_idx, idxs = self.ptrs[i]
        pid, vid, g = self.index[key_idx]
        n = len(g)

        # Dynamic sampling for training
        if idxs is None:
            idxs = _sample_clip_indices(n, self.T, stride=1, mode=(self.sample_mode or 'random'))[0]

        rows = g.iloc[idxs.tolist()]
        imgs = []
        paths = []
        for p in rows.img_path.tolist():
            img = _read_img_rgb(p)
            data = self.tfm(image=img)
            x = data["image"].float()  # (C,H,W)
            imgs.append(x)
            paths.append(p)
        x_clip = torch.stack(imgs, dim=0)  # (T,C,H,W)

        labels = rows.label.values.astype(int)
        if self.label_reduce == 'max':
            y = _label_reduce_max(labels)
        else:
            y = _label_reduce_max(labels)

        out = {
            "x": x_clip,  # (T,C,H,W)
            "y": torch.tensor(int(y), dtype=torch.long),
            "patient": int(pid),
            "video": int(vid),
            "frames": torch.tensor(rows.frame.values.astype(int), dtype=torch.long),
            "img_paths": paths,
        }
        if self.meta_map is not None and int(pid) in self.meta_map:
            clin = torch.tensor(self.meta_map[int(pid)], dtype=torch.float32)
            out["clin"] = clin
        return out


def temporal_collate(batch: List[Dict]) -> Dict:
    # Stack clips into (B,T,C,H,W)
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    patient = torch.tensor([b["patient"] for b in batch], dtype=torch.long)
    video = torch.tensor([b["video"] for b in batch], dtype=torch.long)
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    out = {"x": x, "y": y, "patient": patient, "video": video, "frames": frames, "img_paths": [b["img_paths"] for b in batch]}
    # Optional clinical vectors if provided by dataset
    if all("clin" in b for b in batch):
        try:
            clin = torch.stack([b["clin"] for b in batch], dim=0)
        except Exception:
            # fallback to constructing tensor if items are numpy
            clin = torch.tensor(np.stack([np.asarray(b["clin"]) for b in batch], axis=0), dtype=torch.float32)
        out["clin"] = clin
    return out
