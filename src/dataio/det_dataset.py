import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataio.dataset import (
    load_metadata_any,
    build_text_embed_map,
    compute_meta_stats,
    build_meta_vectors_from_stats,
)
from src.detection_models.build_yolo_detection_dataset import (  # type: ignore
    build_detection_index,
    convert_box_xywh_to_yolo,
)


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return np.stack([img, img, img], axis=-1)


def _load_cadica_meta_with_vectors(
    metadata_path: str,
    selected_features: Optional[Dict] = None,
    text_features: Optional[List[str]] = None,
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    text_cache_dir: str = "outputs/text_embeds",
    fit_patients: Optional[List[int]] = None,
    stats: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict]:
    """
    Helper that mirrors CadicaDataset logic to build per-patient metadata vectors
    and stats, re-using the same code paths to ensure consistency with classification.
    """
    meta_df = load_metadata_any(metadata_path)

    text_map = None
    if text_features:
        text_map = build_text_embed_map(
            meta_df,
            patient_col="patient",
            text_cols=text_features,
            model_name=text_model,
            cache_dir=text_cache_dir,
        )

    if stats is None:
        stats = compute_meta_stats(
            meta_df,
            selected_features=selected_features,
            fit_patients=fit_patients,
        )

    meta_vecs = build_meta_vectors_from_stats(
        meta_df,
        selected_features=selected_features,
        text_embed_map=text_map,
        stats=stats,
    )

    return meta_df, meta_vecs, stats


class CadicaDetDataset(Dataset):
    """
    Detection dataset for CADICA that returns:
      - image tensor
      - YOLO-format targets for one image (tensor of shape [N, 6]: [image_idx, class, x, y, w, h])
      - metadata vector clin (per-patient)

    This is designed for custom detection training loops (not the built-in Ultralytics trainer),
    and mirrors CadicaDataset's metadata processing so that tab_dim and feature semantics match.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata_path: str,
        tfm,
        selected_features: Optional[Dict] = None,
        text_features: Optional[List[str]] = None,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_cache_dir: str = "outputs/text_embeds",
        fit_patients: Optional[List[int]] = None,
        stats: Optional[Dict] = None,
        class_map: Optional[Dict[str, int]] = None,
        single_class: bool = True,
    ):
        """
        Args:
            df: frame-level index with columns:
                ['patient','video','frame','img_path','gt_path'].
            metadata_path: path to CADICA metadata.xlsx.
            tfm: Albumentations-like transform returning {'image': np.ndarray}.
            selected_features: {'numeric':[...], 'categorical':[...]} from configs.
            text_features: list of text column keys (e.g. ['another_comorb','indication']).
            fit_patients: list of patient IDs used to fit metadata stats (for CV).
            stats: optional precomputed stats; if None, computed from meta_df + fit_patients.
            class_map: mapping from CADICA lesion tags (5th column in GT) to integer IDs.
            single_class: if True, ignore CADICA tags and treat all lesions as class 0.
        """
        if df.empty:
            raise RuntimeError("CadicaDetDataset received empty DataFrame.")

        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.single_class = bool(single_class)
        self.class_map = class_map or {"lesion": 0}

        # metadata vectors (per-patient)
        self.meta_df, self.meta_vecs, self.stats = _load_cadica_meta_with_vectors(
            metadata_path=metadata_path,
            selected_features=selected_features,
            text_features=text_features,
            text_model=text_model,
            text_cache_dir=text_cache_dir,
            fit_patients=fit_patients,
            stats=stats,
        )

        # tabular dimension (same as in CadicaDataset)
        # reconstruct feature_names from stats + known category expansions is non-trivial,
        # so infer tab_dim from any one patient vector.
        any_pid = next(iter(self.meta_vecs.keys()))
        self.tab_dim = int(self.meta_vecs[any_pid].shape[0])

        # cache image sizes to avoid repeated disk reads when computing normalized boxes
        self._size_cache: Dict[str, Tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_img_size(self, path: str) -> Tuple[int, int]:
        if path in self._size_cache:
            return self._size_cache[path]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        self._size_cache[path] = (w, h)
        return w, h

    def _load_targets(self, row, img_w: int, img_h: int) -> torch.Tensor:
        """
        Parse CADICA GT file for one frame into YOLO [N,6] tensor:
          [image_idx (0), class_id, x_center, y_center, w, h]
        All coordinates are normalized to [0,1].
        """
        gt_path = row["gt_path"]
        boxes = []
        if not os.path.isfile(gt_path):
            return torch.zeros((0, 6), dtype=torch.float32)
        with open(gt_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x, y, bw, bh = map(float, parts[:4])
                if self.single_class:
                    cls_id = 0
                else:
                    tag = parts[4] if len(parts) >= 5 else next(iter(self.class_map.keys()))
                    cls_id = int(self.class_map.get(tag, 0))
                xc, yc, wn, hn = convert_box_xywh_to_yolo(x, y, bw, bh, img_w, img_h)
                boxes.append([0.0, float(cls_id), xc, yc, wn, hn])
        if not boxes:
            return torch.zeros((0, 6), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img_np = _read_rgb(r.img_path)
        data = self.tfm(image=img_np)
        x = data["image"].float()

        img_w, img_h = self._get_img_size(r.img_path)
        targets = self._load_targets(r, img_w, img_h)

        clin_vec = self.meta_vecs.get(int(r.patient))
        if clin_vec is None:
            # fallback to zeros if metadata missing
            clin = torch.zeros(self.tab_dim, dtype=torch.float32)
        else:
            clin = torch.tensor(clin_vec, dtype=torch.float32)

        return {
            "x": x,
            "targets": targets,
            "clin": clin,
            "patient": int(r.patient),
            "video": int(r.video),
            "frame": int(r.frame),
            "img_path": r.img_path,
        }


