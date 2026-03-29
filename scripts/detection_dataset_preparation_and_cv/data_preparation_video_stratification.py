#!/usr/bin/env python3
"""
Data preparation for YOLO detection with video-level stratification.
Creates nested 5-fold CV where all frames from the same video stay together.
"""

import os
import shutil
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


# =========================
# CONFIG
# =========================
BASE_DIR = Path("/cluster/home/ammaa/Downloads/meta-cadica/data/cadica_yolo_det")
OUTPUT_DIR = Path("/cluster/home/ammaa/Downloads/meta-cadica/data/det_video_stratified_folds")

N_OUTER = 5
INNER_VAL_RATIO = 0.15
TEST_RATIO = 0.10  # Reserve 10% of videos for test set
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================
# SOURCE PATHS (TRAIN + VAL ONLY)
# =========================
IMAGE_DIRS = [
    BASE_DIR / "train/images",
    BASE_DIR / "val/images",
]

LABEL_DIRS = [
    BASE_DIR / "train/labels",
    BASE_DIR / "val/labels",
]


# =========================
# GROUP IMAGES BY VIDEO
# =========================
def load_video_data() -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Group all images by video ID.
    Returns dict: video_id -> [(img_path, lbl_path), ...]
    """
    video_dict = defaultdict(list)
    video_regex = re.compile(r"p(\d+)_v(\d+)_")

    for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
        if not img_dir.exists():
            continue
            
        for img_name in os.listdir(img_dir):
            match = video_regex.match(img_name)
            if not match:
                continue

            patient_id = match.group(1)
            video_id = match.group(2)
            # Create unique video identifier: "p{patient}_v{video}"
            vid = f"p{patient_id}_v{video_id}"
            
            img_path = img_dir / img_name
            lbl_path = lbl_dir / img_name.replace(".png", ".txt").replace(".jpg", ".txt")

            if lbl_path.exists():
                video_dict[vid].append((img_path, lbl_path))

    return video_dict


# =========================
# STRATIFICATION LABEL
# =========================
def video_has_annotation(files: List[Tuple[Path, Path]]) -> int:
    """
    Check if any frame in this video has annotations.
    1 = video has lesions, 0 = video has no lesions
    """
    for _, lbl in files:
        if lbl.stat().st_size > 0:
            return 1
    return 0


# =========================
# FILE COPYING
# =========================
def copy_videos(
    video_ids: List[str],
    video_dict: Dict[str, List[Tuple[Path, Path]]],
    dst_dir: Path,
) -> int:
    """Copy all frames from specified videos to destination."""
    count = 0
    for vid in video_ids:
        for img_p, lbl_p in video_dict[vid]:
            shutil.copy(img_p, dst_dir / "images")
            shutil.copy(lbl_p, dst_dir / "labels")
            count += 1
    return count


# =========================
# MAIN PIPELINE
# =========================
def main():
    video_dict = load_video_data()
    videos = list(video_dict.keys())

    print(f"Total videos: {len(videos)}")
    total_frames = sum(len(frames) for frames in video_dict.values())
    print(f"Total frames: {total_frames}")

    # Create video-level stratification labels
    video_labels = [
        video_has_annotation(video_dict[v]) for v in videos
    ]
    
    pos_videos = sum(video_labels)
    neg_videos = len(video_labels) - pos_videos
    print(f"Positive videos (with lesions): {pos_videos}")
    print(f"Negative videos (no lesions): {neg_videos}")

    # =========================
    # FIRST: RESERVE TEST SET
    # =========================
    print(f"\n{'='*60}")
    print(f"Reserving Test Set ({TEST_RATIO*100:.0f}% of videos, video-stratified)")
    print(f"{'='*60}")
    
    test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=SEED,
    )
    
    trainval_idx, test_idx = next(test_splitter.split(videos, video_labels))
    
    trainval_videos = [videos[i] for i in trainval_idx]
    test_videos = [videos[i] for i in test_idx]
    trainval_labels = [video_labels[i] for i in trainval_idx]
    
    print(f"TrainVal Videos: {len(trainval_videos)}")
    print(f"Test Videos: {len(test_videos)}")
    
    # Create test directory
    test_dir = OUTPUT_DIR / "test"
    (test_dir / "images").mkdir(parents=True, exist_ok=True)
    (test_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    test_imgs = copy_videos(test_videos, video_dict, test_dir)
    print(f"Test frames: {test_imgs}")

    # =========================
    # NESTED 5-FOLD CV ON TRAINVAL
    # =========================
    print(f"\n{'='*60}")
    print(f"Creating {N_OUTER}-Fold Cross-Validation (Video-Stratified)")
    print(f"{'='*60}")
    
    outer_kf = StratifiedKFold(
        n_splits=N_OUTER, shuffle=True, random_state=SEED
    )

    for outer_fold, (train_idx, val_idx) in enumerate(
        outer_kf.split(trainval_videos, trainval_labels), start=1
    ):
        print(f"\n{'='*50}")
        print(f"Fold {outer_fold}")
        print(f"{'='*50}")

        outer_train_videos = [trainval_videos[i] for i in train_idx]
        outer_val_videos   = [trainval_videos[i] for i in val_idx]

        print(f"Outer-Train Videos: {len(outer_train_videos)}")
        print(f"Outer-Val Videos:   {len(outer_val_videos)}")

        # =========================
        # INNER 85 / 15 SPLIT
        # =========================
        inner_labels = [
            trainval_labels[trainval_videos.index(v)]
            for v in outer_train_videos
        ]

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_RATIO,
            random_state=SEED,
        )

        inner_train_idx, inner_val_idx = next(
            sss.split(outer_train_videos, inner_labels)
        )

        inner_train_videos = [
            outer_train_videos[i] for i in inner_train_idx
        ]
        inner_val_videos = [
            outer_train_videos[i] for i in inner_val_idx
        ]

        # =========================
        # CREATE FOLD DIRECTORIES
        # =========================
        fold_dir = OUTPUT_DIR / f"fold_{outer_fold}"
        for split in ["train", "inner_val", "outer_val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # =========================
        # COPY FILES
        # =========================
        train_imgs = copy_videos(
            inner_train_videos, video_dict, fold_dir / "train"
        )
        inner_val_imgs = copy_videos(
            inner_val_videos, video_dict, fold_dir / "inner_val"
        )
        outer_val_imgs = copy_videos(
            outer_val_videos, video_dict, fold_dir / "outer_val"
        )

        print(
            f"Images → "
            f"Train: {train_imgs} | "
            f"Inner-Val: {inner_val_imgs} | "
            f"Outer-Val: {outer_val_imgs}"
        )

    print("\n" + "="*60)
    print("✅ Video-stratified nested 5-Fold CV + Test set completed!")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Test Set: {OUTPUT_DIR / 'test'}")
    print("="*60)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
