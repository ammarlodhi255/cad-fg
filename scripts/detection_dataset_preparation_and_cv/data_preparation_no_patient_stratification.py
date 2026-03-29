#!/usr/bin/env python3
"""
Data preparation for YOLO detection without patient stratification.
Creates a global test set and nested 5-fold CV from all available data.
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
OUTPUT_DIR = Path("/cluster/home/ammaa/Downloads/meta-cadica/data/det_no_patient_stratification")

TEST_RATIO = 0.10  # 10% held-out test set
N_OUTER = 5
INNER_VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================
# SOURCE PATHS (ALL SPLITS)
# =========================
IMAGE_DIRS = [
    BASE_DIR / "train/images",
    BASE_DIR / "val/images",
    BASE_DIR / "test/images",
]

LABEL_DIRS = [
    BASE_DIR / "train/labels",
    BASE_DIR / "val/labels",
    BASE_DIR / "test/labels",
]


# =========================
# COLLECT ALL IMAGES
# =========================
def load_all_data() -> List[Tuple[Path, Path]]:
    """Load all images and labels from train/val/test."""
    all_data = []
    
    for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
        if not img_dir.exists():
            continue
            
        for img_name in os.listdir(img_dir):
            img_path = img_dir / img_name
            lbl_path = lbl_dir / img_name.replace(".png", ".txt").replace(".jpg", ".txt")

            if lbl_path.exists():
                all_data.append((img_path, lbl_path))

    return all_data


# =========================
# STRATIFICATION LABEL
# =========================
def has_annotation(lbl_path: Path) -> int:
    """1 = has lesion annotations, 0 = no annotations"""
    if lbl_path.stat().st_size > 0:
        return 1
    return 0


# =========================
# FILE COPYING
# =========================
def copy_files(
    file_list: List[Tuple[Path, Path]],
    dst_dir: Path,
) -> int:
    """Copy images and labels to destination."""
    count = 0
    for img_p, lbl_p in file_list:
        shutil.copy(img_p, dst_dir / "images")
        shutil.copy(lbl_p, dst_dir / "labels")
        count += 1
    return count


# =========================
# MAIN PIPELINE
# =========================
def main():
    all_data = load_all_data()
    print(f"Total images collected: {len(all_data)}")

    # Create stratification labels
    labels = [has_annotation(lbl) for _, lbl in all_data]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Positive (with lesions): {pos_count}")
    print(f"Negative (no lesions): {neg_count}")

    # =========================
    # 1. Create Global Test Set
    # =========================
    print(f"\n{'='*60}")
    print("Creating global test set ({:.0%})".format(TEST_RATIO))
    print(f"{'='*60}")
    
    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=SEED,
    )
    
    trainval_idx, test_idx = next(sss_test.split(all_data, labels))
    trainval_data = [all_data[i] for i in trainval_idx]
    test_data = [all_data[i] for i in test_idx]
    trainval_labels = [labels[i] for i in trainval_idx]
    
    print(f"Test set: {len(test_data)} images")
    print(f"Train+Val pool: {len(trainval_data)} images")
    
    # Save test set
    test_dir = OUTPUT_DIR / "test"
    (test_dir / "images").mkdir(parents=True, exist_ok=True)
    (test_dir / "labels").mkdir(parents=True, exist_ok=True)
    test_count = copy_files(test_data, test_dir)
    print(f"✅ Test set saved: {test_count} images")

    # =========================
    # 2. Create 5-Fold CV from Remaining Data
    # =========================
    print(f"\n{'='*60}")
    print(f"Creating {N_OUTER}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    outer_kf = StratifiedKFold(
        n_splits=N_OUTER, shuffle=True, random_state=SEED
    )

    for outer_fold, (train_idx, val_idx) in enumerate(
        outer_kf.split(trainval_data, trainval_labels), start=1
    ):
        print(f"\n{'='*50}")
        print(f"Fold {outer_fold}")
        print(f"{'='*50}")

        outer_train_data = [trainval_data[i] for i in train_idx]
        outer_val_data = [trainval_data[i] for i in val_idx]
        outer_train_labels = [trainval_labels[i] for i in train_idx]

        print(f"Outer-Train: {len(outer_train_data)} images")
        print(f"Outer-Val:   {len(outer_val_data)} images")

        # =========================
        # INNER 85 / 15 SPLIT
        # =========================
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_RATIO,
            random_state=SEED,
        )

        inner_train_idx, inner_val_idx = next(
            sss.split(outer_train_data, outer_train_labels)
        )

        inner_train_data = [outer_train_data[i] for i in inner_train_idx]
        inner_val_data = [outer_train_data[i] for i in inner_val_idx]

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
        train_imgs = copy_files(inner_train_data, fold_dir / "train")
        inner_val_imgs = copy_files(inner_val_data, fold_dir / "inner_val")
        outer_val_imgs = copy_files(outer_val_data, fold_dir / "outer_val")

        print(
            f"Images → "
            f"Train: {train_imgs} | "
            f"Inner-Val: {inner_val_imgs} | "
            f"Outer-Val: {outer_val_imgs}"
        )

    print("\n" + "="*60)
    print("✅ Non-patient-stratified nested 5-Fold CV completed!")
    print(f"   Output: {OUTPUT_DIR}")
    print("="*60)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
