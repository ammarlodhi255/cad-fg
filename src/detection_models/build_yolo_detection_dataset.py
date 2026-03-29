#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from typing import Dict, List, Tuple

import cv2
import yaml
import pandas as pd
from tqdm import tqdm

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cadica import parser as cad_parser  # type: ignore
from src.cadica.splits import split_by_patient  # type: ignore


def build_detection_index(root: str) -> pd.DataFrame:
    """
    Build a frame-level index for CADICA detection using only frames that have
    corresponding ground-truth bounding box files.

    Returns a DataFrame with at least:
      - patient (int)
      - video (int)
      - frame (int)
      - img_path (str)
      - gt_path (str)
    """
    base = os.path.join(root, "selectedVideos")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Expected directory not found: {base}")

    patients = sorted(glob.glob(os.path.join(base, "p*")))
    if not patients:
        raise RuntimeError(f"No patient folders like p*/ found under {base}.")

    rows: List[Dict] = []

    for patient_dir in patients:
        mp = None
        basename = os.path.basename(patient_dir)
        # patient_dir names are 'pX'
        if basename.startswith("p") and basename[1:].isdigit():
            mp = int(basename[1:])
        if mp is None:
            continue
        p_id = mp

        videos = sorted(glob.glob(os.path.join(patient_dir, "v*")))
        for video_dir in videos:
            v_base = os.path.basename(video_dir)
            if not (v_base.startswith("v") and v_base[1:].isdigit()):
                continue
            v_id = int(v_base[1:])

            inp = os.path.join(video_dir, "input")
            if not os.path.isdir(inp):
                continue

            fmap = cad_parser._index_input_frames(inp)  # type: ignore[attr-defined]

            # selected frame ids for this (patient, video)
            kf = cad_parser._read_selected_frames(patient_dir, video_dir, p_id, v_id)  # type: ignore[attr-defined]
            if not kf:
                continue

            gt_dir = os.path.join(video_dir, "groundtruth")
            if not os.path.isdir(gt_dir):
                # no lesion annotations in this video
                continue

            for k in kf:
                # image filename resolution mirrors classification parser logic
                fname = fmap["num_map"].get(k)
                if fname is None:
                    fname = fmap["pad_map"].get(f"_{k:05d}")
                if fname is None:
                    exp_stem = f"p{p_id}_v{v_id}_{k:05d}".lower()
                    fname = fmap["ext_map"].get(exp_stem)
                if fname is None:
                    for delta in (-1, +1):
                        alt = k + delta
                        if alt >= 0:
                            fname = fmap["num_map"].get(alt)
                            if fname:
                                break
                if fname is None:
                    suffix = f"_{k:05d}".lower()
                    for nm in fmap["names"]:
                        low = nm.lower()
                        if (
                            low.endswith(suffix + ".png")
                            or low.endswith(suffix + ".jpg")
                            or low.endswith(suffix + ".jpeg")
                        ):
                            fname = nm
                            break

                if fname is None:
                    continue

                img_path = os.path.join(inp, fname)
                if not os.path.isfile(img_path):
                    continue

                # ground-truth txt for this frame
                stem, _ = os.path.splitext(fname)
                gt_txt = os.path.join(gt_dir, f"{stem}.txt")
                if not os.path.isfile(gt_txt):
                    # Some videos may not have GT for all selected frames
                    continue

                rows.append(
                    {
                        "patient": p_id,
                        "video": v_id,
                        "frame": k,
                        "img_path": img_path,
                        "gt_path": gt_txt,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Detection index is empty. Check CADICA structure under {base}.")

    print(
        f"[build_detection_index] frames with GT: {len(df)} | "
        f"patients: {df['patient'].nunique()} | videos: {df['video'].nunique()}"
    )
    return df


def collect_classes(df: pd.DataFrame) -> Dict[str, int]:
    """
    Scan all ground-truth txt files to collect unique lesion category tokens
    (5th column: e.g., 'p0_20') and map them to integer class IDs.
    """
    classes = set()
    for path in df["gt_path"].unique():
        try:
            with open(path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        classes.add(parts[4])
        except FileNotFoundError:
            continue
    names = sorted(classes)
    return {name: idx for idx, name in enumerate(names)}


def convert_box_xywh_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    CADICA GT: [x, y, w, h] in absolute pixels, (x,y) top-left.
    YOLO format expects: [x_center, y_center, w, h] normalized to [0,1].
    """
    xc = (x + w / 2.0) / float(img_w)
    yc = (y + h / 2.0) / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)
    return xc, yc, wn, hn


def write_split(
    df_split: pd.DataFrame,
    split_name: str,
    out_root: str,
    class_map: Dict[str, int],
    single_class: bool = False,
) -> None:
    """
    Write images and YOLO-format label files for one split.
    """
    img_out = os.path.join(out_root, split_name, "images")
    lbl_out = os.path.join(out_root, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # Cache image sizes to avoid re-reading many times
    size_cache: Dict[str, Tuple[int, int]] = {}

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"{split_name}"):
        src_img = row["img_path"]
        gt_txt = row["gt_path"]
        # deterministic output name
        fname = os.path.basename(src_img)
        stem, ext = os.path.splitext(fname)
        out_img = os.path.join(img_out, fname)
        out_lbl = os.path.join(lbl_out, f"{stem}.txt")

        # copy image (cv2 imread/imwrite to avoid shutil on network FS issues)
        if src_img not in size_cache:
            img = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            size_cache[src_img] = (w, h)
            cv2.imwrite(out_img, img)
        else:
            w, h = size_cache[src_img]
            if not os.path.isfile(out_img):
                img = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                cv2.imwrite(out_img, img)

        w, h = size_cache[src_img]

        lines_out: List[str] = []
        try:
            with open(gt_txt) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    x, y, bw, bh = map(float, parts[:4])
                    if single_class:
                        cls_id = 0
                    else:
                        # if a 5th token exists, use it as category; else default to first class
                            # fallback to first class in map if present
                        tag = parts[4] if len(parts) >= 5 else next(iter(class_map.keys()))
                        cls_id = int(class_map.get(tag, 0))
                    xc, yc, wn, hn = convert_box_xywh_to_yolo(x, y, bw, bh, w, h)
                    lines_out.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        except FileNotFoundError:
            continue

        if lines_out:
            with open(out_lbl, "w") as fh:
                fh.write("\n".join(lines_out))


def write_yaml(
    out_root: str,
    class_map: Dict[str, int],
    single_class: bool,
    yaml_name: str = "cadica_yolo_det.yaml",
) -> str:
    """
    Write a YOLO dataset YAML describing train/val/test splits.
    """
    if single_class:
        names = ["lesion"]
    else:
        # ensure list is ordered by class id
        names = [None] * len(class_map)
        for k, v in class_map.items():
            if 0 <= v < len(names):
                names[v] = k
        names = [n if n is not None else f"class_{i}" for i, n in enumerate(names)]

    cfg = {
        "path": os.path.abspath(out_root),
        "train": os.path.join(os.path.abspath(out_root), "train", "images"),
        "val": os.path.join(os.path.abspath(out_root), "val", "images"),
        "test": os.path.join(os.path.abspath(out_root), "test", "images"),
        "nc": len(names),
        "names": names,
    }

    yaml_path = os.path.join(out_root, yaml_name)
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"[write_yaml] Wrote dataset YAML to: {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description="Build YOLO-style detection dataset from CADICA.")
    ap.add_argument(
        "--cadica-root",
        type=str,
        default="data/CADICA",
        help="Path to CADICA root folder (containing metadata.xlsx and selectedVideos/).",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="data/cadica_yolo_det",
        help="Output root directory for YOLO dataset (train/val/test with images+labels).",
    )
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction (by patient).")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (by patient).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for splitting.")
    ap.add_argument(
        "--single-class",
        action="store_true",
        help="Collapse all lesion categories into a single class 'lesion'.",
    )
    args = ap.parse_args()

    df = build_detection_index(args.cadica_root)

    if args.single_class:
        class_map = {"lesion": 0}
    else:
        class_map = collect_classes(df)
        if not class_map:
            # Fall back to single-class if something went wrong
            print("[warn] No classes discovered in GT; falling back to single-class setup.")
            class_map = {"lesion": 0}
            args.single_class = True

    # patient-stratified train/val/test
    df_tr, df_va, df_te = split_by_patient(df, train=args.train_frac, val=args.val_frac, seed=args.seed)
    print(
        f"[split_by_patient] train={len(df_tr)} frames, "
        f"val={len(df_va)} frames, test={len(df_te)} frames"
    )

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    write_split(df_tr, "train", out_root, class_map, single_class=args.single_class)
    write_split(df_va, "val", out_root, class_map, single_class=args.single_class)
    write_split(df_te, "test", out_root, class_map, single_class=args.single_class)

    write_yaml(out_root, class_map, single_class=args.single_class)


if __name__ == "__main__":
    main()


