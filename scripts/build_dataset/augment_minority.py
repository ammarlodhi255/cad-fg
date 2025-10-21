#!/usr/bin/env python3
"""
Create a self-contained augmented copy of the CADICA dataset focusing on upsampling the
minority class (label 0, nonlesion) by image-space augmentations of existing negative frames.

Outputs:
- data/augmented-cadica/ ... a mirror of selectedVideos with symlinked originals and new aug images
- outputs/index/all_aug_keyframes.csv ... full index including originals (pointing to augmented root) + augmented negatives
- data/augmented-cadica/metadata.xlsx ... copied from source to preserve patient metadata

Notes:
- Original images are symlinked (fallback to copy if symlink not permitted) to avoid duplication.
- Augmented negatives are written as new files alongside originals with unique numeric frame ids.
- Augmentation count is computed to target near 1:1 frame balance (neg ~ pos) unless overridden.

Example:
  python scripts/augment_minority.py \
    --src_root data/CADICA \
    --src_index outputs/index/all_keyframes.csv \
    --dst_root data/augmented-cadica \
    --dst_index outputs/index/all_aug_keyframes.csv
"""
import os
import sys
import csv
import math
import json
import argparse
import shutil
import random
from collections import defaultdict

import cv2


def read_csv_rows(path):
    rows = []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # normalize types
            row['patient'] = int(row['patient'])
            row['video'] = int(row['video'])
            row['frame'] = int(row['frame'])
            row['label'] = int(row['label'])
            rows.append(row)
    return rows


def write_csv_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ['patient', 'video', 'frame', 'img_path', 'label']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def ensure_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        shutil.copy2(src, dst)


def try_import_albumentations():
    try:
        import albumentations as A  # type: ignore
        return A
    except Exception:
        return None


def build_aug_pipeline(img_size=None):
    """
    Build an augmentation pipeline. Uses albumentations if available, otherwise a minimal fallback.
    """
    A = try_import_albumentations()
    if A is None:
        return None  # fallback handled in apply_aug

    ops = [
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.7),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
    ]
    # GaussNoise signature differs across albumentations versions; pass only supported args
    try:
        import inspect
        params = inspect.signature(A.GaussNoise.__init__).parameters
        if 'var_limit' in params:
            ops.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
        else:
            ops.append(A.GaussNoise(p=0.3))
    except Exception:
        ops.append(A.GaussNoise(p=0.3))
    ops.append(A.CLAHE(clip_limit=(2, 4), p=0.2))
    if img_size is not None:
        ops.insert(0, A.LongestMaxSize(max_size=img_size))
    return A.Compose(ops)


def apply_aug(img, aug):
    if aug is None:
        # Minimal fallback: horizontal flip or slight gaussian blur
        if random.random() < 0.5:
            img2 = cv2.flip(img, 1)
        else:
            img2 = cv2.GaussianBlur(img, (3, 3), 0)
        return img2
    out = aug(image=img)
    return out['image']


def main():
    ap = argparse.ArgumentParser(description='Augment minority (label 0) frames into a new dataset root.')
    ap.add_argument('--src_root', default='data/CADICA', help='Source dataset root (expects selectedVideos/, metadata.xlsx)')
    ap.add_argument('--src_index', default='outputs/index/all_keyframes.csv', help='Source index CSV (full set)')
    ap.add_argument('--dst_root', default='data/augmented-cadica', help='Destination augmented dataset root')
    ap.add_argument('--dst_index', default='outputs/index/all_aug_keyframes.csv', help='Output augmented index CSV')
    ap.add_argument('--exclude_csv', default='outputs/index/test_pristine.csv', help='CSV with pristine test set to exclude by patient id')
    ap.add_argument('--metadata_path', default=None, help='Path to metadata.xlsx (default: <src_root>/metadata.xlsx)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--img_size', type=int, default=None, help='Optional resize for augmented images (e.g., 224)')
    ap.add_argument('--max_aug_per_frame', type=int, default=1, help='Upper bound per original frame (to prevent overuse)')
    ap.add_argument('--target_neg_ratio', type=float, default=0.5, help='Target negative frame ratio after augmentation (approximate)')
    args = ap.parse_args()

    random.seed(args.seed)
    src_root = args.src_root
    dst_root = args.dst_root
    src_idx_path = args.src_index
    dst_idx_path = args.dst_index
    meta_src = args.metadata_path or os.path.join(src_root, 'metadata.xlsx')
    meta_dst = os.path.join(dst_root, 'metadata.xlsx')

    # 1) Load source index
    rows = read_csv_rows(src_idx_path)
    if len(rows) == 0:
        raise RuntimeError('Source index is empty: %s' % src_idx_path)

    # 1b) Exclude pristine test patients if provided
    excluded_patients = set()
    if args.exclude_csv and os.path.exists(args.exclude_csv):
        try:
            with open(args.exclude_csv, 'r', newline='') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if 'patient' in row and row['patient'] != '' and row['patient'] is not None:
                        try:
                            excluded_patients.add(int(row['patient']))
                        except Exception:
                            pass
        except Exception as e:
            print(f"[warn] failed to read exclude_csv {args.exclude_csv}: {e}")

    if excluded_patients:
        rows = [r for r in rows if int(r['patient']) not in excluded_patients]
        if len(rows) == 0:
            raise RuntimeError('All rows excluded by --exclude_csv; nothing left to augment.')

    # 2) Prepare augmentation plan
    pos = [r for r in rows if r['label'] == 1]
    neg = [r for r in rows if r['label'] == 0]
    n_pos, n_neg = len(pos), len(neg)

    # compute target extra negatives to approach target_neg_ratio
    # Let total_final = n_pos + n_neg + x, with negatives_final = n_neg + x
    # We want (n_neg + x) / total_final ~ target_neg_ratio
    # => (n_neg + x) = t * (n_pos + n_neg + x) => x * (1 - t) = t*(n_pos + n_neg) - n_neg
    t = float(args.target_neg_ratio)
    if t <= 0.0 or t >= 1.0:
        t = 0.5
    rhs = t * (n_pos + n_neg) - n_neg
    extra_needed = int(math.floor(rhs / (1.0 - t))) if (1.0 - t) > 1e-6 else 0
    # Ensure at least to parity (if t==0.5 this equals n_pos - n_neg)
    extra_needed = max(0, extra_needed)

    # Keep augmentation bounded to not explode disk usage
    # Upper bound: at most 2x neg (i.e., add <= n_neg)
    extra_needed = min(extra_needed, n_pos - n_neg if n_pos > n_neg else 0)

    # 3) Prepare destination structure and metadata copy
    os.makedirs(dst_root, exist_ok=True)
    # Copy/overwrite metadata to keep consistency
    if os.path.isfile(meta_src):
        os.makedirs(os.path.dirname(meta_dst), exist_ok=True)
        shutil.copy2(meta_src, meta_dst)
    else:
        print(f"[warn] metadata not found at {meta_src}; skipping copy")

    # 4) Mirror originals to dst_root via symlink (fallback copy) and remap their paths
    aug_rows = []
    # Track max frame id per (patient, video) to generate new indices
    max_frame = defaultdict(int)
    for r in rows:
        p, v, k = r['patient'], r['video'], r['frame']
        src_img = r['img_path']
        # source expects paths like data/CADICA/selectedVideos/pX/vY/input/...
        # map into dst_root
        base_name = os.path.basename(src_img)
        dst_dir = os.path.join(dst_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
        dst_img = os.path.join(dst_dir, base_name)
        ensure_link_or_copy(src_img, dst_img)
        # update index path to point to augmented root
        aug_rows.append({
            'patient': p,
            'video': v,
            'frame': k,
            'img_path': dst_img.replace('\\', '/'),
            'label': r['label'],
        })
        if k > max_frame[(p, v)]:
            max_frame[(p, v)] = k

    # Early exit if nothing to add
    if extra_needed <= 0 or n_neg == 0:
        write_csv_rows(dst_idx_path, aug_rows)
        print(f"[info] No augmentation needed (pos={n_pos}, neg={n_neg}). Wrote mirrored index to {dst_idx_path}")
        return

    # 5) Build per-video negative frame lists for balanced augmentation
    neg_by_vid = defaultdict(list)
    for r in aug_rows:
        if r['label'] == 0:
            neg_by_vid[(r['patient'], r['video'])].append(r)

    vid_keys = list(neg_by_vid.keys())
    if not vid_keys:
        write_csv_rows(dst_idx_path, aug_rows)
        print(f"[info] No negative videos found. Wrote mirrored index to {dst_idx_path}")
        return

    # Prepare augmentation pipeline
    aug = build_aug_pipeline(img_size=args.img_size)

    # Track usage per original frame to respect max_aug_per_frame
    used_count = defaultdict(int)  # key: (patient, video, frame)

    produced = 0
    # Round-robin videos for uniform coverage
    rr_idx = 0
    attempts = 0
    max_attempts = extra_needed * 10
    while produced < extra_needed and attempts < max_attempts:
        attempts += 1
        key = vid_keys[rr_idx % len(vid_keys)]
        rr_idx += 1
        frames = neg_by_vid[key]
        if not frames:
            continue
        # pick a candidate frame with usage < max_aug_per_frame
        random.shuffle(frames)
        chosen = None
        for cand in frames:
            ck = (cand['patient'], cand['video'], cand['frame'])
            if used_count[ck] < args.max_aug_per_frame:
                chosen = cand
                break
        if chosen is None:
            continue

        p = chosen['patient']; v = chosen['video']; k = chosen['frame']
        # read original (from dst since we mirrored already)
        img = cv2.imread(chosen['img_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # fallback to source path if symlink/copy failed somehow
            src_img = rows[0]['img_path'] if rows else None
            img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[warn] Failed to read image for aug: {chosen['img_path']}")
            continue

        # expand to 3 channels for albumentations compatibility (if available)
        if len(img.shape) == 2:
            img3 = cv2.merge([img, img, img])
        else:
            img3 = img
        aug_img = apply_aug(img3, aug)
        # convert back to single channel if needed
        if aug_img.ndim == 3 and aug_img.shape[2] == 3:
            aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
        else:
            aug_gray = aug_img

        # assign new frame id and filename
        max_frame[(p, v)] = max_frame[(p, v)] + 1
        new_k = max_frame[(p, v)]
        new_name = f"p{p}_v{v}_{new_k:05d}.png"
        out_dir = os.path.join(dst_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, new_name)
        ok = cv2.imwrite(out_path, aug_gray)
        if not ok:
            print(f"[warn] Failed to write augmented image: {out_path}")
            continue

        aug_rows.append({
            'patient': p,
            'video': v,
            'frame': new_k,
            'img_path': out_path.replace('\\', '/'),
            'label': 0,
        })
        used_count[(p, v, k)] += 1
        produced += 1

    # 6) Persist index
    write_csv_rows(dst_idx_path, aug_rows)

    # 7) Manifest for bookkeeping
    manifest = {
        'src_index': src_idx_path,
        'dst_index': dst_idx_path,
        'dst_root': dst_root,
        'seed': args.seed,
        'excluded_patients': sorted(list(excluded_patients)),
        'pos_frames': int(n_pos),
        'neg_frames': int(n_neg),
        'extra_neg_created': int(produced),
        'final_total': int(len(aug_rows)),
        'final_neg': int(n_neg + produced),
        'final_pos': int(n_pos),
    }
    with open(os.path.join(os.path.dirname(dst_idx_path), 'all_aug_keyframes_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] Augmented negatives created: {produced} (pos={n_pos}, neg={n_neg} -> final_neg={n_neg+produced}).")
    print(f"[info] Wrote augmented dataset under {dst_root} and index at {dst_idx_path}")


if __name__ == '__main__':
    main()
