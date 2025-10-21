#!/usr/bin/env python3
"""
Copy existing patient-stratified folds and augment ONLY the selected class
(positives or negatives) in each fold's train.csv by a user-provided factor,
or by matching the opposite class count.
The outer held-out val for each fold is left untouched. Augmented images are
written under data/augmented-cadica/selectedVideos/... and new rows are
appended to train.csv with the augmented image paths.

Crucially, augmented samples are derived only from samples of the chosen label
already present in the train.csv of each fold (so they cannot appear in that
fold's val).

Source folds are expected at outputs/folds/patient_stratified_5fold.
Destination folds are written to outputs/folds/aug_patient_stratified_5fold.
"""
import os
import csv
import json
import shutil
import random
from collections import defaultdict
import argparse

import cv2


def try_import_albumentations():
    try:
        import albumentations as A  # type: ignore
        return A
    except Exception:
        return None


def build_aug_pipeline(img_size=None):
    A = try_import_albumentations()
    if A is None:
        return None
    ops = [
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.7),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
    ]
    # version-safe GaussNoise
    try:
        import inspect
        params = inspect.signature(A.GaussNoise.__init__).parameters
        if 'var_limit' in params:
            ops.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
        else:
            ops.append(A.GaussNoise(p=0.3))
    except Exception:
        try:
            ops.append(A.GaussNoise(p=0.3))
        except Exception:
            pass
    ops.append(A.CLAHE(clip_limit=(2, 4), p=0.2))
    if img_size is not None:
        ops.insert(0, A.LongestMaxSize(max_size=img_size))
    return A.Compose(ops)


def apply_aug(img, aug):
    if aug is None:
        # minimal fallback: H-flip or mild blur
        if random.random() < 0.5:
            return cv2.flip(img, 1)
        return cv2.GaussianBlur(img, (3, 3), 0)
    out = aug(image=img)
    return out['image']


def read_csv_rows(path):
    rows = []
    with open(path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({
                'patient': int(r['patient']),
                'video': int(r['video']),
                'frame': int(r['frame']),
                'img_path': r['img_path'],
                'label': int(r['label']),
            })
    return rows


def write_csv_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patient','video','frame','img_path','label'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def next_frame_id_for(p, v, existing_rows):
    mx = 0
    for r in existing_rows:
        if r['patient'] == p and r['video'] == v:
            if r['frame'] > mx:
                mx = r['frame']
    return mx + 1


def main():
    ap = argparse.ArgumentParser(description='Augment selected class in fold train.csv by a factor, copying existing folds.')
    ap.add_argument('--src_folds_root', default='outputs/folds/patient_stratified_5fold')
    ap.add_argument('--dst_folds_root', default='outputs/folds/aug_patient_stratified_5fold')
    ap.add_argument('--aug_root', default='data/augmented-cadica', help='Where to write augmented images')
    ap.add_argument('--augment_factor', type=float, default=1.0, help='Multiplier for the chosen label count in train.csv (e.g., 2.0 doubles). Ignored if --match_count is set.')
    ap.add_argument('--augment_label', type=int, choices=[0, 1], default=0, help='Which label to augment: 0=negatives, 1=positives')
    ap.add_argument('--match_count', action='store_true', help='If set, augment the chosen label until its final count matches the opposite class in train.csv.')
    ap.add_argument('--max_aug_per_frame', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--img_size', type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.dst_folds_root, exist_ok=True)

    # Copy top-level manifest if present
    src_manifest = os.path.join(args.src_folds_root, 'manifest.json')
    if os.path.isfile(src_manifest):
        shutil.copy2(src_manifest, os.path.join(args.dst_folds_root, 'manifest.json'))

    # discover fold dirs
    fold_dirs = [d for d in sorted(os.listdir(args.src_folds_root)) if d.startswith('fold_')]
    if not fold_dirs:
        raise RuntimeError(f'No folds found under {args.src_folds_root}')

    aug = build_aug_pipeline(img_size=args.img_size)

    fold_summaries = []
    for d in fold_dirs:
        src_fold = os.path.join(args.src_folds_root, d)
        dst_fold = os.path.join(args.dst_folds_root, d)
        os.makedirs(dst_fold, exist_ok=True)

        src_train = os.path.join(src_fold, 'train.csv')
        src_val = os.path.join(src_fold, 'val.csv')
        if not (os.path.isfile(src_train) and os.path.isfile(src_val)):
            print(f'[warn] missing train/val in {src_fold}; skipping')
            continue

        # Copy val as-is
        shutil.copy2(src_val, os.path.join(dst_fold, 'val.csv'))

        train_rows = read_csv_rows(src_train)
        # compute counts
        pos = sum(1 for r in train_rows if r['label'] == 1)
        neg = sum(1 for r in train_rows if r['label'] == 0)

        # Determine how many to add based on factor and chosen label, or match_count
        target_label = int(args.augment_label)
        orig_count = pos if target_label == 1 else neg
        other_count = neg if target_label == 1 else pos
        if args.match_count:
            factor = None
            extra_needed = max(0, other_count - orig_count)
        else:
            factor = max(1.0, float(args.augment_factor))
            extra_needed = int(max(0, round(orig_count * (factor - 1.0))))

        if extra_needed == 0:
            # just copy train unchanged
            write_csv_rows(os.path.join(dst_fold, 'train.csv'), train_rows)
            fold_summaries.append({'fold': d, 'pos': pos, 'neg': neg, 'added': 0, 'label': target_label, 'factor': factor, 'match_count': bool(args.match_count)})
            if args.match_count:
                print(f"[info] {d}: match_count no-op for label={target_label} (pos={pos}, neg={neg}); already matched or over")
            else:
                print(f"[info] {d}: factor<=1 or no-op for label={target_label} (pos={pos}, neg={neg}); no augmentation")
            continue

        # Prepare candidate rows from TRAIN ONLY, for the selected label
        cand_rows = [r for r in train_rows if r['label'] == target_label]
        if not cand_rows:
            write_csv_rows(os.path.join(dst_fold, 'train.csv'), train_rows)
            fold_summaries.append({'fold': d, 'pos': pos, 'neg': neg, 'added': 0, 'label': target_label, 'factor': factor, 'match_count': bool(args.match_count)})
            print(f"[info] {d}: no label={target_label} samples present; cannot augment")
            continue

        # Prepare allocation in rounds: use every frame once per round up to the cap
        used_count = defaultdict(int)
        produced = 0
        out_rows = list(train_rows)

        capacity = len(cand_rows) * max(0, int(args.max_aug_per_frame))
        to_add = min(extra_needed, capacity)

        for round_idx in range(max(0, int(args.max_aug_per_frame))):
            if produced >= to_add:
                break
            frames_order = cand_rows[:]
            random.shuffle(frames_order)
            for chosen in frames_order:
                if produced >= to_add:
                    break
                ck = (chosen['patient'], chosen['video'], chosen['frame'])
                if used_count[ck] >= args.max_aug_per_frame:
                    continue

                p, v, k = chosen['patient'], chosen['video'], chosen['frame']
                img = cv2.imread(chosen['img_path'], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[warn] cannot read {chosen['img_path']}; skipping one")
                    continue
                if len(img.shape) == 2:
                    img3 = cv2.merge([img, img, img])
                else:
                    img3 = img
                aug_img = apply_aug(img3, aug)
                if aug_img.ndim == 3 and aug_img.shape[2] == 3:
                    aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                else:
                    aug_gray = aug_img

                # allocate new frame id within this (p,v)
                new_k = next_frame_id_for(p, v, out_rows)
                out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                os.makedirs(out_dir, exist_ok=True)
                out_name = f'p{p}_v{v}_{new_k:05d}.png'
                out_path = os.path.join(out_dir, out_name)
                ok = cv2.imwrite(out_path, aug_gray)
                if not ok:
                    print(f"[warn] failed to write {out_path}; skipping one")
                    continue

                out_rows.append({
                    'patient': p,
                    'video': v,
                    'frame': new_k,
                    'img_path': out_path.replace('\\', '/'),
                    'label': target_label,
                })
                used_count[(p, v, k)] += 1
                produced += 1

        write_csv_rows(os.path.join(dst_fold, 'train.csv'), out_rows)
        fold_summaries.append({'fold': d, 'pos': pos, 'neg': neg, 'added': produced, 'label': target_label, 'factor': factor, 'match_count': bool(args.match_count)})
        if args.match_count:
            print(f"[done] {d}: pos={pos}, neg={neg}, added={produced} label={target_label} match_count=True -> total={len(out_rows)}")
        else:
            print(f"[done] {d}: pos={pos}, neg={neg}, added={produced} label={target_label} factor={factor} -> total={len(out_rows)}")

    # summary manifest
    with open(os.path.join(args.dst_folds_root, 'aug_manifest.json'), 'w') as f:
        json.dump({'folds': fold_summaries}, f, indent=2)


if __name__ == '__main__':
    main()
