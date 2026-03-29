#!/usr/bin/env python3
import os
import sys
import json
import time
import yaml
import argparse
import random
import itertools
import importlib.util
from copy import deepcopy
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_train_tf_basic, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate
from src.trainloop.metrics import metrics_from_probs
import cv2


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _import_base_builder():
    base_path = os.path.join(os.path.dirname(__file__), "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold_base", base_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader, "Failed to locate 02_train_kfold.py"
    spec.loader.exec_module(module)  # type: ignore
    return module.build_models  # type: ignore[attr-defined]


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    n = (name or "").lower()
    if n in ("adamw", "adam"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if n == "lion":
        try:
            from timm.optim.lion import Lion as TimmLion  # type: ignore
            return TimmLion(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            class _LionFallback(torch.optim.Optimizer):
                def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                    super().__init__(params, defaults)
                @torch.no_grad()
                def step(self, closure=None):
                    loss = None if closure is None else closure()
                    for group in self.param_groups:
                        beta1, beta2 = group['betas']
                        lr = group['lr']
                        wd = group['weight_decay']
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            g = p.grad
                            if wd:
                                p.mul_(1 - lr * wd)
                            state = self.state[p]
                            if 'exp_avg' not in state:
                                state['exp_avg'] = torch.zeros_like(p)
                            m = state['exp_avg']
                            m.mul_(beta1).add_(g, alpha=1 - beta1)
                            d = torch.sign(m.mul(1 - beta2).add_(g, alpha=beta2))
                            p.add_(d, alpha=-lr)
                    return loss
            return _LionFallback(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def parse_feature_choice(cfg_sel: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[str]], str]:
    nums = cfg_sel.get("numeric") if isinstance(cfg_sel, dict) else None
    cats = cfg_sel.get("categorical") if isinstance(cfg_sel, dict) else None
    texts = cfg_sel.get("text") if isinstance(cfg_sel, dict) else None
    sel_nc = {"numeric": nums, "categorical": cats}
    return sel_nc, texts, "all"


def best_thr_from_arrays(y_true, y_prob, metric_key: str, min_spec: float | None = None, min_sens: float | None = None) -> float:
    if metric_key == "auc":
        return 0.5
    uniq = np.unique(y_prob)
    cand = np.unique(np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
    best_t, best_s, best_sens = 0.5, -1.0, -1.0
    fb_t, fb_s, fb_sens = 0.5, -1.0, -1.0
    eps = 1e-12
    for t in cand:
        md = metrics_from_probs(y_true, y_prob, thr=float(t))
        m = md.get(metric_key, float("nan"))
        s = float(m) if np.isfinite(m) else -1.0
        sens = float(md.get("sensitivity", float("nan")))
        if not np.isfinite(sens):
            sens = -1.0
        # fallback unconstrained
        fb_update = False
        if s > fb_s:
            fb_update = True
        elif abs(s - fb_s) <= eps and sens > fb_sens:
            fb_update = True
        elif abs(s - fb_s) <= eps and abs(sens - fb_sens) <= eps and float(t) < float(fb_t):
            fb_update = True
        if fb_update:
            fb_s, fb_sens, fb_t = s, sens, float(t)
        # constraints
        if min_spec is not None:
            spec = float(md.get("specificity", float("nan")))
            if not np.isfinite(spec) or spec + eps < float(min_spec):
                continue
        if min_sens is not None and (not np.isfinite(sens) or sens + eps < float(min_sens)):
            continue
        update = False
        if s > best_s:
            update = True
        elif abs(s - best_s) <= eps and sens > best_sens:
            update = True
        elif abs(s - best_s) <= eps and abs(sens - best_sens) <= eps and float(t) < float(best_t):
            update = True
        if update:
            best_s, best_sens, best_t = s, sens, float(t)
    if (min_spec is not None or min_sens is not None) and best_s < 0.0:
        return fb_t
    return best_t


def main():
    ap = argparse.ArgumentParser(description="K-fold HP search across backbones and optimizers/LRs; mirrors 02_train_kfold.py behavior.")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--fusions", default="imageonly,late,early")
    ap.add_argument("--backbones", default="resnet50,resnet18,effnet_b0,convnext_tiny,inception_v3,densenet121,mobilenet_v2,vit_b16")
    ap.add_argument("--outroot", default="outputs/checkpoints/cv_hpsearch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--video_reduce", default="tsmooth_max",
                    choices=["max","mean","median","topk2","topk3","logit_mean","ew_mean","tsmooth_mean","tsmooth_max","all"],
                    help="video-level probability aggregation strategy")
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--save_checkpoints", action="store_true", help="Save best.pt per fold (off by default)")
    ap.add_argument("--save_val_probs", action="store_true", help="Save val_video_probs.csv per fold (off by default)")
    ap.add_argument("--select_domain", default="frame", choices=["frame","video"], help="Which validation domain to select best epoch on")
    ap.add_argument("--select_metric", default="auc", help="Validation metric key to select best HP (e.g., auc, bal_acc)")
    ap.add_argument("--thr_metric", default="bal_acc", help="Metric to tune decision thresholds (e.g., bal_acc, f1, sensitivity)")
    ap.add_argument("--min_specificity", type=float, default=None)
    ap.add_argument("--min_sensitivity", type=float, default=None)
    ap.add_argument("--min_classic_sensitivity", type=float, default=None)
    ap.add_argument("--require_floors", action="store_true")
    ap.add_argument("--inner_val_frac", type=float, default=0.3)
    ap.add_argument("--balanced_inner_val", action="store_true")
    ap.add_argument("--inner_val_target_pos_frac", type=float, default=0.5)
    ap.add_argument("--inner_val_max_instances", type=int, default=None)
    ap.add_argument("--joint_select", action="store_true")
    ap.add_argument("--tune_end", "--tune-end", action="store_true")
    ap.add_argument("--eval_outer_each_epoch", action="store_true")
    ap.add_argument("--augment", action="store_true", help="On-disk inner-train augmentation")
    ap.add_argument("--special_aug", action="store_true", help="Extended/legacy augmentation; default is basic affine only")
    ap.add_argument("--aug_label", type=int, choices=[0,1], default=0)
    ap.add_argument("--aug_match_count", action="store_true")
    ap.add_argument("--aug_factor", type=float, default=1.0)
    ap.add_argument("--aug_max_per_frame", type=int, default=5)
    ap.add_argument("--aug_img_size", type=int, default=None)
    ap.add_argument("--aug_root", default="data/augmented-cadica")
    ap.add_argument("--log_aug_examples", action="store_true")
    ap.add_argument("--augment_pos_factor_neg_half", type=int, default=None)
    ap.add_argument("--aug_neg_percent", type=float, default=None)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--class_weights", type=str, default=None)
    ap.add_argument("--amp", default="off", choices=["on","off"], help="Enable AMP")
    ap.add_argument("--final_test_eval", action="store_true", help="After CV, retrain on train+val and evaluate on test (not used in HP search)")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv")
    ap.add_argument("--trainval_pool_csv", default="outputs/index/trainval_pool.csv")
    # HP grid
    ap.add_argument("--lrs", default="0.1,0.01,0.001,0.0001")
    ap.add_argument("--opts", default="sgd,adamw,adam,lion,rmsprop")
    ap.add_argument("--imgsz", default="224,384,640", help="Image sizes to sweep (comma-separated)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    patience = args.patience if args.patience is not None else cfg.get("early_stopping_patience", 20)
    bs = int(cfg.get("batch_size", 64))

    os.makedirs(args.outroot, exist_ok=True)

    fusions = [x.strip().lower() for x in args.fusions.split(',') if x.strip()]
    backbones = [x.strip().lower() for x in args.backbones.split(',') if x.strip()]
    lr_list = [float(x) for x in args.lrs.split(',') if x.strip()]
    opt_list = [x.strip().lower() for x in args.opts.split(',') if x.strip()]
    size_list = [int(x) for x in args.imgsz.split(',') if x.strip()]

    # discover folds
    fold_dirs = [d for d in sorted(os.listdir(args.folds_root)) if d.startswith("fold_")]
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {args.folds_root}.")

    # metadata and features
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    cfg_sel = cfg.get("metadata_features", {}) or {}
    sel_nc, text_cols, combo_name = parse_feature_choice(cfg_sel)

    build_models = _import_base_builder()

    # Precompute fixed inner splits per fold (by patient), mirroring 02_train_kfold.py
    fold_splits: Dict[str, Dict[str, Any]] = {}
    for fd in fold_dirs:
        fold_path = os.path.join(args.folds_root, fd)
        train_csv = os.path.join(fold_path, "train.csv")
        val_csv = os.path.join(fold_path, "val.csv")
        train_df = pd.read_csv(train_csv)
        if args.balanced_inner_val:
            pats = []
            for pid, g in train_df.groupby("patient"):
                pid = int(pid)
                pos = int((g["label"] == 1).sum())
                neg = int((g["label"] == 0).sum())
                tot = pos + neg
                if tot <= 0:
                    continue
                frac = pos / float(max(1, tot))
                pats.append({"patient": pid, "pos": pos, "neg": neg, "tot": tot, "pos_frac": frac})
            target = float(args.inner_val_target_pos_frac)
            pats.sort(key=lambda x: (abs(x["pos_frac"] - target), -x["tot"], x["patient"]))
            n_target = max(1, int(round(len(pats) * float(args.inner_val_frac))))
            selected = []
            remaining = pats[:]
            pos_sum = 0
            neg_sum = 0
            cap = int(args.inner_val_max_instances) if args.inner_val_max_instances is not None else None
            while len(selected) < n_target and remaining:
                if cap is not None and (pos_sum + neg_sum) >= cap:
                    break
                best_i = None
                best_key = None
                for i, cand in enumerate(remaining):
                    if cap is not None and (pos_sum + neg_sum + cand["tot"]) > cap:
                        continue
                    new_pos = pos_sum + cand["pos"]
                    new_neg = neg_sum + cand["neg"]
                    new_tot = max(1, new_pos + new_neg)
                    new_frac = new_pos / float(new_tot)
                    delta = abs(new_frac - target)
                    key = (delta, -cand["tot"], cand["patient"])  # prefer larger patients, then smaller id
                    if best_i is None or key < best_key:
                        best_i, best_key = i, key
                if best_i is None:
                    break
                cand = remaining.pop(best_i)
                selected.append(cand)
                pos_sum += cand["pos"]
                neg_sum += cand["neg"]
            va_pats = set([c["patient"] for c in selected])
            tr_pats = set([c["patient"] for c in pats if c["patient"] not in va_pats])
        else:
            pl_tr = train_df.groupby("patient")["label"].max().reset_index().rename(columns={"label": "patient_label"})
            y_p = pl_tr["patient_label"].astype(int).values
            if len(np.unique(y_p)) >= 2:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=seed)
                tr_pidx, va_pidx = next(sss.split(np.zeros(len(pl_tr)), y_p))
                tr_pats = set(pl_tr.iloc[tr_pidx]["patient"].tolist())
                va_pats = set(pl_tr.iloc[va_pidx]["patient"].tolist())
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=seed)
                tr_idx_tmp, va_idx_tmp = next(gss.split(train_df, groups=train_df["patient"]))
                tr_pats = set(train_df.iloc[tr_idx_tmp]["patient"].unique().tolist())
                va_pats = set(train_df.iloc[va_idx_tmp]["patient"].unique().tolist())
        inner_tr_idx = np.where(train_df["patient"].isin(tr_pats))[0]
        inner_va_idx = np.where(train_df["patient"].isin(va_pats))[0]
        fit_patients = sorted(train_df["patient"].astype(int).unique().tolist())
        fold_splits[fd] = dict(train_csv=train_csv, val_csv=val_csv,
                               inner_tr_idx=inner_tr_idx, inner_va_idx=inner_va_idx,
                               fit_patients=fit_patients)

    global_rows: List[Dict[str, Any]] = []

    for backbone in backbones:
        for fusion in fusions:
            model_tag = f"{backbone}/{fusion} [{combo_name}]"
            print(f"\n=== HP-Search: {model_tag} ===")
            combo_results: List[Dict[str, Any]] = []

            for lr, opt_name, img_size in itertools.product(lr_list, opt_list, size_list):
                hp_name = f"lr{lr}_opt{opt_name}_sz{img_size}"
                run_root = os.path.join(args.outroot, f"{backbone}_{fusion}", combo_name.replace('+','_'), hp_name)
                os.makedirs(run_root, exist_ok=True)
                print(f"\n[hp] {hp_name}")

                per_fold_metrics: List[Dict[str, Any]] = []

                for fd in fold_dirs:
                    fold_out = os.path.join(run_root, fd)
                    os.makedirs(fold_out, exist_ok=True)
                    print(f"{model_tag} | {fd} | Epoch 1/{epochs}")

                    split = fold_splits[fd]
                    tr_csv = split['train_csv']
                    va_csv = split['val_csv']
                    if not (os.path.isfile(tr_csv) and os.path.isfile(va_csv)):
                        print(f"[warn] missing train/val for {fd}; skipping")
                        continue
                    inner_tr_idx = split['inner_tr_idx']
                    inner_va_idx = split['inner_va_idx']
                    fit_patients = split['fit_patients']

                    # transforms
                    tf_train = get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)
                    tf_val = get_val_tf(img_size)

                    # datasets
                    base_train_ds = CadicaDataset(
                        tr_csv, metadata, tf_train,
                        selected_features=sel_nc, text_features=text_cols,
                        fit_patients=fit_patients,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )
                    if fusion in ("late", "early") and getattr(base_train_ds, "tab_dim", 0) <= 0:
                        print(f"[skip] features '{combo_name}' yield tab_dim=0; not valid for fusion {fusion}")
                        continue
                    _stats = base_train_ds.stats
                    base_train_val_ds = CadicaDataset(
                        tr_csv, metadata, tf_val,
                        selected_features=sel_nc, text_features=text_cols,
                        stats=_stats,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )
                    outer_val_ds = CadicaDataset(
                        va_csv, metadata, tf_val,
                        selected_features=sel_nc, text_features=text_cols,
                        stats=_stats,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                    )

                    # optional size limits
                    if args.limit_train is not None:
                        inner_tr_idx = np.array(inner_tr_idx)[: int(args.limit_train)]
                    if args.limit_val is not None:
                        inner_va_idx = np.array(inner_va_idx)[: int(args.limit_val)]

                    # Build default inner-train/val datasets
                    inner_tr_ds = Subset(base_train_ds, inner_tr_idx)
                    inner_va_ds = Subset(base_train_val_ds, inner_va_idx)

                    # Augmentation pipeline (mirrors 02_train_kfold.py), applied to inner-train only as on-disk synthesis
                    class_weights = None
                    if args.augment_pos_factor_neg_half is not None or args.aug_neg_percent is not None or args.augment:
                        try:
                            import albumentations as A  # type: ignore
                        except Exception:
                            A = None  # type: ignore

                        def _make_augmenter(img_size=None, special=False):
                            if A is None:
                                return None
                            if special:
                                ops = [
                                    A.HorizontalFlip(p=0.5),
                                    A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-10,10), p=0.7),
                                    A.RandomBrightnessContrast(0.15,0.15,p=0.5),
                                    A.MotionBlur(blur_limit=3,p=0.2),
                                ]
                                try:
                                    import inspect as _inspect
                                    params = _inspect.signature(A.GaussNoise.__init__).parameters
                                    if 'var_limit' in params:
                                        ops.append(A.GaussNoise(var_limit=(10.0,50.0), p=0.3))
                                    else:
                                        ops.append(A.GaussNoise(p=0.3))
                                except Exception:
                                    try:
                                        ops.append(A.GaussNoise(p=0.3))
                                    except Exception:
                                        pass
                                ops.append(A.CLAHE(clip_limit=(2,4), p=0.2))
                                if img_size is not None:
                                    ops.insert(0, A.LongestMaxSize(max_size=img_size))
                                return A.Compose(ops)
                            try:
                                use_px = 'translate_px' in __import__('inspect').signature(A.Affine.__init__).parameters
                            except Exception:
                                use_px = False
                            ops = [A.LongestMaxSize(max_size=img_size)] if img_size is not None else []
                            if use_px:
                                ops.append(A.Affine(translate_px={"x": (-25,25), "y": (-25,25)}, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                            else:
                                base = img_size if img_size is not None else 224
                                shift = 25.0/float(base)
                                ops.append(A.Affine(translate_percent=shift, scale=(0.8,1.7), rotate=(-25,25), p=1.0))
                            return A.Compose(ops)

                        def _apply_aug(img3, aug):
                            if aug is None:
                                h, w = img3.shape[:2]
                                cx, cy = w/2.0, h/2.0
                                angle = random.uniform(-25.0, 25.0)
                                scale = random.uniform(0.8, 1.7)
                                dx = random.uniform(-25.0, 25.0)
                                dy = random.uniform(-25.0, 25.0)
                                M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
                                M[0,2] += dx
                                M[1,2] += dy
                                return cv2.warpAffine(img3, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                            out = aug(image=img3)
                            return out['image']

                    # If new special augmentation is requested, it overrides other aug args
                    if args.augment_pos_factor_neg_half is not None:
                        fpos = max(0, int(args.augment_pos_factor_neg_half))
                        fold_out = os.path.join(run_root, fd)
                        os.makedirs(fold_out, exist_ok=True)
                        inner_tr_df = pd.read_csv(tr_csv).iloc[inner_tr_idx].copy()
                        pos_df = inner_tr_df[inner_tr_df['label'] == 1]
                        neg_df = inner_tr_df[inner_tr_df['label'] == 0]
                        pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                        pos_after = pos_ct + pos_ct * fpos
                        target_neg_total = int(np.ceil(0.5 * pos_after))
                        neg_needed = max(0, target_neg_total - neg_ct)
                        aug = _make_augmenter(args.aug_img_size, special=args.special_aug) if ('_make_augmenter' in locals()) else None
                        print(f"[augment-pos-neg-half] {fd} enabled | pos_factor={fpos} | target_neg_total={target_neg_total} | neg_needed={neg_needed} | albu={'on' if aug is not None else 'off'}")
                        aug_rows = []
                        # positives
                        if fpos > 0 and pos_ct > 0:
                            for r in pos_df.itertuples(index=False):
                                p = int(r.patient); v = int(r.video); k = int(r.frame)
                                sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                                base_max = int(sub['frame'].max()) if len(sub) else 0
                                for i in range(fpos):
                                    img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is None:
                                        continue
                                    img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug) if ('_apply_aug' in locals()) else img3
                                    aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    new_k = base_max + 1 + i
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    ok = cv2.imwrite(out_path, aug_gray)
                                    if not ok:
                                        continue
                                    aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 1})
                        # negatives
                        if neg_needed > 0 and neg_ct > 0:
                            produced = 0
                            rr = 0
                            neg_list = list(neg_df.itertuples(index=False))
                            while produced < neg_needed and len(neg_list) > 0:
                                for row in neg_list:
                                    if produced >= neg_needed:
                                        break
                                    p = int(row.patient); v = int(row.video); k = int(row.frame)
                                    sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                                    base_max = int(sub['frame'].max()) if len(sub) else 0
                                    extra_for_pv = sum(1 for r in aug_rows if r['patient']==p and r['video']==v)
                                    new_k = base_max + extra_for_pv + 1
                                    img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is None:
                                        continue
                                    img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                    aug_img = _apply_aug(img3, aug) if ('_apply_aug' in locals()) else img3
                                    aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                    out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                    os.makedirs(out_dir, exist_ok=True)
                                    out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                    out_path = os.path.join(out_dir, out_name)
                                    ok = cv2.imwrite(out_path, aug_gray)
                                    if not ok:
                                        continue
                                    aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0})
                                    produced += 1
                                rr += 1
                        if len(aug_rows) > 0:
                            inner_aug_df = pd.concat([inner_tr_df, pd.DataFrame(aug_rows)], ignore_index=True)
                            aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                            is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                            def _ok_path(pth: str) -> bool:
                                try:
                                    return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                except Exception:
                                    return False
                            ok_mask = (~is_aug) | inner_aug_df['img_path'].astype(str).apply(_ok_path)
                            inner_aug_df = inner_aug_df[ok_mask].reset_index(drop=True)
                            aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                            inner_aug_df.to_csv(aug_csv, index=False)
                            inner_tr_ds = CadicaDataset(
                                aug_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
                                selected_features=sel_nc, text_features=text_cols,
                                text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                stats=base_train_ds.stats,
                            )
                            # compute class weights from augmented DF if requested
                            if args.weighted:
                                manual = args.class_weights
                                if manual:
                                    try:
                                        cw = [float(x) for x in manual.split(',') if x.strip()]
                                        if len(cw) == 2 and all(np.isfinite(c) and c > 0 for c in cw):
                                            class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)
                                        else:
                                            class_weights = None
                                    except Exception:
                                        class_weights = None
                                if class_weights is None:
                                    new_pos = int((inner_aug_df['label'] == 1).sum())
                                    new_neg = int((inner_aug_df['label'] == 0).sum())
                                    tot = float(new_pos + new_neg)
                                    w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                    w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                    elif args.aug_neg_percent is not None:
                        r = float(args.aug_neg_percent)
                        if not (0.0 < r < 1.0):
                            print(f"[augment-neg-percent] {fd} invalid aug_neg_percent={r}; skipping this mode")
                        else:
                            fold_out = os.path.join(run_root, fd)
                            os.makedirs(fold_out, exist_ok=True)
                            inner_tr_df = pd.read_csv(tr_csv).iloc[inner_tr_idx].copy()
                            pos_df = inner_tr_df[inner_tr_df['label'] == 1]
                            neg_df = inner_tr_df[inner_tr_df['label'] == 0]
                            pos_ct, neg_ct = int(len(pos_df)), int(len(neg_df))
                            target_neg_total = int(np.ceil((r / max(1e-6, 1.0 - r)) * float(pos_ct)))
                            neg_needed = max(0, target_neg_total - neg_ct)
                            aug = _make_augmenter(args.aug_img_size, special=args.special_aug) if ('_make_augmenter' in locals()) else None
                            print(f"[augment-neg-percent] {fd} enabled | r={r:.3f} | target_neg_total={target_neg_total} | neg_needed={neg_needed} | albu={'on' if aug is not None else 'off'}")
                            aug_rows = []
                            # only negatives augmented here
                            if neg_needed > 0 and neg_ct > 0:
                                produced = 0
                                neg_list = list(neg_df.itertuples(index=False))
                                while produced < neg_needed and len(neg_list) > 0:
                                    for row in neg_list:
                                        if produced >= neg_needed:
                                            break
                                        p = int(row.patient); v = int(row.video); k = int(row.frame)
                                        sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                                        base_max = int(sub['frame'].max()) if len(sub) else 0
                                        extra_for_pv = sum(1 for r in aug_rows if r['patient']==p and r['video']==v)
                                        new_k = base_max + extra_for_pv + 1
                                        img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                        if img is None:
                                            continue
                                        img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                        aug_img = _apply_aug(img3, aug) if ('_apply_aug' in locals()) else img3
                                        aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                        out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                        os.makedirs(out_dir, exist_ok=True)
                                        out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                        out_path = os.path.join(out_dir, out_name)
                                        ok = cv2.imwrite(out_path, aug_gray)
                                        if not ok:
                                            continue
                                        aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': 0})
                                        produced += 1
                            if len(aug_rows) > 0:
                                inner_aug_df = pd.concat([inner_tr_df, pd.DataFrame(aug_rows)], ignore_index=True)
                                aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                                is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                                def _ok_path(pth: str) -> bool:
                                    try:
                                        return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                    except Exception:
                                        return False
                                ok_mask = (~is_aug) | inner_aug_df['img_path'].astype(str).apply(_ok_path)
                                inner_aug_df = inner_aug_df[ok_mask].reset_index(drop=True)
                                aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                                inner_aug_df.to_csv(aug_csv, index=False)
                                inner_tr_ds = CadicaDataset(
                                    aug_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
                                    selected_features=sel_nc, text_features=text_cols,
                                    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                    stats=base_train_ds.stats,
                                )
                                if args.weighted:
                                    new_pos = int((inner_aug_df['label'] == 1).sum())
                                    new_neg = int((inner_aug_df['label'] == 0).sum())
                                    tot = float(new_pos + new_neg)
                                    w0 = tot / (2.0 * max(1.0, float(new_neg)))
                                    w1 = tot / (2.0 * max(1.0, float(new_pos)))
                                    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                    elif args.augment:
                        fold_out = os.path.join(run_root, fd)
                        os.makedirs(fold_out, exist_ok=True)
                        inner_tr_df = pd.read_csv(tr_csv).iloc[inner_tr_idx].copy()
                        tgt_label = int(args.aug_label)
                        factor = float(args.aug_factor)
                        aug = _make_augmenter(args.aug_img_size, special=args.special_aug) if ('_make_augmenter' in locals()) else None
                        print(f"[augment] {fd} enabled | label={tgt_label} | match={bool(args.aug_match_count)} | factor={None if args.aug_match_count else factor} | max_per={int(args.aug_max_per_frame)} | albu={'on' if aug is not None else 'off'}")
                        src_df = inner_tr_df[inner_tr_df['label'] == tgt_label]
                        opp_df = inner_tr_df[inner_tr_df['label'] != tgt_label]
                        S = int(len(src_df)); O = int(len(opp_df))
                        if args.aug_match_count:
                            target_ct = max(0, O - S)
                        else:
                            target_ct = max(0, int(round(S * max(0.0, float(factor) - 1.0))))
                        aug_rows = []
                        produced = 0
                        for row in src_df.itertuples(index=False):
                            if produced >= target_ct:
                                break
                            p = int(row.patient); v = int(row.video); k = int(row.frame)
                            sub = inner_tr_df[(inner_tr_df['patient']==p) & (inner_tr_df['video']==v)]
                            base_max = int(sub['frame'].max()) if len(sub) else 0
                            per = 0
                            while per < int(args.aug_max_per_frame) and produced < target_ct:
                                img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    break
                                img3 = cv2.merge([img, img, img]) if len(img.shape) == 2 else img
                                aug_img = _apply_aug(img3, aug) if ('_apply_aug' in locals()) else img3
                                aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY) if aug_img.ndim == 3 and aug_img.shape[2] == 3 else aug_img
                                out_dir = os.path.join(args.aug_root, 'selectedVideos', f'p{p}', f'v{v}', 'input')
                                os.makedirs(out_dir, exist_ok=True)
                                extra_for_pv = sum(1 for r in aug_rows if r['patient']==p and r['video']==v)
                                new_k = base_max + extra_for_pv + 1
                                out_name = f'p{p}_v{v}_{new_k:05d}.png'
                                out_path = os.path.join(out_dir, out_name)
                                ok = cv2.imwrite(out_path, aug_gray)
                                if not ok:
                                    break
                                aug_rows.append({'patient': p,'video': v,'frame': new_k,'img_path': out_path.replace('\\','/'),'label': tgt_label})
                                produced += 1
                                per += 1
                        if len(aug_rows) > 0:
                            inner_aug_df = pd.concat([inner_tr_df, pd.DataFrame(aug_rows)], ignore_index=True)
                            aug_tag = os.path.join(args.aug_root, 'selectedVideos').replace('\\','/')
                            is_aug = inner_aug_df['img_path'].astype(str).str.contains(aug_tag)
                            def _ok_path(pth: str) -> bool:
                                try:
                                    return os.path.exists(pth) and (cv2.imread(pth, cv2.IMREAD_GRAYSCALE) is not None)
                                except Exception:
                                    return False
                            ok_mask = (~is_aug) | inner_aug_df['img_path'].astype(str).apply(_ok_path)
                            inner_aug_df = inner_aug_df[ok_mask].reset_index(drop=True)
                            aug_csv = os.path.join(fold_out, 'inner_train_aug.csv')
                            inner_aug_df.to_csv(aug_csv, index=False)
                            inner_tr_ds = CadicaDataset(
                                aug_csv, metadata, (get_train_tf(img_size) if args.special_aug else get_train_tf_basic(img_size)),
                                selected_features=sel_nc, text_features=text_cols,
                                text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                                text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                                stats=base_train_ds.stats,
                            )
                            if args.weighted:
                                cnt_pos = int((inner_aug_df['label'] == 1).sum())
                                cnt_neg = int((inner_aug_df['label'] == 0).sum())
                                tot = float(cnt_pos + cnt_neg)
                                w0 = tot / (2.0 * max(1.0, float(cnt_neg)))
                                w1 = tot / (2.0 * max(1.0, float(cnt_pos)))
                                class_weights = torch.tensor([w0, w1], dtype=torch.float32)

                    # DataLoaders
                    tr_loader = DataLoader(inner_tr_ds, batch_size=bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    in_va_loader = DataLoader(inner_va_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers)
                    out_va_loader = DataLoader(outer_val_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers)

                    # models
                    bb_for_run = backbone
                    if backbone == "vit_b16" and int(img_size) == 384:
                        bb_for_run = "vit_b16_384"
                    pat_channels = int(cfg.get("pat_channels", 4))
                    img_model, clin_mlp, late_head, params = build_models(
                        backbone=bb_for_run, fusion=fusion, tab_dim=base_train_ds.tab_dim,
                        pat_channels=pat_channels, device=device,
                    )

                    # optimizer
                    wd = float(cfg.get("weight_decay", 0.0))
                    opt = make_optimizer(opt_name, params, lr=float(lr), weight_decay=wd)

                    # train with early stopping
                    best = -1e9
                    best_sel_sens = -1.0
                    best_sel_thr: Optional[float] = None
                    best_state: Dict[str, Any] = {"img": None, "clin": None, "late": None}
                    epochs_no_improve = 0
                    best_epoch = None
                    best_metrics = None
                    best_inner_frame = {"y": None, "p": None}
                    best_inner_video_df = None
                    t0 = time.time()
                    use_amp_flag = (args.amp == "on")
                    # if --weighted requested but not set via augmentation blocks, compute from current inner-train
                    if args.weighted and class_weights is None:
                        manual = args.class_weights
                        if manual:
                            try:
                                cw = [float(x) for x in manual.split(',') if x.strip()]
                                if len(cw) == 2 and all(np.isfinite(c) and c > 0 for c in cw):
                                    class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)
                                    print(f"[loss] class_weights set manually [neg={cw[0]:.3f}, pos={cw[1]:.3f}] (pre-train)")
                            except Exception:
                                pass
                        if class_weights is None:
                            # estimate from inner split labels
                            inner_tr_df_counts = pd.read_csv(tr_csv).iloc[inner_tr_idx]
                            cnt_pos = int((inner_tr_df_counts['label'] == 1).sum())
                            cnt_neg = int((inner_tr_df_counts['label'] == 0).sum())
                            tot = float(cnt_pos + cnt_neg)
                            if tot > 0:
                                w0 = tot / (2.0 * max(1.0, float(cnt_neg)))
                                w1 = tot / (2.0 * max(1.0, float(cnt_pos)))
                                class_weights = torch.tensor([w0, w1], dtype=torch.float32)

                    for epoch in range(1, epochs + 1):
                        improved_this_epoch = False
                        # Epoch header with since_best counter (parity with 02_train_kfold.py)
                        try:
                            print(f"\n{backbone}/{fusion} | {fd} | Epoch {epoch}/{epochs} | since_best={epochs_no_improve}")
                        except Exception:
                            print(f"\n{backbone}/{fusion} | {fd} | Epoch {epoch}/{epochs}")
                        train_one_epoch(
                            img_model, opt, tr_loader, device, mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                            use_amp=use_amp_flag, class_weights=class_weights,
                        )
                        frame_metrics, y_true_f, y_prob_f, video_metrics, video_df = evaluate(
                            img_model, in_va_loader, device, mode=fusion,
                            clin_mlp=clin_mlp, late_head=late_head, also_video=True, video_reduce=args.video_reduce,
                        )
                        # Print inner-val metrics (frame + video) similar to 02_train_kfold.py
                        try:
                            print("[inner_val/frame]", {k: round(float(v), 4) for k, v in frame_metrics.items()})
                        except Exception:
                            print("[inner_val/frame]", frame_metrics)
                        try:
                            print("[inner_val/video]", {k: round(float(v), 4) for k, v in video_metrics.items() if k != "_multi"})
                        except Exception:
                            print("[inner_val/video]", {k: v for k, v in video_metrics.items() if k != "_multi"})
                        if args.eval_outer_each_epoch:
                            ofm, _, _, ovm, _ = evaluate(
                                img_model, out_va_loader, device, mode=fusion,
                                clin_mlp=clin_mlp, late_head=late_head, also_video=True, video_reduce=args.video_reduce,
                            )
                            print("[outer_val-each-epoch/frame]", {k: round(float(v), 4) for k, v in ofm.items()})
                            print("[outer_val-each-epoch/video]", {k: round(float(v), 4) for k, v in ovm.items() if k != "_multi"})
                        # selection: classic or joint (mirrors 02_train_kfold.py)
                        if args.joint_select:
                            if args.select_domain == "frame":
                                y_sel, p_sel = y_true_f, y_prob_f
                            else:
                                y_sel = video_df["y_true"].values
                                p_sel = video_df["y_prob"].values
                            thr_cand = best_thr_from_arrays(y_sel, p_sel, args.thr_metric, args.min_specificity, args.min_sensitivity)
                            md = metrics_from_probs(y_sel, p_sel, thr=float(thr_cand))
                            score = float(md.get(args.thr_metric, 0.0))
                            sens = float(md.get("sensitivity", 0.0)) if np.isfinite(md.get("sensitivity", float("nan"))) else 0.0
                            spec = float(md.get("specificity", 0.0)) if np.isfinite(md.get("specificity", float("nan"))) else 0.0
                            if args.require_floors:
                                need_spec = args.min_specificity if args.min_specificity is not None else None
                                need_sens = args.min_sensitivity if args.min_sensitivity is not None else None
                                feasible = True
                                if need_spec is not None and spec + 1e-12 < float(need_spec):
                                    feasible = False
                                if need_sens is not None and sens + 1e-12 < float(need_sens):
                                    feasible = False
                                if not feasible:
                                    epochs_no_improve += 1
                                    continue
                            eps = 1e-12
                            update = False
                            if score > best:
                                update = True
                            elif abs(score - best) <= eps and sens > best_sel_sens:
                                update = True
                            elif abs(score - best) <= eps and abs(sens - best_sel_sens) <= eps and float(thr_cand) < float(best_sel_thr if best_sel_thr is not None else 1.0):
                                update = True
                            if update:
                                best = score
                                best_sel_sens = sens
                                best_sel_thr = float(thr_cand)
                                # store metrics (exclude _multi)
                                try:
                                    best_metrics = {
                                        "frame": {k: float(v) for k, v in frame_metrics.items() if np.isfinite(v)},
                                        "video": {k: float(v) for k, v in video_metrics.items() if k != "_multi" and np.isfinite(v)},
                                    }
                                except Exception:
                                    best_metrics = {
                                        "frame": {k: v for k, v in frame_metrics.items() if k},
                                        "video": {k: v for k, v in video_metrics.items() if k != "_multi"},
                                    }
                                best_epoch = int(epoch)
                                best_state["img"] = deepcopy(img_model.state_dict())
                                best_state["clin"] = deepcopy(clin_mlp.state_dict()) if clin_mlp else None
                                best_state["late"] = deepcopy(late_head.state_dict()) if late_head else None
                                if args.save_checkpoints:
                                    try:
                                        torch.save({
                                            "img": best_state["img"],
                                            "clin": best_state["clin"],
                                            "late": best_state["late"],
                                            "best_epoch": best_epoch,
                                            "hp": hp_name,
                                            "thr_metric": args.thr_metric,
                                            "joint_select": True,
                                            "sel_thr": best_sel_thr,
                                        }, os.path.join(fold_out, "best.pt"))
                                    except Exception:
                                        pass
                                best_inner_frame = {"y": y_true_f, "p": y_prob_f}
                                best_inner_video_df = video_df.copy() if isinstance(video_df, pd.DataFrame) else None
                                improved_this_epoch = True
                        else:
                            sel_dict = frame_metrics if args.select_domain == "frame" else {k: v for k, v in video_metrics.items() if k != "_multi"}
                            score = float(sel_dict.get(args.select_metric, 0.0)) if args.select_metric in sel_dict else 0.0
                            cur_satisfies = True
                            if args.min_classic_sensitivity is not None:
                                try:
                                    cur_sens = float(sel_dict.get("sensitivity", float("nan")))
                                except Exception:
                                    cur_sens = float("nan")
                                if not np.isfinite(cur_sens) or cur_sens + 1e-12 < float(args.min_classic_sensitivity):
                                    cur_satisfies = False
                            if cur_satisfies and score > best:
                                best = score
                                try:
                                    best_metrics = {
                                        "frame": {k: float(v) for k, v in frame_metrics.items() if np.isfinite(v)},
                                        "video": {k: float(v) for k, v in video_metrics.items() if k != "_multi" and np.isfinite(v)},
                                    }
                                except Exception:
                                    best_metrics = {
                                        "frame": {k: v for k, v in frame_metrics.items() if k},
                                        "video": {k: v for k, v in video_metrics.items() if k != "_multi"},
                                    }
                                best_epoch = int(epoch)
                                best_state["img"] = deepcopy(img_model.state_dict())
                                best_state["clin"] = deepcopy(clin_mlp.state_dict()) if clin_mlp else None
                                best_state["late"] = deepcopy(late_head.state_dict()) if late_head else None
                                if args.save_checkpoints:
                                    try:
                                        torch.save({
                                            "img": best_state["img"],
                                            "clin": best_state["clin"],
                                            "late": best_state["late"],
                                            "best_epoch": best_epoch,
                                            "hp": hp_name,
                                        }, os.path.join(fold_out, "best.pt"))
                                    except Exception:
                                        pass
                                best_inner_frame = {"y": y_true_f, "p": y_prob_f}
                                best_inner_video_df = video_df.copy() if isinstance(video_df, pd.DataFrame) else None
                                improved_this_epoch = True

                        if improved_this_epoch:
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if patience and epochs_no_improve >= patience:
                            print(f"[early-stopping] stopping at epoch {epoch}")
                            break

                    # restore best and evaluate on outer validation
                    if best_state["img"] is not None:
                        img_model.load_state_dict(best_state["img"])
                        if clin_mlp and best_state["clin"] is not None:
                            clin_mlp.load_state_dict(best_state["clin"])
                        if late_head and best_state["late"] is not None:
                            late_head.load_state_dict(best_state["late"])

                    # optional evaluate outer each epoch already handled in main 02_train_kfold.py; here evaluate after training
                    outer_frame_metrics, yf_outer, pf_outer, outer_video_metrics, outer_video_df = evaluate(
                        img_model, out_va_loader, device, mode=fusion,
                        clin_mlp=clin_mlp, late_head=late_head, also_video=True, video_reduce=args.video_reduce,
                    )

                    # threshold tuning: joint vs classic/tune_end (mirror 02_train_kfold.py)
                    thr_frame = 0.5
                    thr_video = 0.5
                    did_tune = False
                    if args.joint_select and best_sel_thr is not None:
                        if args.select_domain == "frame":
                            thr_frame = float(best_sel_thr)
                            if best_inner_video_df is not None:
                                yv_in = best_inner_video_df["y_true"].values
                                pv_in = best_inner_video_df["y_prob"].values
                                thr_video = best_thr_from_arrays(yv_in, pv_in, args.thr_metric, args.min_specificity, args.min_sensitivity)
                        else:
                            thr_video = float(best_sel_thr)
                            if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                                thr_frame = best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.thr_metric, args.min_specificity, args.min_sensitivity)
                        did_tune = True
                    elif args.tune_end:
                        if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                            thr_frame = best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.thr_metric, args.min_specificity, args.min_sensitivity)
                        if best_inner_video_df is not None:
                            yv_in = best_inner_video_df["y_true"].values
                            pv_in = best_inner_video_df["y_prob"].values
                            thr_video = best_thr_from_arrays(yv_in, pv_in, args.thr_metric, args.min_specificity, args.min_sensitivity)
                        did_tune = True

                    tuned_frame_metrics = metrics_from_probs(yf_outer, pf_outer, thr=thr_frame)
                    yv = outer_video_df["y_true"].values
                    pv = outer_video_df["y_prob"].values
                    tuned_video_metrics = metrics_from_probs(yv, pv, thr=thr_video)

                    # timing + GPU
                    fold_train_seconds = time.time() - t0
                    gpu_name = "cpu"; alloc_mb = None; reserved_mb = None
                    if device.startswith("cuda") and torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                            dev_idx = torch.cuda.current_device()
                            gpu_name = torch.cuda.get_device_name(dev_idx)
                            alloc_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024.0**2)
                            reserved_mb = torch.cuda.max_memory_reserved(dev_idx) / (1024.0**2)
                        except Exception:
                            pass

                    row = {
                        "backbone": backbone,
                        "fusion": fusion,
                        "hp": hp_name,
                        "fold": int(fd.split('_')[-1]),
                        "best_epoch": best_epoch if best_epoch is not None else -1,
                        "train_seconds": float(fold_train_seconds),
                        "device": str(device),
                        "gpu_name": str(gpu_name),
                        "cuda_mem_alloc_peak_mb": float(alloc_mb) if alloc_mb is not None else None,
                        "cuda_mem_reserved_peak_mb": float(reserved_mb) if reserved_mb is not None else None,
                        **{f"val_frame_{k}": float(v) for k, v in tuned_frame_metrics.items()},
                        **{f"val_video_{k}": float(v) for k, v in tuned_video_metrics.items()},
                    }
                    per_fold_metrics.append(row)

                    # Persist optional artifacts
                    if args.save_val_probs and isinstance(outer_video_df, pd.DataFrame):
                        try:
                            outer_video_df.to_csv(os.path.join(fold_out, "val_video_probs.csv"), index=False)
                        except Exception:
                            pass
                    if args.save_checkpoints:
                        try:
                            torch.save({
                                "img": best_state["img"],
                                "clin": best_state["clin"],
                                "late": best_state["late"],
                                "best_epoch": best_epoch,
                                "hp": hp_name,
                            }, os.path.join(fold_out, "best.pt"))
                        except Exception:
                            pass

                    # persist fold report
                    fold_report = dict(row)
                    fold_report.update({
                        "selection": {"domain": args.select_domain, "metric": args.select_metric, "thr_metric": args.thr_metric, "joint": bool(args.joint_select), "tune_end": bool(args.tune_end), "min_specificity": args.min_specificity, "min_sensitivity": args.min_sensitivity, "min_classic_sensitivity": args.min_classic_sensitivity},
                        "thresholds": {"frame": float(thr_frame), "video": float(thr_video)},
                        "cfg": {"img_size": img_size, "batch_size": bs, "weight_decay": float(cfg.get("weight_decay", 0.0))},
                    })
                    with open(os.path.join(fold_out, "val_best.json"), "w") as f:
                        json.dump(fold_report, f, indent=2)

                # summarize across folds for this HP point
                if per_fold_metrics:
                    df = pd.DataFrame(per_fold_metrics)
                    means = df.mean(numeric_only=True).to_dict()
                    sel_key = f"val_{'video' if args.select_domain=='video' else 'frame'}_{args.select_metric}"
                    combo_results.append({"hp": hp_name, **means, "select_key": sel_key})
                    with open(os.path.join(run_root, "summary.json"), "w") as f:
                        json.dump({"rows": per_fold_metrics, "means": means}, f, indent=2)

            # choose best HP for this backbone/fusion by selection metric mean
            if combo_results:
                sel_key = combo_results[0]["select_key"]
                best_combo = max(combo_results, key=lambda r: r.get(sel_key, -1e9))
                global_rows.append({
                    "backbone": backbone,
                    "fusion": fusion,
                    **{k: v for k, v in best_combo.items() if k != "select_key"}
                })

    # global summary
    out_sum = os.path.join(args.outroot, "hpsearch_summary.json")
    with open(out_sum, "w") as f:
        json.dump({"rows": global_rows}, f, indent=2)
    print(f"\n[done] Wrote HP search summary -> {out_sum}")


if __name__ == "__main__":
    main()
