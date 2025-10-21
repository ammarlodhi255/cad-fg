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
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate
from src.trainloop.metrics import metrics_from_probs


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _import_base_builder():
    """Import build_models from scripts/02_train_kfold.py without executing its main()."""
    base_path = os.path.join(os.path.dirname(__file__), "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold_base", base_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader, "Failed to locate 02_train_kfold.py"
    spec.loader.exec_module(module)  # type: ignore
    return module.build_models  # type: ignore[attr-defined]


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    n = name.lower()
    if n == "adam":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if n == "lion":
        # Prefer timm Lion if available; else a minimal fallback (sign momentum)
        try:
            from timm.optim.lion import Lion as TimmLion  # type: ignore
            return TimmLion(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            # Very simple Lion-like optimizer (fallback): sign of momentum-avg grad
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
                            # update by sign of combined momentum and grad
                            d = torch.sign(m.mul(1 - beta2).add_(g, alpha=beta2))
                            p.add_(d, alpha=-lr)
                    return loss
            return _LionFallback(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def hp_grid(lrs: List[float], opts: List[str], img_sizes: List[int], pat_channels: Optional[List[int]], fusion: str):
    """Yield dicts for the HP search space.
    - Always search lr, optimizer, img_size
    - If fusion == 'early', iterate pat_channels; otherwise ignore it
    """
    if (fusion or "").lower() == "early":
        pcs = pat_channels or [4, 8]
    else:
        pcs = [None]
    for lr, opt, sz, pc in itertools.product(lrs, opts, img_sizes, pcs):
        yield dict(lr=float(lr), optimizer=str(opt), img_size=int(sz), pat_channels=pc)


def parse_feature_choice(choice: str, cfg_sel: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[str]], str]:
    """Map a user choice like 'all', 'cat+text', 'num' to (sel_nc, text_cols, combo_name).
    sel_nc -> {numeric: list|None, categorical: list|None}
    text_cols -> list|None
    combo_name -> normalized label used in paths (e.g., 'cat+text' -> 'cat_text')
    """
    choice = (choice or "").strip().lower()
    nums = cfg_sel.get("numeric") if isinstance(cfg_sel, dict) else None
    cats = cfg_sel.get("categorical") if isinstance(cfg_sel, dict) else None
    texts = cfg_sel.get("text") if isinstance(cfg_sel, dict) else None

    def has(tok: str) -> bool:
        return tok in choice.split("+") or choice == tok

    if choice in ("all", "num+cat+text", "cat+num+text", "text+num+cat"):
        sel_nc = {"numeric": nums, "categorical": cats}
        text_cols = texts
        name = "all"
    elif choice in ("num",):
        sel_nc = {"numeric": nums, "categorical": None}
        text_cols = None
        name = "num"
    elif choice in ("cat",):
        sel_nc = {"numeric": None, "categorical": cats}
        text_cols = None
        name = "cat"
    elif choice in ("text",):
        sel_nc = {"numeric": None, "categorical": None}
        text_cols = texts
        name = "text"
    elif choice in ("num+cat", "cat+num"):
        sel_nc = {"numeric": nums, "categorical": cats}
        text_cols = None
        name = "num+cat"
    elif choice in ("num+text", "text+num"):
        sel_nc = {"numeric": nums, "categorical": None}
        text_cols = texts
        name = "num+text"
    elif choice in ("cat+text", "text+cat"):
        sel_nc = {"numeric": None, "categorical": cats}
        text_cols = texts
        name = "cat+text"
    else:
        # default: treat as 'all'
        sel_nc = {"numeric": nums, "categorical": cats}
        text_cols = texts
        name = "all"
    return sel_nc, text_cols, name


def main():
    ap = argparse.ArgumentParser(description="K-fold HP search for a single model and feature selection")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    # single model + fusion
    ap.add_argument("--backbone", required=True, help="single backbone, e.g., resnet50")
    ap.add_argument("--fusion", required=True, choices=["imageonly","late","early"], help="fusion type")
    # metadata feature groups selector
    ap.add_argument("--features", default="all", help="one of: all, num, cat, text, num+cat, num+text, cat+text")
    # knobs and runtime
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outroot", default="outputs/checkpoints/cv_hpsearch_single")
    ap.add_argument("--pim_film_early", default="on", choices=["on","off"], help="Enable FiLM early fusion inside PIM models (early mode)")
    ap.add_argument("--pim_fpn_size", type=int, default=1536)
    ap.add_argument("--pim_use_selection", default="on", choices=["on","off"])
    ap.add_argument("--pim_use_combiner", default="on", choices=["on","off"])
    ap.add_argument("--pim_num_selects", default="2048,512,128,32")
    ap.add_argument("--pim_lambda_b", type=float, default=0.5)
    ap.add_argument("--pim_lambda_s", type=float, default=0.0)
    ap.add_argument("--pim_lambda_n", type=float, default=5.0)
    ap.add_argument("--pim_lambda_c", type=float, default=1.0)
    ap.add_argument("--pim_update_freq", type=int, default=2)
    ap.add_argument("--pim_amp", default="on", choices=["on","off"])
    ap.add_argument(
        "--video_reduce",
        default="all",
        choices=[
            "max","mean","median","topk2","topk3","logit_mean","ew_mean","tsmooth_mean","tsmooth_max","all"
        ],
        help="video-level probability aggregation strategy",
    )
    ap.add_argument("--select_domain", default="frame", choices=["frame","video"], help="best-epoch selection domain")
    ap.add_argument("--select_metric", default="bal_acc", help="selection metric key (e.g., bal_acc)")
    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--save-ckpt", action="store_true", help="save best weights per fold into each run folder")
    # HP spaces (as requested)
    ap.add_argument("--lrs", default="0.00001,0.0001,0.001,0.01")
    ap.add_argument("--opts", default="rmsprop,sgd,adam,lion")
    ap.add_argument("--img_sizes", default="224, 384")
    ap.add_argument("--pat_channels", default="4,8", help="only applies when --fusion early")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    patience = args.patience if args.patience is not None else cfg.get("early_stopping_patience", 20)

    os.makedirs(args.outroot, exist_ok=True)

    backbone = args.backbone.strip().lower()
    fusion = args.fusion.strip().lower()
    lr_list = [float(x) for x in args.lrs.split(',') if x.strip()]
    opt_list = [x.strip().lower() for x in args.opts.split(',') if x.strip()]
    size_list = [int(x) for x in args.img_sizes.split(',') if x.strip()]
    pc_list = [int(x) for x in args.pat_channels.split(',') if x.strip()]

    # discover folds
    fold_dirs = [d for d in sorted(os.listdir(args.folds_root)) if d.startswith("fold_")]
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {args.folds_root}.")

    # common dataset knobs
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    cfg_sel = cfg.get("metadata_features", {}) or {}
    sel_nc, text_cols, combo_name = parse_feature_choice(args.features, cfg_sel)

    # dynamic import of builder to reuse all supported backbones
    build_models = _import_base_builder()

    # Precompute fixed inner splits per fold (by patient) so all HP combos use the same data
    fold_splits = {}
    for fd in fold_dirs:
        fold_path = os.path.join(args.folds_root, fd)
        train_csv = os.path.join(fold_path, "train.csv")
        val_csv = os.path.join(fold_path, "val.csv")
        train_df = pd.read_csv(train_csv)
        # patient-level stratified split if possible
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

    summary_rows = []

    model_tag = f"{backbone}/{fusion} [{combo_name}]"
    print(f"\n=== HP-Search: {model_tag} ===")

    combo_summaries = []

    for hp in hp_grid(lr_list, opt_list, size_list, pc_list, fusion):
        pc_str = f"_pc{hp['pat_channels']}" if fusion == "early" and hp.get('pat_channels') is not None else ""
        hp_name = f"lr{hp['lr']}_opt{hp['optimizer']}_sz{hp['img_size']}{pc_str}"
        run_root = os.path.join(args.outroot, f"{backbone}_{fusion}", combo_name.replace('+','_'), hp_name)
        os.makedirs(run_root, exist_ok=True)
        print(f"\n[hp] {hp_name}")

        per_fold_metrics = []

        for fd in fold_dirs:
            fold_out = os.path.join(run_root, fd)
            os.makedirs(fold_out, exist_ok=True)
            print(f"{model_tag} | {fd} | Epoch 1/{epochs}")

            # load CSVs and fixed inner split
            split = fold_splits[fd]
            tr_csv = split['train_csv']
            va_csv = split['val_csv']
            if not (os.path.isfile(tr_csv) and os.path.isfile(va_csv)):
                print(f"[warn] missing train/val for {fd}; skipping")
                continue
            inner_tr_idx = split['inner_tr_idx']
            inner_va_idx = split['inner_va_idx']
            fit_patients = split['fit_patients']

            # datasets
            tf_train = get_train_tf(hp['img_size'])
            tf_val = get_val_tf(hp['img_size'])
            # Build base train ds to compute stats on outer-train patients
            base_train_ds = CadicaDataset(
                tr_csv, metadata, tf_train,
                selected_features=sel_nc, text_features=text_cols,
                fit_patients=fit_patients,
                text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
            )
            # enforce that late/early have tab_dim > 0
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

            # loaders (use cfg batch size)
            from torch.utils.data import Subset
            inner_tr_ds = Subset(base_train_ds, inner_tr_idx)
            inner_va_ds = Subset(base_train_val_ds, inner_va_idx)

            bs = int(cfg.get("batch_size", 64))
            tr_loader = DataLoader(inner_tr_ds, batch_size=bs, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
            in_va_loader = DataLoader(inner_va_ds, batch_size=bs, shuffle=False,
                                      num_workers=args.num_workers)
            out_va_loader = DataLoader(outer_val_ds, batch_size=bs, shuffle=False,
                                       num_workers=args.num_workers)

            # models
            # If using ViT and size=384, switch to the 384-capable variant
            bb_for_run = backbone
            if backbone == "vit_b16" and int(hp['img_size']) == 384:
                bb_for_run = "vit_b16_384"

            pat_c = int(hp['pat_channels']) if (fusion == "early" and hp.get('pat_channels') is not None) else int(cfg.get("pat_channels", 4))
            img_model, clin_mlp, late_head, params = build_models(
                backbone=bb_for_run, fusion=fusion, tab_dim=base_train_ds.tab_dim,
                pat_channels=pat_c, device=device,
                pim_film_early=(args.pim_film_early=="on"),
                pim_fpn_size=int(args.pim_fpn_size),
                pim_use_selection=(args.pim_use_selection=="on"),
                pim_use_combiner=(args.pim_use_combiner=="on"),
                pim_num_selects=(lambda s: {"layer1":int(s.split(',')[0]), "layer2":int(s.split(',')[1]), "layer3":int(s.split(',')[2]), "layer4":int(s.split(',')[3])} if s and s.count(',')==3 else None)(args.pim_num_selects),
            )

            # optimizer
            opt = make_optimizer(hp['optimizer'], params, lr=hp['lr'], weight_decay=cfg.get("weight_decay", 1e-4))

            # train with early stopping on inner validation domain/metric
            best = -1e9
            best_state: Dict[str, Any] = {"img": None, "clin": None, "late": None}
            epochs_no_improve = 0
            best_epoch = None
            best_metrics = None
            best_inner_frame = {"y": None, "p": None}
            best_inner_video_df = None
            t0 = time.time()
            for epoch in range(1, epochs + 1):
                train_one_epoch(
                    img_model, opt, tr_loader, device, mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                    pim_lambdas={"b":args.pim_lambda_b, "s":args.pim_lambda_s, "n":args.pim_lambda_n, "c":args.pim_lambda_c},
                    pim_update_freq=int(args.pim_update_freq), use_amp=(args.pim_amp=="on"),
                )
                frame_metrics, y_true_f, y_prob_f, video_metrics, video_df = evaluate(
                    img_model, in_va_loader, device, mode=fusion,
                    clin_mlp=clin_mlp, late_head=late_head, also_video=True, video_reduce=args.video_reduce,
                )
                sel_dict = frame_metrics if args.select_domain == "frame" else {k: v for k, v in video_metrics.items() if k != "_multi"}
                score = float(sel_dict.get(args.select_metric, 0.0))
                if score > best:
                    best = score
                    best_epoch = int(epoch)
                    best_state["img"] = deepcopy(img_model.state_dict())
                    best_state["clin"] = deepcopy(clin_mlp.state_dict()) if clin_mlp else None
                    best_state["late"] = deepcopy(late_head.state_dict()) if late_head else None
                    best_metrics = {
                        "frame": {k: float(v) for k, v in frame_metrics.items()},
                        "video": {k: float(v) for k, v in video_metrics.items()},
                    }
                    best_inner_frame = {"y": y_true_f, "p": y_prob_f}
                    best_inner_video_df = video_df.copy() if isinstance(video_df, pd.DataFrame) else None
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if patience and epochs_no_improve >= patience:
                    break

            # restore best and evaluate on outer validation
            if best_state["img"] is not None:
                img_model.load_state_dict(best_state["img"])
                if clin_mlp and best_state["clin"] is not None:
                    clin_mlp.load_state_dict(best_state["clin"])
                if late_head and best_state["late"] is not None:
                    late_head.load_state_dict(best_state["late"])

            outer_frame_metrics, yf_outer, pf_outer, outer_video_metrics, outer_video_df = evaluate(
                img_model, out_va_loader, device, mode=fusion,
                clin_mlp=clin_mlp, late_head=late_head, also_video=True, video_reduce=args.video_reduce,
            )

            # optionally persist best checkpoint for this fold
            if args.save_ckpt:
                ckpt = {
                    "params": {
                        "backbone": bb_for_run,
                        "fusion": fusion,
                        "optimizer": hp['optimizer'],
                        "lr": hp['lr'],
                        "img_size": hp['img_size'],
                        "pat_channels": int(hp['pat_channels']) if (fusion == "early" and hp.get('pat_channels') is not None) else 0,
                        "fold": fd,
                        "features": combo_name,
                    },
                    "model": best_state["img"],
                    "clin_mlp": best_state["clin"],
                    "late_head": best_state["late"],
                    "tab_dim": int(getattr(base_train_ds, "tab_dim", 0)),
                    "cfg": {
                        **cfg,
                        "backbone": bb_for_run,
                        "fusion": fusion,
                        "optimizer": hp['optimizer'],
                        "lr": hp['lr'],
                        "img_size": hp['img_size'],
                        "pat_channels": int(hp['pat_channels']) if (fusion == "early" and hp.get('pat_channels') is not None) else cfg.get("pat_channels", 4),
                        "metadata_features_used": {**(base_train_ds.get_feature_groups() if hasattr(base_train_ds, 'get_feature_groups') else {}), "text": text_cols},
                        "feature_choice": combo_name,
                    },
                }
                torch.save(ckpt, os.path.join(fold_out, "best.pt"))

            # GPU/timing info
            fold_train_seconds = time.time() - t0
            gpu_name = "cpu"
            alloc_mb = None
            reserved_mb = None
            if device.startswith("cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    dev_idx = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(dev_idx)
                    alloc_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024.0**2)
                    reserved_mb = torch.cuda.max_memory_reserved(dev_idx) / (1024.0**2)
                except Exception:
                    pass

            # threshold tuning using inner-val best predictions
            def _best_thr_from_arrays(y_true, y_prob, metric_key: str) -> float:
                if metric_key == "auc":
                    return 0.5
                uniq = np.unique(y_prob)
                cand = np.linspace(0.0, 1.0, num=201) if uniq.size > 200 else np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0)
                best_t, best_s = 0.5, -1.0
                for t in cand:
                    m = metrics_from_probs(y_true, y_prob, thr=float(t)).get(metric_key, float("nan"))
                    s = float(m) if np.isfinite(m) else -1.0
                    if s > best_s:
                        best_s, best_t = s, float(t)
                return best_t

            thr_frame, thr_video = 0.5, 0.5
            if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                thr_frame = _best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.select_metric)
            if best_inner_video_df is not None:
                yv_in = best_inner_video_df["y_true"].values
                pv_in = best_inner_video_df["y_prob"].values
                thr_video = _best_thr_from_arrays(yv_in, pv_in, args.select_metric)

            # tuned metrics on outer
            tuned_frame_metrics = metrics_from_probs(yf_outer, pf_outer, thr=thr_frame)
            yv = outer_video_df["y_true"].values
            pv = outer_video_df["y_prob"].values
            tuned_video_metrics = metrics_from_probs(yv, pv, thr=thr_video)

            row = {
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
            mv = outer_video_metrics.get("_multi") if isinstance(outer_video_metrics, dict) else None
            if isinstance(mv, dict):
                for agg_name, md in mv.items():
                    for mk, mvv in md.items():
                        row[f"val_video_{agg_name}_{mk}"] = float(mvv)
            per_fold_metrics.append(row)

            # save fold report
            fold_report = dict(row)
            fold_report.update({
                "selection": {"domain": args.select_domain, "metric": args.select_metric},
                "inner_best": best_metrics,
                "thresholds": {"frame": float(thr_frame), "video": float(thr_video)},
            })
            with open(os.path.join(fold_out, "val_best.json"), "w") as f:
                json.dump(fold_report, f, indent=2)

        # summarize this hp combo across folds
        if per_fold_metrics:
            df = pd.DataFrame(per_fold_metrics)
            means = df.mean(numeric_only=True).to_dict()
            sel_key = f"val_{'video' if args.select_domain=='video' else 'frame'}_{args.select_metric}"
            combo_summaries.append({"hp": hp_name, **means, "select_key": sel_key})
            with open(os.path.join(run_root, "summary.json"), "w") as f:
                json.dump({"rows": per_fold_metrics, "means": means}, f, indent=2)

    # choose best combo by mean of selection metric
    if combo_summaries:
        sel_key = combo_summaries[0]["select_key"]
        best_combo = max(combo_summaries, key=lambda r: r.get(sel_key, -1e9))
        summary_rows.append({
            "backbone": backbone,
            "fusion": fusion,
            "features": combo_name,
            **{k: v for k, v in best_combo.items() if k != "select_key"}
        })

    # write global summary
    out_sum = os.path.join(args.outroot, "hpsearch_summary.json")
    with open(out_sum, "w") as f:
        json.dump({"rows": summary_rows}, f, indent=2)
    print(f"\n[done] Wrote HP search summary -> {out_sum}")


if __name__ == "__main__":
    main()
