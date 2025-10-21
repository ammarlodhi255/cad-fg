#!/usr/bin/env python3
import os
import sys
import json
import time
import yaml
import argparse
import random
from copy import deepcopy
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate
from src.trainloop.metrics import metrics_from_probs


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _import_build_models():
    """Import build_models from 02_train_kfold.py to reuse all supported backbones/fusions."""
    import importlib.util
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
    if n in ("sgd", "sgdm"):
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    # default
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def combos_from_cfg(cfg_sel: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], List[str] | None]]:
    """
    Build 6 metadata combinations using all available features in cfg:
      - num, cat, text, num+cat, num+text, cat+text
    Returns a list of tuples: (combo_name, selected_features_dict, text_cols)
    """
    nums = cfg_sel.get("numeric") if isinstance(cfg_sel, dict) else None
    cats = cfg_sel.get("categorical") if isinstance(cfg_sel, dict) else None
    texts = cfg_sel.get("text") if isinstance(cfg_sel, dict) else None

    out = []
    out.append(("num", {"numeric": nums, "categorical": None}, None))
    out.append(("cat", {"numeric": None, "categorical": cats}, None))
    out.append(("text", {"numeric": None, "categorical": None}, texts))
    out.append(("num+cat", {"numeric": nums, "categorical": cats}, None))
    out.append(("num+text", {"numeric": nums, "categorical": None}, texts))
    out.append(("cat+text", {"numeric": None, "categorical": cats}, texts))
    return out


def _compute_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    cols = [c for c in df.columns if c.startswith("val_")]
    for col in sorted(cols):
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "n": int(s.shape[0]),
        }
    return stats


def _fmt_hms(seconds: float) -> str:
    try:
        seconds = float(seconds or 0.0)
    except Exception:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _write_combo_summary_txt(path: str, backbone: str, fusion: str, combo: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stats = _compute_stats(df)
    lines = []
    lines.append(f"Model: {backbone}/{fusion} [{combo}]")
    lines.append(f"Folds completed: {max(df['fold'])+1 if 'fold' in df.columns else df.shape[0]}")
    for col in sorted(stats.keys()):
        st = stats[col]
        lines.append(f"{col}: {st['mean']:.4f} ± {st['std']:.4f} (min={st['min']:.4f}, max={st['max']:.4f}, n={st['n']})")

    if "convergence_epoch" in df.columns:
        try:
            cavg = float(pd.to_numeric(df["convergence_epoch"], errors="coerce").mean())
            lines.append(f"Avg convergence epoch: {cavg:.2f}")
        except Exception:
            pass
    if "train_seconds" in df.columns:
        try:
            tsum = float(pd.to_numeric(df["train_seconds"], errors="coerce").sum())
            tmean = float(pd.to_numeric(df["train_seconds"], errors="coerce").mean())
            lines.append(f"Total train time: {_fmt_hms(tsum)} (mean per fold: {_fmt_hms(tmean)})")
        except Exception:
            pass
    if "cuda_mem_alloc_peak_mb" in df.columns:
        try:
            mu = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").mean())
            sd = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").std())
            lines.append(f"GPU peak alloc: {mu:.1f} ± {sd:.1f} MB")
        except Exception:
            pass
    if "cuda_mem_reserved_peak_mb" in df.columns:
        try:
            mu = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").mean())
            sd = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").std())
            lines.append(f"GPU peak reserved: {mu:.1f} ± {sd:.1f} MB")
        except Exception:
            pass

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="K-fold CV for meta-feature ablations (num/cat/text combinations).")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--fusions", default="imageonly,late,early")
    ap.add_argument("--backbones", default="resnet50,resnet18,effnet_b0,convnext_tiny")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outroot", default="outputs/checkpoints/cv_meta")
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
    ap.add_argument("--save_checkpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    patience = args.patience if args.patience is not None else cfg.get("early_stopping_patience", 20)

    os.makedirs(args.outroot, exist_ok=True)

    fusions = [x.strip().lower() for x in args.fusions.split(',') if x.strip()]
    backbones = [x.strip().lower() for x in args.backbones.split(',') if x.strip()]

    # global dataset knobs
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    cfg_sel = cfg.get("metadata_features", {}) or {}
    combos = combos_from_cfg(cfg_sel)

    # discover folds
    fold_dirs = [d for d in sorted(os.listdir(args.folds_root)) if d.startswith("fold_")]
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {args.folds_root}.")

    build_models = _import_build_models()

    global_rows = []

    for fusion in fusions:
        for backbone in backbones:
            for combo_name, sel_nc, text_cols in combos:
                # Early fusion requires tabular features; skip if combo yields no features
                # We'll detect per-fold using tab_dim computed on base_train_ds.
                tag = f"{backbone}/{fusion}/{combo_name}"
                print(f"\n=== {tag} ===")
                run_root = os.path.join(args.outroot, f"{backbone}_{fusion}", combo_name.replace('+', '_'))
                os.makedirs(run_root, exist_ok=True)

                fold_metrics: List[Dict[str, Any]] = []

                for fold_name in fold_dirs:
                    fold_idx = int(fold_name.split('_')[-1])
                    fold_path = os.path.join(args.folds_root, fold_name)
                    train_csv = os.path.join(fold_path, "train.csv")
                    val_csv = os.path.join(fold_path, "val.csv")
                    fold_out = os.path.join(run_root, fold_name)
                    os.makedirs(fold_out, exist_ok=True)
                    print(f"{tag} | {fold_name} | Epoch 1/{epochs}")

                    # Read outer-train CSV for patient groups and stats
                    train_df = pd.read_csv(train_csv)
                    fit_patients = sorted(train_df["patient"].astype(int).unique().tolist())

                    # base datasets with this combo
                    base_train_ds = CadicaDataset(
                        train_csv, metadata, get_train_tf(cfg.get("img_size", 224)),
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                        fit_patients=fit_patients,
                    )
                    if fusion in ("late", "early") and base_train_ds.tab_dim <= 0:
                        print(f"[skip] combo {combo_name} yields tab_dim=0 for fusion {fusion}; skipping fold")
                        continue
                    _stats = base_train_ds.stats

                    base_train_val_ds = CadicaDataset(
                        train_csv, metadata, get_val_tf(cfg.get("img_size", 224)),
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                        stats=_stats,
                    )
                    outer_val_ds = CadicaDataset(
                        val_csv, metadata, get_val_tf(cfg.get("img_size", 224)),
                        selected_features=sel_nc,
                        text_features=text_cols,
                        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                        stats=_stats,
                    )

                    # inner split by patient on outer-train
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

                    inner_tr_ds = Subset(base_train_ds, inner_tr_idx)
                    inner_va_ds = Subset(base_train_val_ds, inner_va_idx)

                    tr_loader = DataLoader(inner_tr_ds, batch_size=cfg.get("batch_size", 64), shuffle=True,
                                           num_workers=args.num_workers, pin_memory=True)
                    in_va_loader = DataLoader(inner_va_ds, batch_size=cfg.get("batch_size", 64), shuffle=False,
                                              num_workers=args.num_workers)
                    out_va_loader = DataLoader(outer_val_ds, batch_size=cfg.get("batch_size", 64), shuffle=False,
                                               num_workers=args.num_workers)

                    # build model(s)
                    def _parse_num_selects(s: str):
                        try:
                            a,b,c,d = [int(x) for x in s.split(',')]
                            return {"layer1":a, "layer2":b, "layer3":c, "layer4":d}
                        except Exception:
                            return None
                    pim_num_selects = _parse_num_selects(args.pim_num_selects)

                    img_model, clin_mlp, late_head, params = build_models(
                        backbone=backbone, fusion=fusion, tab_dim=base_train_ds.tab_dim,
                        pat_channels=cfg.get("pat_channels", 4), device=device,
                        pim_film_early=(args.pim_film_early=="on"),
                        pim_fpn_size=int(args.pim_fpn_size),
                        pim_use_selection=(args.pim_use_selection=="on"),
                        pim_use_combiner=(args.pim_use_combiner=="on"),
                        pim_num_selects=pim_num_selects,
                    )

                    opt = make_optimizer(cfg.get("optimizer", "adamw"), params,
                                        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4))

                    # train with early stopping based on requested domain/metric
                    best = -1e9
                    best_state: Dict[str, Any] = {"img": None, "clin": None, "late": None}
                    best_metrics = None
                    best_inner_frame = {"y": None, "p": None}
                    best_inner_video_df = None
                    epochs_no_improve = 0
                    best_epoch = None
                    t0 = time.time()
                    for epoch in range(1, epochs + 1):
                        train_one_epoch(
                            img_model, opt, tr_loader, device, mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                            pim_lambdas={"b":args.pim_lambda_b, "s":args.pim_lambda_s, "n":args.pim_lambda_n, "c":args.pim_lambda_c},
                            pim_update_freq=int(args.pim_update_freq), use_amp=(args.pim_amp=="on"),
                        )
                        frame_metrics, y_true_f, y_prob_f, video_metrics, video_df = evaluate(
                            img_model, in_va_loader, device, mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                            also_video=True, video_reduce=args.video_reduce,
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
                            best_inner_video_df = video_df.copy()
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if patience and epochs_no_improve >= patience:
                            break

                    # restore best and evaluate on outer held-out validation
                    if best_state["img"] is not None:
                        img_model.load_state_dict(best_state["img"])
                        if clin_mlp and best_state["clin"] is not None:
                            clin_mlp.load_state_dict(best_state["clin"])
                        if late_head and best_state["late"] is not None:
                            late_head.load_state_dict(best_state["late"])

                    outer_frame_metrics, yf_outer, pf_outer, outer_video_metrics, outer_video_df = evaluate(
                        img_model, out_va_loader, device, mode=fusion, clin_mlp=clin_mlp, late_head=late_head,
                        also_video=True, video_reduce=args.video_reduce,
                    )

                    # GPU/timing info
                    fold_train_seconds = time.time() - t0
                    gpu_name = "cpu"; alloc_mb = None; reserved_mb = None
                    if device.startswith("cuda") and torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize(); dev_idx = torch.cuda.current_device()
                            gpu_name = torch.cuda.get_device_name(dev_idx)
                            alloc_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024.0**2)
                            reserved_mb = torch.cuda.max_memory_reserved(dev_idx) / (1024.0**2)
                        except Exception:
                            pass

                    # threshold tuning from inner best
                    def _best_thr_from_arrays(y_true, y_prob, metric_key: str) -> float:
                        if metric_key == "auc":
                            return 0.5
                        uniq = np.unique(y_prob)
                        cand = np.linspace(0.0, 1.0, num=201) if uniq.size > 200 else np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0)
                        best_t, best_s = 0.5, -1.0
                        for t in cand:
                            m = metrics_from_probs(y_true, y_prob, thr=float(t)).get(args.select_metric, float("nan"))
                            s = float(m) if np.isfinite(m) else -1.0
                            if s > best_s:
                                best_s, best_t = s, float(t)
                        return best_t

                    thr_frame = 0.5; thr_video = 0.5
                    if best_inner_frame["y"] is not None and best_inner_frame["p"] is not None:
                        thr_frame = _best_thr_from_arrays(best_inner_frame["y"], best_inner_frame["p"], args.select_metric)
                    if isinstance(best_inner_video_df, pd.DataFrame):
                        yv_in = best_inner_video_df["y_true"].values; pv_in = best_inner_video_df["y_prob"].values
                        thr_video = _best_thr_from_arrays(yv_in, pv_in, args.select_metric)

                    tuned_frame_metrics = metrics_from_probs(yf_outer, pf_outer, thr=thr_frame)
                    yv = outer_video_df["y_true"].values; pv = outer_video_df["y_prob"].values
                    tuned_video_metrics = metrics_from_probs(yv, pv, thr=thr_video)

                    row = {
                        "backbone": backbone,
                        "fusion": fusion,
                        "combo": combo_name,
                        "fold": fold_idx,
                        "best_epoch": int(best_epoch) if best_epoch is not None else None,
                        "train_seconds": float(fold_train_seconds),
                        "device": str(device),
                        "gpu_name": str(gpu_name),
                        "cuda_mem_alloc_peak_mb": float(alloc_mb) if alloc_mb is not None else None,
                        "cuda_mem_reserved_peak_mb": float(reserved_mb) if reserved_mb is not None else None,
                        **{f"val_frame_{k}": float(tuned_frame_metrics[k]) for k in tuned_frame_metrics},
                        **{f"val_video_{k}": float(tuned_video_metrics[k]) for k in tuned_video_metrics},
                    }
                    mv = outer_video_metrics.get("_multi") if isinstance(outer_video_metrics, dict) else None
                    if isinstance(mv, dict):
                        for agg_name, md in mv.items():
                            for mk, mvv in md.items():
                                row[f"val_video_{agg_name}_{mk}"] = float(mvv)
                    fold_metrics.append(row)

                    # persist fold report
                    with open(os.path.join(fold_out, "val_best.json"), "w") as f:
                        json.dump({
                            **row,
                            "selection": {"domain": args.select_domain, "metric": args.select_metric},
                            "thresholds": {"frame": float(thr_frame), "video": float(thr_video)},
                            "inner_best": best_metrics,
                        }, f, indent=2)

                # summarize per combo
                if fold_metrics:
                    df = pd.DataFrame(fold_metrics)
                    df.to_csv(os.path.join(run_root, "fold_metrics.csv"), index=False)
                    stats = _compute_stats(df)
                    # write human-readable summary
                    _write_combo_summary_txt(os.path.join(run_root, "summary.txt"), backbone, fusion, combo_name, df)
                    # persist machine-readable
                    with open(os.path.join(run_root, "summary.json"), "w") as f:
                        json.dump({"stats": stats, "rows": fold_metrics}, f, indent=2)
                    # add to global rows with mean/std/min/max per metric
                    row = {"backbone": backbone, "fusion": fusion, "combo": combo_name}
                    for k, st in stats.items():
                        row[f"mean_{k}"] = st["mean"]
                        row[f"std_{k}"] = st["std"]
                        row[f"min_{k}"] = st["min"]
                        row[f"max_{k}"] = st["max"]
                    # extras
                    if "convergence_epoch" in df.columns:
                        try:
                            row["avg_convergence_epoch"] = float(pd.to_numeric(df["convergence_epoch"], errors="coerce").mean())
                        except Exception:
                            pass
                    if "train_seconds" in df.columns:
                        try:
                            row["total_train_seconds"] = float(pd.to_numeric(df["train_seconds"], errors="coerce").sum())
                            row["mean_train_seconds"] = float(pd.to_numeric(df["train_seconds"], errors="coerce").mean())
                        except Exception:
                            pass
                    if "cuda_mem_alloc_peak_mb" in df.columns:
                        try:
                            row["mean_cuda_alloc_mb"] = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").mean())
                            row["std_cuda_alloc_mb"] = float(pd.to_numeric(df["cuda_mem_alloc_peak_mb"], errors="coerce").std())
                        except Exception:
                            pass
                    if "cuda_mem_reserved_peak_mb" in df.columns:
                        try:
                            row["mean_cuda_reserved_mb"] = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").mean())
                            row["std_cuda_reserved_mb"] = float(pd.to_numeric(df["cuda_mem_reserved_peak_mb"], errors="coerce").std())
                        except Exception:
                            pass
                    global_rows.append(row)

    # global summary across all runs
    if global_rows:
        sdf = pd.DataFrame(global_rows)
        out_csv = os.path.join(args.outroot, "cv_meta_summary.csv")
        sdf.to_csv(out_csv, index=False)
        # also write concise text summary
        lines = ["Cross-Model CV Summary (meta)"]
        for _, r in sdf.iterrows():
            lines.append("")
            lines.append(f"Model: {r['backbone']}/{r['fusion']} [{r['combo']}]")
            # print a few key metrics if available
            for key in ("val_frame_bal_acc", "val_frame_f1", "val_video_bal_acc", "val_video_f1"):
                m = f"mean_{key}"
                s = f"std_{key}"
                if m in r and s in r:
                    try:
                        lines.append(f"{key}: {float(r[m]):.4f} ± {float(r[s]):.4f}")
                    except Exception:
                        pass
        with open(os.path.join(args.outroot, "cv_meta_summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[done] wrote global meta summary -> {out_csv}")


if __name__ == "__main__":
    main()
