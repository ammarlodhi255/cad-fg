'''

TRY WEIGHTED SAMPLING
Try ap as select metric

'''


import os
import sys
import json
import time
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from temporal_reasoning.datasets import TemporalClipDataset, temporal_collate
from temporal_reasoning.model_vit_temporal import TemporalViTClassifier
from temporal_reasoning.model_vit_temporal_meta import TemporalViTEarlyFusionClassifier
from temporal_reasoning.model_vit_temporal_late import TemporalViTLateFusionClassifier
from src.dataio.dataset import load_metadata_any
from temporal_reasoning.engine import train_one_epoch_temporal, evaluate_temporal
from src.trainloop.metrics import metrics_from_probs


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser(description="K-fold temporal ViT training (video-level) without modifying base code.")
    ap.add_argument("--cfg", default=os.path.join("temporal_reasoning", "vit_temporal.yaml"))
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--outroot", default="outputs/temporal_vit")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--video_reduce", default="tsmooth_max", choices=["max","mean","median","topk2","topk3","logit_mean","ew_mean","tsmooth_mean","tsmooth_max","topk_auto","tsmooth_topk_auto","lse"])
    ap.add_argument("--video_topk_frac", type=float, default=0.05)
    ap.add_argument("--video_topk_cap", type=int, default=30)
    ap.add_argument("--video_smooth_win", type=int, default=5)
    ap.add_argument("--video_smooth_kind", default="median", choices=["median","mean"])
    ap.add_argument("--video_lse_alpha", type=float, default=8.0)
    ap.add_argument("--fusions", default="imageonly,early,late", help="Comma-separated list of fusions: imageonly,early,late")
    ap.add_argument("--variants", default=None, help="Comma-separated early-fusion variants to run (plain,meta_token,film,prompt,cls_cross,patient)")
    ap.add_argument("--save_checkpoints", action="store_true", help="Save best.pt on improvement per fold/fusion")
    ap.add_argument("--save_val_probs", action="store_true", help="Save validation video probs CSV when a new best is found")
    ap.add_argument("--select_metric", default="bal_acc", choices=["ap","auc","bal_acc","f1"], help="Validation metric key for best-epoch selection (e.g., ap, auc, bal_acc, f1)")
    ap.add_argument("--threshold_metric", default="bal_acc", help="Metric to maximize when tuning decision threshold on validation (e.g., bal_acc, acc, f1)")
    ap.add_argument("--amp", default="off", choices=["on","off"]) 
    ap.add_argument("--weighted_sampling", action="store_true", help="Use inverse-frequency class-balanced sampling for training videos")
    ap.add_argument("--inner_val_frac", type=float, default=0.2, help="Fraction of fold train used for inner validation")
    ap.add_argument("--eval_outer_each_epoch", action="store_true", help="Also log outer validation each epoch (not used for selection)")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 1337))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = int(cfg.get("img_size", 224))
    T = int(cfg.get("num_frames", 8))
    epochs = int(args.epochs or cfg.get("epochs", 30))
    batch_size = int(args.batch_size or cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 3e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    inner_val_frac = float(cfg.get("inner_val_frac", 0.2))
    # metadata & model variant
    use_metadata = bool(cfg.get("use_metadata", False))
    metadata_path = cfg.get("metadata_path")
    meta_features = cfg.get("metadata_features")  # dict with numeric/categorical/text
    text_cols = None
    if isinstance(meta_features, dict) and meta_features.get("text"):
        text_cols = list(meta_features.get("text"))
    text_model = cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")
    text_cache_dir = cfg.get("text_cache_dir", "outputs/text_embeds")
    vit_variant = str(cfg.get("vit_temporal_variant", "plain")).lower()  # 'plain'|'meta_token'|'film'|'prompt'|'cls_cross'|'patient'
    pat_channels = int(cfg.get("pat_channels", 4))
    prompt_K = int(cfg.get("prompt_tokens", 4))
    cls_cross_heads = int(cfg.get("cls_cross_heads", 8))

    fold_dirs = sorted([d for d in os.listdir(args.folds_root) if d.startswith("fold_")])
    os.makedirs(args.outroot, exist_ok=True)

    summary_rows = []

    fusions = [f.strip().lower() for f in str(args.fusions).split(',') if f.strip()]
    variants = [v.strip().lower() for v in str(args.variants or vit_variant).split(',') if v.strip()]

    # Model-major ordering: iterate variants x fusions, then folds
    for variant in variants:
        for fusion in fusions:
            run_root = os.path.join(args.outroot, f"{variant}", f"{fusion}")
            os.makedirs(run_root, exist_ok=True)
            fold_metrics = []

            for fold_name in fold_dirs:
                fold_idx = int(fold_name.split("_")[-1])
                fold_path = os.path.join(args.folds_root, fold_name)
                train_csv = os.path.join(fold_path, "train.csv")
                val_csv = os.path.join(fold_path, "val.csv")

                # Datasets (+metadata if enabled)
                fit_patients = None
                if use_metadata:
                    tr_df = pd.read_csv(train_csv)
                    fit_patients = sorted(tr_df["patient"].astype(int).unique().tolist())
                tr_ds_vis = TemporalClipDataset(
                    train_csv,
                    img_size=img_size,
                    num_frames=T,
                    sample_mode='random',
                    is_train=True,
                    metadata_path=(metadata_path if use_metadata else None),
                    selected_features=meta_features,
                    fit_patients=fit_patients,
                    stats=None,
                    text_features=(text_cols if use_metadata else None),
                    text_model=text_model,
                    text_cache_dir=text_cache_dir,
                )
                va_ds = TemporalClipDataset(
                    val_csv,
                    img_size=img_size,
                    num_frames=T,
                    sample_mode='chunked',
                    is_train=False,
                    metadata_path=(metadata_path if use_metadata else None),
                    selected_features=meta_features,
                    fit_patients=None,
                    stats=getattr(tr_ds_vis, 'meta_stats', None),
                    text_features=(text_cols if use_metadata else None),
                    text_model=text_model,
                    text_cache_dir=text_cache_dir,
                )
                # Build inner train/val splits (patient-level)
                train_df_full = pd.read_csv(train_csv)
                pl = train_df_full.groupby('patient')['label'].max().reset_index().rename(columns={'label':'patient_label'})
                y_p = pl['patient_label'].astype(int).values
                if len(np.unique(y_p)) >= 2:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=seed)
                    tr_idx_p, va_idx_p = next(sss.split(np.zeros(len(pl)), y_p))
                    tr_pats = set(pl.iloc[tr_idx_p]['patient'].tolist()); va_pats = set(pl.iloc[va_idx_p]['patient'].tolist())
                else:
                    gss = GroupShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=seed)
                    tr_i, va_i = next(gss.split(train_df_full, groups=train_df_full['patient']))
                    tr_pats = set(train_df_full.iloc[tr_i]['patient'].unique().tolist()); va_pats = set(train_df_full.iloc[va_i]['patient'].unique().tolist())

                # Map dataset items (videos) to patient sets
                def _idx_by_pat(ds, pats: set):
                    idxs = []
                    for i, (key_idx, _) in enumerate(ds.ptrs):
                        pid, vid = ds.index.keys[key_idx]
                        if int(pid) in pats:
                            idxs.append(i)
                    return idxs
                inner_tr_idx = _idx_by_pat(tr_ds_vis, tr_pats)
                inner_va_idx = _idx_by_pat(tr_ds_vis, va_pats)

                # Inner datasets/loaders
                from torch.utils.data import Subset
                inner_tr_ds = Subset(tr_ds_vis, inner_tr_idx)
                base_train_val = TemporalClipDataset(
                    train_csv,
                    img_size=img_size,
                    num_frames=T,
                    sample_mode='chunked',
                    is_train=False,
                    metadata_path=(metadata_path if use_metadata else None),
                    selected_features=meta_features,
                    fit_patients=None,
                    stats=getattr(tr_ds_vis, 'meta_stats', None),
                    text_features=(text_cols if use_metadata else None),
                    text_model=text_model,
                    text_cache_dir=text_cache_dir,
                )
                inner_va_ds = Subset(base_train_val, inner_va_idx)

                # DataLoaders (optionally weighted on inner_tr)
                if args.weighted_sampling:
                    labels = []
                    for i in inner_tr_idx:
                        key_idx, _ = tr_ds_vis.ptrs[i]
                        pid, vid = tr_ds_vis.index.keys[key_idx]
                        lab = int(tr_ds_vis.index.video_labels.get((pid, vid), 0))
                        labels.append(lab)
                    labels = np.asarray(labels, dtype=int)
                    counts = np.bincount(labels, minlength=2).astype(float)
                    counts[counts == 0] = 1.0
                    inv = 1.0 / counts
                    weights = torch.tensor(inv[labels], dtype=torch.double)
                    sampler = WeightedRandomSampler(weights, num_samples=len(inner_tr_idx), replacement=True)
                    tr_loader = DataLoader(inner_tr_ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=temporal_collate)
                else:
                    tr_loader = DataLoader(inner_tr_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=temporal_collate)
                inner_va_loader = DataLoader(inner_va_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=temporal_collate)
                outer_va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=temporal_collate)

                # Determine tab_dim if metadata present (per fold)
                tab_dim = None
                if hasattr(tr_ds_vis, 'meta_map') and tr_ds_vis.meta_map:
                    any_pid = next(iter(tr_ds_vis.meta_map.keys()))
                    tab_dim = int(len(tr_ds_vis.meta_map[any_pid]))

                # Validation of metadata requirements
                if fusion in ("early", "late") and (not use_metadata or tab_dim is None):
                    print(f"[skip] fold={fold_idx} fusion={fusion} requires metadata; set use_metadata:true and metadata_features in cfg.")
                    continue
                if fusion == 'early' and variant == 'plain':
                    print(f"[skip] fold={fold_idx} fusion=early with variant=plain is not applicable; skipping.")
                    continue

                # Build model per fusion/variant
                if fusion == 'imageonly':
                    model = TemporalViTClassifier(
                        pretrained_vit=True,
                        n_layers=int(cfg.get("temporal_layers", 2)),
                        n_heads=int(cfg.get("temporal_heads", 8)),
                        pool=cfg.get("temporal_pool", "cls"),
                    )
                elif fusion == 'early':
                    model = TemporalViTEarlyFusionClassifier(
                        variant=variant,
                        tab_dim=tab_dim,
                        n_layers=int(cfg.get("temporal_layers", 2)),
                        n_heads=int(cfg.get("temporal_heads", 8)),
                        pool=cfg.get("temporal_pool", "cls"),
                        c_pat=pat_channels,
                        img_size=img_size,
                        prompt_K=prompt_K,
                        cls_cross_heads=cls_cross_heads,
                    )
                elif fusion == 'late':
                    model = TemporalViTLateFusionClassifier(
                        tab_dim=tab_dim,
                        n_layers=int(cfg.get("temporal_layers", 2)),
                        n_heads=int(cfg.get("temporal_heads", 8)),
                        pool=cfg.get("temporal_pool", "cls"),
                        out_dim=2,
                        clin_hid=int(cfg.get("late_clin_hid", 64)),
                        clin_out=int(cfg.get("late_clin_out", 32)),
                    )
                else:
                    print(f"[warn] unknown fusion '{fusion}', skipping.")
                    continue

                model = model.to(device)
                # Optimizer selection
                if opt_name in ("adamw", "adam"):
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                elif opt_name in ("sgd", "sgdm"):
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                elif opt_name == "rmsprop":
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                # Train with early stopping on selected metric
                best = -1.0
                best_epoch = None
                best_state = None
                patience = int(cfg.get("early_stopping_patience", 10))
                no_improve = 0
                use_amp = args.amp == "on"
                t0 = time.time()
                best_inner_vdf = None

                # reset CUDA peak stats
                gpu_name = "cpu"; alloc_mb = None; reserved_mb = None
                if device.startswith('cuda') and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                    except Exception:
                        pass

                for epoch in range(1, epochs + 1):
                    print(f"\n[temporal-{fusion}] variant={variant} fold={fold_idx} epoch {epoch}/{epochs}")
                    train_loss = train_one_epoch_temporal(model, optimizer, tr_loader, device, use_amp=use_amp)
                    reduce_params = {
                        'topk_frac': getattr(args, 'video_topk_frac', 0.05),
                        'topk_cap': getattr(args, 'video_topk_cap', 30),
                        'smooth_win': getattr(args, 'video_smooth_win', 5),
                        'smooth_kind': getattr(args, 'video_smooth_kind', 'median'),
                        'lse_alpha': getattr(args, 'video_lse_alpha', 8.0),
                    }
                    # selection on inner validation
                    v_metrics, _, vdf = evaluate_temporal(model, inner_va_loader, device, video_reduce=args.video_reduce, reduce_params=reduce_params)
                    print("  train_loss=", round(train_loss, 4))
                    print("  val(video)=", {k: round(float(v), 4) for k, v in v_metrics.items()})
                    if args.eval_outer_each_epoch:
                        ovm, _, _ovdf = evaluate_temporal(model, outer_va_loader, device, video_reduce=args.video_reduce, reduce_params=reduce_params)
                        print("  outer(video)=", {k: round(float(v), 4) for k, v in ovm.items()})
                    score = float(v_metrics.get(args.select_metric, 0.0))
                    if score > best:
                        best = score
                        best_epoch = epoch
                        best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                        no_improve = 0
                        best_inner_vdf = vdf.copy() if isinstance(vdf, pd.DataFrame) else None
                        if args.save_checkpoints:
                            ckpt_root = os.path.join(run_root, fold_name)
                            os.makedirs(ckpt_root, exist_ok=True)
                            torch.save(
                                {
                                    "cfg": cfg,
                                    "variant": variant,
                                    "fusion": fusion,
                                    "fold": fold_idx,
                                    "model": best_state,
                                    "best_val": {k: float(v) for k, v in v_metrics.items()},
                                    "best_epoch": int(best_epoch),
                                    "video_reduce": args.video_reduce,
                                    "video_reduce_params": reduce_params,
                                    "select_metric": str(args.select_metric),
                                },
                                os.path.join(ckpt_root, "best.pt"),
                            )
                            if args.save_val_probs and isinstance(vdf, pd.DataFrame):
                                vdf.to_csv(os.path.join(ckpt_root, "val_video_probs.csv"), index=False)
                            print(f"[ckpt] new best saved @ epoch {best_epoch}: {args.select_metric}={best:.4f}")
                    else:
                        no_improve += 1
                    if no_improve >= patience:
                        print(f"[early-stop] no improvement for {patience} epochs; best {args.select_metric}={best:.4f} at epoch {best_epoch}")
                        break

                # capture GPU peaks
                if device.startswith('cuda') and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize();
                        alloc_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
                        reserved_mb = torch.cuda.max_memory_reserved() / (1024.0**2)
                    except Exception:
                        pass
                train_seconds = time.time() - t0

                # Threshold tuning on best validation predictions
                thr_video = 0.5
                tuned_video_metrics = {}
                if isinstance(best_inner_vdf, pd.DataFrame):
                    yv_in = best_inner_vdf["y_true"].values
                    pv_in = best_inner_vdf["y_prob"].values
                if args.select_metric == "auc" or args.select_metric == "ap":
                    thr_video = 0.5
                else:
                    uniq = np.unique(pv_in)
                    cand = np.linspace(0.0, 1.0, num=201) if uniq.size > 200 else np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0)
                    best_t, best_s = 0.5, -1.0
                    for t in cand:
                        m = metrics_from_probs(yv_in, pv_in, thr=float(t)).get(args.threshold_metric, float("nan"))
                        s = float(m) if np.isfinite(m) else -1.0
                        if s > best_s:
                            best_s, best_t = s, float(t)
                    thr_video = best_t
                tuned_video_metrics_inner = metrics_from_probs(yv_in, pv_in, thr=thr_video)

                # Evaluate on pristine outer validation with tuned threshold
                ov_metrics, _, ov_df = evaluate_temporal(model, outer_va_loader, device, video_reduce=args.video_reduce, reduce_params=reduce_params)
                yv = ov_df['y_true'].values; pv = ov_df['y_prob'].values
                tuned_video_metrics = metrics_from_probs(yv, pv, thr=thr_video)
                if args.save_val_probs:
                    ov_df.to_csv(os.path.join(os.path.join(run_root, fold_name), "outer_val_video_probs.csv"), index=False)

                # Save per-fold summary and checkpoint
                row = {
                    "variant": variant,
                    "fusion": fusion,
                    "fold": fold_idx,
                    "val_auc": float(best),
                    "convergence_epoch": int(best_epoch or 0),
                    "best_epoch": int(best_epoch or 0),
                    "train_seconds": float(train_seconds),
                    "select_metric": str(args.select_metric),
                    "threshold_metric": str(args.threshold_metric),
                    "thr_video": float(thr_video),
                    "device": str(device),
                    "gpu_name": str(gpu_name),
                    "cuda_mem_alloc_peak_mb": float(alloc_mb) if alloc_mb is not None else None,
                    "cuda_mem_reserved_peak_mb": float(reserved_mb) if reserved_mb is not None else None,
                    **({f"val_{k}": float(v) for k, v in tuned_video_metrics.items()} if tuned_video_metrics else {}),
                }
                fold_metrics.append(row)
                summary_rows.append(row)

                fold_out = os.path.join(run_root, fold_name)
                os.makedirs(fold_out, exist_ok=True)
                with open(os.path.join(fold_out, "fold_summary.json"), "w") as f:
                    json.dump(row, f, indent=2)
                if best_state is not None:
                    torch.save({
                        "model": best_state,
                        "cfg": cfg,
                        "variant": variant,
                        "fusion": fusion,
                        "fold": fold_idx,
                        "best_epoch": int(best_epoch or 0),
                        "thr_video": float(thr_video),
                        "select_metric": str(args.select_metric),
                        "threshold_metric": str(args.threshold_metric),
                        "video_reduce": args.video_reduce,
                        "video_reduce_params": reduce_params,
                    }, os.path.join(fold_out, "best.pt"))

            # Per-model summary after all folds
            if fold_metrics:
                dfm = pd.DataFrame(fold_metrics)
                dfm.to_csv(os.path.join(run_root, "cv_summary.csv"), index=False)
                def _fmt_hms(sec: float) -> str:
                    sec = float(max(0.0, sec))
                    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"
                lines = [
                    f"Temporal ViT CV Summary - variant={variant} fusion={fusion}",
                    f"Folds completed: {len(dfm)}",
                ]
                # print mean±std for all val_* metrics
                val_cols = [c for c in dfm.columns if c.startswith('val_')]
                for col in sorted(val_cols):
                    try:
                        mu = float(dfm[col].mean()); sd = float(dfm[col].std()); mn = float(dfm[col].min()); mx = float(dfm[col].max())
                        lines.append(f"{col}: {mu:.4f} ± {sd:.4f} (min={mn:.4f}, max={mx:.4f})")
                    except Exception:
                        pass
                # Per-fold convergence epoch lines
                for _, r in dfm.sort_values('fold').iterrows():
                    ce = int(r.get('convergence_epoch', r.get('best_epoch', 0)) or 0)
                    tsec = float(r.get('train_seconds', 0.0) or 0.0)
                    lines.append(f"fold_{int(r['fold'])}: train={_fmt_hms(tsec)}, converge_epoch={ce}")
                with open(os.path.join(run_root, "summary.txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")

    # Global summary
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(args.outroot, "cv_summary.csv"), index=False)
        lines = ["Temporal ViT CV Summary (Global)"]
        for variant in sorted(df['variant'].unique()):
            lines.append("")
            lines.append(f"Variant: {variant}")
            subv = df[df['variant'] == variant]
            for fusion in sorted(subv['fusion'].unique()):
                sub = subv[subv['fusion'] == fusion]
                lines.append(f"  Fusion: {fusion}")
                val_cols = [c for c in sub.columns if c.startswith('val_')]
                for col in sorted(val_cols):
                    try:
                        mu = float(sub[col].mean()); sd = float(sub[col].std()); mn = float(sub[col].min()); mx = float(sub[col].max())
                        lines.append(f"  {col}: {mu:.4f} ± {sd:.4f} (min={mn:.4f}, max={mx:.4f}, n={len(sub)})")
                    except Exception:
                        pass
        with open(os.path.join(args.outroot, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
