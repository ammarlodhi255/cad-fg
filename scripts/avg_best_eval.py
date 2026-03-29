#!/usr/bin/env python3
import os
import sys
import re
import glob
import json
import argparse
import importlib.util
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from sklearn.calibration import calibration_curve
except Exception:
    calibration_curve = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.transforms import get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.metrics import metrics_from_probs
import cv2
import torch.nn.functional as F


def import_build_models():
    base_path = os.path.join(os.path.dirname(__file__), "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold_base", base_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader, "Failed to locate 02_train_kfold.py"
    spec.loader.exec_module(module)  # type: ignore
    return module.build_models  # type: ignore[attr-defined]


@torch.no_grad()
def evaluate_ensemble(models, loader, device, fusion_mode: str, video_reduce: str = "tsmooth_max", pos_class: int = 1):
    """Evaluate an ensemble of models.
    models: list of (img_model, clin_mlp, late_head)
    fusion_mode: 'late' | 'early' | 'imageonly'
    Returns: (frame_metrics, y_frame, p_frame, video_metrics, video_df)
    """
    for img_model, clin_mlp, late_head in models:
        img_model.eval()
        if clin_mlp: clin_mlp.eval()
        if late_head: late_head.eval()

    probs, ys = [], []
    patients, videos, frames = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        clin = batch["clin"].to(device)

        # accumulate probabilities over ensemble
        p_sum = None
        for img_model, clin_mlp, late_head in models:
            if fusion_mode == "late":
                feats, _ = img_model(x)
                logits = late_head(feats, clin_mlp(clin))
            elif fusion_mode == "early":
                _, logits = img_model(x, clin)
            elif fusion_mode == "imageonly":
                _, logits = img_model(x)
            else:
                raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
            pr = torch.softmax(logits, 1)
            p_sum = pr if p_sum is None else (p_sum + pr)
        pr_avg = p_sum / float(max(1, len(models)))
        pc = int(pos_class) if pos_class in (0, 1) else 1
        p = pr_avg[:, pc].detach().cpu().numpy()

        probs.append(p); ys.append(y)
        patients.append(np.asarray(batch["patient"]))
        videos.append(np.asarray(batch["video"]))
        frames.append(np.asarray(batch.get("frame", np.arange(len(y)))))

    y_frame = np.concatenate(ys)
    p_frame = np.concatenate(probs)

    frame_metrics = metrics_from_probs(y_frame, p_frame, thr=0.5)

    # ----- video-level aggregation -----
    df = pd.DataFrame({
        "patient": np.concatenate(patients),
        "video":   np.concatenate(videos),
        "frame":   np.concatenate(frames),
        "y_true":  y_frame,
        "y_prob":  p_frame,
    })

    def _agg_true(g):
        return float(np.max(g["y_true"].values))

    reducers = {
        "max": lambda g: float(np.max(g["y_prob"].values)),
        "mean": lambda g: float(np.mean(g["y_prob"].values)),
        "median": lambda g: float(np.median(g["y_prob"].values)),
        "topk2": lambda g: float(np.mean(np.sort(g["y_prob"].values)[-min(2, len(g)):])),
        "topk3": lambda g: float(np.mean(np.sort(g["y_prob"].values)[-min(3, len(g)):])),
        "logit_mean": lambda g: float(1.0/(1.0+np.exp(-np.mean(np.log(np.clip(g["y_prob"].values,1e-6,1-1e-6)/np.clip(1-g["y_prob"].values,1e-6,1-1e-6)))))),
        "ew_mean": lambda g: float(np.average(g["y_prob"].values, weights=np.abs(g["y_prob"].values-0.5)*2 + 1e-6)),
        "tsmooth_mean": lambda g: float(np.mean(pd.Series(g.sort_values("frame")["y_prob"]).rolling(window=5, min_periods=1, center=True).mean().values)),
        "tsmooth_max": lambda g: float(np.max(pd.Series(g.sort_values("frame")["y_prob"]).rolling(window=5, min_periods=1, center=True).mean().values)),
    }

    dfs = {}
    metrics_map = {}
    for name, fn in reducers.items():
        vals = []
        trues = []
        for (pid, vid), g in df.groupby(["patient", "video"], sort=False):
            trues.append(_agg_true(g))
            try:
                vals.append(fn(g))
            except Exception:
                vals.append(float(np.mean(g["y_prob"].values)))
        vdf = pd.DataFrame({"y_true": trues, f"y_prob_{name}": vals})
        dfs[name] = vdf
        metrics_map[name] = metrics_from_probs(vdf["y_true"].values, vdf[f"y_prob_{name}"].values, thr=0.5)

    sel = video_reduce if video_reduce in reducers else "tsmooth_max"
    video_df = dfs[sel].rename(columns={f"y_prob_{sel}": "y_prob"}).copy()
    # Merge additional columns for convenience
    for name, vdf in dfs.items():
        if name == sel:
            continue
        video_df = video_df.merge(vdf[[f"y_prob_{name}"]], left_index=True, right_index=True, how="left")

    video_metrics = metrics_map[sel]
    video_metrics["_multi"] = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics_map.items()}

    return frame_metrics, y_frame, p_frame, video_metrics, video_df


def write_summary_txt(path: str, model_label: str, df: pd.DataFrame, final_test: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # compute mean/std for val_* columns even though this is a single row
    stats = {}
    for col in sorted([c for c in df.columns if c.startswith("val_")]):
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "n": int(s.shape[0]),
        }
    lines = []
    lines.append(f"Model: {model_label}")
    lines.append(f"Folds completed: {df.shape[0]}")
    for col in sorted(stats.keys()):
        st = stats[col]
        lines.append(f"{col}: {st['mean']:.4f} ± {st['std']:.4f} (min={st['min']:.4f}, max={st['max']:.4f}, n={st['n']})")
    if final_test is not None:
        lines.append("")
        lines.append("Final Test (lockbox)")
        for dom in ("frame","video"):
            md = final_test.get(dom, {})
            for k in sorted(md.keys()):
                lines.append(f"test_{dom}_{k}: {float(md[k]):.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_figures(out_dir: str, y_frame, p_frame, frame_metrics, video_df, video_metrics):
    PALETTE = ["#2078B3", "#FA9B9A", "#B2DF8A", "#FEBE70", "#FA9B9A"]
    os.makedirs(out_dir, exist_ok=True)
    if plt is None:
        print("[warn] matplotlib not available; skipping figures")
        return
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

    # ROC (frame)
    try:
        fpr, tpr, _ = roc_curve(y_frame, p_frame)
        plt.figure(figsize=(5,4), dpi=150)
        plt.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"ROC (AUC={float(frame_metrics.get('auc', 0.0)):.3f})")
        plt.plot([0,1],[0,1], color="#888", ls="--", lw=1)
        plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)"); plt.title("ROC (fold 3 outer val) ")
        plt.grid(alpha=0.25); plt.legend(loc="lower right"); plt.tight_layout()
        for ext in ("png","pdf","svg"):
            plt.savefig(os.path.join(out_dir, f"roc_test.{ext}"))
        plt.close()
    except Exception as _e:
        print(f"[warn] ROC plot failed: {_e}")

    # PR (frame)
    try:
        prec, rec, _ = precision_recall_curve(y_frame, p_frame)
        plt.figure(figsize=(5,4), dpi=150)
        plt.plot(rec, prec, color=PALETTE[3], lw=2, label=f"PR (AP={float(frame_metrics.get('ap', 0.0)):.3f})")
        plt.xlabel("Recall (Sensitivity)"); plt.ylabel("Precision (PPV)"); plt.title("PR (fold 3 outer val) ")
        plt.grid(alpha=0.25); plt.legend(loc="lower left"); plt.tight_layout()
        for ext in ("png","pdf","svg"):
            plt.savefig(os.path.join(out_dir, f"pr_test.{ext}"))
        plt.close()
    except Exception as _e:
        print(f"[warn] PR plot failed: {_e}")

    # Confusion at 0.5
    try:
        yhat = (p_frame >= 0.5).astype(int)
        cm = confusion_matrix(y_frame, yhat, labels=[0,1])
        plt.figure(figsize=(4.5,4.0), dpi=150)
        plt.imshow(cm, cmap="Blues")
        plt.xticks([0,1],["Pred 0","Pred 1"]); plt.yticks([0,1],["True 0","True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(int(cm[i,j])), ha='center', va='center', color="#111", fontsize=12)
        plt.title("Confusion Matrix (thr=0.5)"); plt.tight_layout()
        for ext in ("png","pdf","svg"):
            plt.savefig(os.path.join(out_dir, f"confusion_test.{ext}"))
        plt.close()
    except Exception as _e:
        print(f"[warn] Conf matrix plot failed: {_e}")

    # Sens–Spec (frame)
    try:
        uniq = np.unique(p_frame)
        cand = np.unique(np.clip(np.concatenate(([0.0], uniq, [1.0])), 0.0, 1.0))
        sens_l, spec_l = [], []
        for t in cand:
            md = metrics_from_probs(y_frame, p_frame, thr=float(t))
            sens_l.append(float(md.get("sensitivity", 0.0)))
            spec_l.append(float(md.get("specificity", 0.0)))
        plt.figure(figsize=(5,4), dpi=150)
        plt.plot(spec_l, sens_l, color=PALETTE[2], lw=2)
        plt.xlabel("Specificity"); plt.ylabel("Sensitivity"); plt.title("Sensitivity–Specificity (fold 3 outer val) ")
        plt.grid(alpha=0.25); plt.tight_layout()
        for ext in ("png","pdf","svg"):
            plt.savefig(os.path.join(out_dir, f"sens_spec_test.{ext}"))
        plt.close()
    except Exception as _e:
        print(f"[warn] Sens–Spec plot failed: {_e}")

    # Video reducer comparison bar chart
    try:
        vmulti = video_metrics.get("_multi") if isinstance(video_metrics, dict) else None
        if isinstance(vmulti, dict):
            reducers = sorted(list(vmulti.keys()))
            aucs = [float(vmulti[r].get("auc", float("nan"))) for r in reducers]
            aps  = [float(vmulti[r].get("ap", float("nan"))) for r in reducers]
            bals = [float(vmulti[r].get("bal_acc", float("nan"))) for r in reducers]
            width = 0.25
            x = np.arange(len(reducers))
            plt.figure(figsize=(8,4), dpi=150)
            plt.bar(x - width, aucs, width, color=PALETTE[0], label="AUC")
            plt.bar(x,         aps,  width, color=PALETTE[3], label="AP")
            plt.bar(x + width, bals, width, color=PALETTE[2], label="BalAcc")
            plt.xticks(x, reducers, rotation=30, ha='right')
            plt.ylabel("Score"); plt.title("Reducers comparison (video)")
            plt.grid(axis='y', alpha=0.25)
            plt.legend(); plt.tight_layout()
            for ext in ("png","pdf","svg"):
                plt.savefig(os.path.join(out_dir, f"reducers_compare_video.{ext}"))
            plt.close()
    except Exception as _e:
        print(f"[warn] Reducers compare (video) failed: {_e}")


def _find_last_4d_module(model: nn.Module, x: torch.Tensor, clin: torch.Tensor | None, fusion_mode: str):
    """
    Run a dry forward and return the last module that outputs a 4D tensor
    with spatial size > 1x1 (skip global pooling like AdaptiveAvgPool2d).
    This yields meaningful Grad-CAM maps for YOLOv5/YOLOv8 classifiers and
    mid-fusion variants (e.g., *_cls_xattn_mid), where the prepool can be 1x1.
    """
    hits = []
    hooks = []
    def _hook(mod, inp, out):
        try:
            t = out
            if torch.is_tensor(t) and t.dim() == 4:
                H, W = int(t.shape[-2]), int(t.shape[-1])
                if H > 1 and W > 1:
                    hits.append(mod)
        except Exception:
            pass
        return None
    for m in model.modules():
        try:
            hooks.append(m.register_forward_hook(_hook))
        except Exception:
            continue
    try:
        model.eval()
        with torch.no_grad():
            if fusion_mode == "late":
                feats, _ = model(x)
            elif fusion_mode == "early":
                _ = model(x, clin)
            else:
                _ = model(x)
    finally:
        for h in hooks:
            h.remove()
    return hits[-1] if hits else None


def generate_gradcam_samples(out_dir: str, model_triplet, loader, device, fusion_mode: str, pos_class: int = 1, per_class: int = 8):
    """Generate Grad-CAM overlays for a few positive/negative samples using the first model.
    Saves PNG overlays under out_dir/gradcam.
    """
    img_model, clin_mlp, late_head = model_triplet
    img_model.eval();  
    grad_root = os.path.join(out_dir, "gradcam")
    grad_dir_cam_pos = os.path.join(grad_root, "overlay", "pos")
    grad_dir_cam_neg = os.path.join(grad_root, "overlay", "neg")
    grad_dir_org_pos = os.path.join(grad_root, "original", "pos")
    grad_dir_org_neg = os.path.join(grad_root, "original", "neg")
    for d in (grad_dir_cam_pos, grad_dir_cam_neg, grad_dir_org_pos, grad_dir_org_neg):
        os.makedirs(d, exist_ok=True)

    pos_need = int(per_class)
    neg_need = int(per_class)
    got_pos = 0
    got_neg = 0

    for batch in loader:
        if got_pos >= pos_need and got_neg >= neg_need:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        clin = batch["clin"].to(device)
        paths = batch["img_path"]
        B = x.size(0)

        # find target layer lazily (once)
        target_layer = _find_last_4d_module(img_model, x[:1], clin[:1] if clin is not None else None, fusion_mode)
        if target_layer is None:
            print("[gradcam] could not locate a 4D feature layer; skipping")
            return

        activations = []
        gradients = []

        def fwd_hook(_m, _inp, out):
            activations.append(out)
            try:
                out.register_hook(lambda g: gradients.append(g))
            except Exception:
                pass
            return None

        h = target_layer.register_forward_hook(fwd_hook)
        try:
            x.requires_grad_(True)
            # forward
            if fusion_mode == "late":
                feats, _ = img_model(x)
                logits = late_head(feats, clin_mlp(clin)) if clin_mlp is not None else feats
            elif fusion_mode == "early":
                _, logits = img_model(x, clin)
            else:
                _, logits = img_model(x)
            pc = int(pos_class) if pos_class in (0,1) else 1
            # Sum across batch for a single backward pass targeting pos_class
            score = logits[:, pc].sum()
            img_model.zero_grad();
            if clin_mlp is not None: clin_mlp.zero_grad()
            if late_head is not None: late_head.zero_grad()
            score.backward()

            if not activations or not gradients:
                print("[gradcam] hooks did not capture activations/gradients; skipping batch")
                continue
            act = activations[-1]      # [B,C,H,W]
            grad = gradients[-1]       # [B,C,H,W]
            weights = grad.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
            cam = (act * weights).sum(dim=1, keepdim=True).clamp(min=0)  # [B,1,H,W]
            cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            cam = cam.detach().cpu().numpy()  # [B,1,H,W]

            # save per sample until quotas are met
            for i in range(B):
                yi = int(y[i].item())
                if yi == 1 and got_pos >= pos_need:
                    continue
                if yi == 0 and got_neg >= neg_need:
                    continue
                heat = cam[i, 0]
                heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                heat_uint8 = np.uint8(255 * heat)
                heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
                # load original image for overlay
                ipath = paths[i]
                img0 = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
                if img0 is None:
                    continue
                # Resize CAM to original image size for proper overlay
                if heat_color.shape[0] != img0.shape[0] or heat_color.shape[1] != img0.shape[1]:
                    heat_color = cv2.resize(heat_color, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_LINEAR)
                img3 = cv2.merge([img0, img0, img0])
                try:
                    overlay = cv2.addWeighted(img3, 0.5, heat_color, 0.5, 0)
                except Exception:
                    # Fallback: if overlay fails, just use heat_color
                    overlay = heat_color
                base_name = os.path.basename(ipath)

                # Save original with real filename under original/pos|neg
                if yi == 1:
                    org_dir = grad_dir_org_pos
                    cam_dir = grad_dir_cam_pos
                else:
                    org_dir = grad_dir_org_neg
                    cam_dir = grad_dir_cam_neg
                try:
                    cv2.imwrite(os.path.join(org_dir, base_name), img0)
                except Exception:
                    pass

                # Save overlay using the same base filename for easy mapping
                cv2.imwrite(os.path.join(cam_dir, base_name), overlay)
                if yi == 1:
                    got_pos += 1
                else:
                    got_neg += 1
                if got_pos >= pos_need and got_neg >= neg_need:
                    break
        finally:
            h.remove()
    print(f"[gradcam] saved overlays: pos={got_pos}, neg={got_neg} -> {os.path.join(out_dir, 'gradcam')}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate an ensemble of best.pt checkpoints on pristine test; save figures and summary")
    ap.add_argument("--ckpts", nargs="+", required=True, help="Paths or directories (or globs) to best.pt files (>=1)")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--test_csv", default="outputs/index/test_pristine.csv")
    ap.add_argument("--outroot", required=True, help="Directory to write outputs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--video_reduce", default="tsmooth_max", choices=["max","mean","median","topk2","topk3","logit_mean","ew_mean","tsmooth_mean","tsmooth_max"])
    ap.add_argument("--pos_class", type=int, default=1, choices=[0,1], help="Which class index to treat as positive when computing metrics (default: 1)")
    ap.add_argument("--fusion", type=str, default=None, choices=["late","early","imageonly"], help="Override fusion mode to interpret checkpoints (e.g., imageonly for image-only weights). If omitted, inferred from checkpoint or path.")
    ap.add_argument("--gradcam_per_class", type=int, default=100, help="Number of Grad-CAM samples to save per class (pos/neg). Set 0 to disable.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load cfg
    cfg = json.load(open(args.cfg, "r")) if args.cfg.endswith(".json") else None
    if cfg is None:
        import yaml
        cfg = yaml.safe_load(open(args.cfg))

    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    sel = cfg.get("metadata_features", {}) or {}
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)
    img_size = int(cfg.get("img_size", 224))

    # dataset
    test_ds = CadicaDataset(
        args.test_csv, metadata, get_val_tf(img_size),
        selected_features=sel_nc, text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    te = torch.utils.data.DataLoader(test_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=4)

    # build models
    build_models = import_build_models()
    # resolve checkpoint paths (accept files, dirs, or globs)
    def _resolve_ckpt(p: str) -> list[str]:
        out = []
        # direct file first to avoid infinite recursion when no wildcards
        if os.path.isfile(p):
            return [p]
        # if directory, look for best.pt inside
        if os.path.isdir(p):
            cand = os.path.join(p, "best.pt")
            if os.path.isfile(cand):
                return [cand]
            # otherwise any *.pt
            any_pt = sorted(glob.glob(os.path.join(p, "*.pt")))
            return any_pt
        # expand glob only if wildcards present
        if any(ch in p for ch in ("*", "?", "[", "]")):
            matches = glob.glob(p)
            for m in matches:
                out.extend(_resolve_ckpt(m))
            return out
        # try to fix common fold naming: fold0 -> fold_0
        m = re.search(r"/(fold)(\d)(/|$)", p)
        if m:
            fixed = p.replace(f"{m.group(1)}{m.group(2)}", f"{m.group(1)}_{m.group(2)}")
            if os.path.isfile(fixed):
                return [fixed]
            if os.path.isdir(fixed):
                cand = os.path.join(fixed, "best.pt")
                if os.path.isfile(cand):
                    return [cand]
        return out

    ckpt_files: list[str] = []
    for p in args.ckpts:
        ckpt_files.extend(_resolve_ckpt(p))
    # de-dup and verify
    ckpt_files = sorted(set([os.path.abspath(x) for x in ckpt_files if os.path.isfile(x)]))
    if not ckpt_files:
        raise SystemExit("No valid checkpoints resolved from --ckpts. Check paths or use globs.")

    print("[ensemble] using checkpoints:")
    for x in ckpt_files:
        print("  ", x)

    ensemble = []
    fusion_mode = None
    for cpath in ckpt_files:
        ck = torch.load(cpath, map_location="cpu")
        backbone = ck.get("backbone", None)
        # precedence: CLI --fusion > checkpoint key > infer from path > default
        fusion = args.fusion or ck.get("fusion", None)
        # fallback: infer from path like .../<backbone>_<fusion>/fold_*/best.pt
        if backbone is None or fusion is None:
            parts = os.path.normpath(cpath).split(os.sep)
            # find a segment that looks like backbone_fusion (with one underscore at the end)
            for seg in parts[::-1]:
                if "_" in seg and not seg.endswith(".pt"):
                    b, f = seg.rsplit("_", 1)
                    if f in ("late","early","imageonly"):
                        backbone = backbone or b
                        fusion = fusion or f
                        break
        # ensure fusion is set
        fusion = fusion or "early"
        fusion_mode = fusion or fusion_mode or "early"
        img_model, clin_mlp, late_head, params = build_models(
            backbone=backbone, fusion=fusion, tab_dim=test_ds.tab_dim,
            pat_channels=cfg.get("pat_channels", 4), device=device,
        )
        # load weights
        try:
            img_model.load_state_dict(ck["model"], strict=False)
            if clin_mlp and ck.get("clin_mlp") is not None:
                clin_mlp.load_state_dict(ck["clin_mlp"], strict=False)
            if late_head and ck.get("late_head") is not None:
                late_head.load_state_dict(ck["late_head"], strict=False)
        except Exception as e:
            print(f"[warn] failed to load some weights from {cpath}: {e}")
        ensemble.append((img_model, clin_mlp, late_head))

    # evaluate each model individually and print metrics
    def _flat_video(m):
        return {k: float(v) for k, v in (m.items() if isinstance(m, dict) else []) if k != "_multi"}

    per_model_rows = []
    for i, (img_model, clin_mlp, late_head) in enumerate(ensemble):
        fmi, yfi, pfi, vmi, vdfi = evaluate_ensemble([(img_model, clin_mlp, late_head)], te, device, fusion_mode, video_reduce=args.video_reduce, pos_class=args.pos_class)
        vf = _flat_video(vmi)
        row = {"index": i, "ckpt": ckpt_files[i], **{f"frame_{k}": float(v) for k, v in fmi.items()}, **{f"video_{k}": v for k, v in vf.items()}}
        per_model_rows.append(row)
        # pretty print
        print(f"\n[ckpt {i}] {ckpt_files[i]}")
        print("  frame:", {k: round(float(v), 4) for k, v in fmi.items()})
        print("  video:", {k: round(float(v), 4) for k, v in vf.items()})

    # eval ensemble
    fmet, y_frame, p_frame, vmet, vdf = evaluate_ensemble(ensemble, te, device, fusion_mode, video_reduce=args.video_reduce, pos_class=args.pos_class)

    # write outputs
    out_dir = args.outroot
    os.makedirs(out_dir, exist_ok=True)
    # metrics json (flatten video, exclude _multi)
    video_flat = {k: float(v) for k, v in vmet.items() if k != "_multi"}
    final_test_metrics = {"frame": {k: float(v) for k, v in fmet.items()}, "video": video_flat}
    with open(os.path.join(out_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    # figures
    figs_dir = os.path.join(out_dir, "figs")
    save_figures(figs_dir, y_frame, p_frame, fmet, vdf, vmet)

    # Grad-CAM: use the first model in the ensemble for visualization
    try:
        if len(ensemble) > 0 and int(args.gradcam_per_class) > 0:
            generate_gradcam_samples(out_dir, ensemble[0], te, device, fusion_mode, pos_class=args.pos_class, per_class=int(args.gradcam_per_class))
    except Exception as _e:
        print(f"[warn] grad-cam generation failed: {_e}")

    # summary.txt
    df_row = {"fold": 0}
    for k, v in fmet.items():
        df_row[f"val_frame_{k}"] = float(v)
    for k, v in video_flat.items():
        df_row[f"val_video_{k}"] = float(v)
    df_one = pd.DataFrame([df_row])
    write_summary_txt(os.path.join(out_dir, "summary.txt"), model_label=f"ensemble({len(ensemble)})/{fusion_mode}", df=df_one, final_test=final_test_metrics)

    # save per-model metrics table
    try:
        pm_df = pd.DataFrame(per_model_rows)
        pm_df.to_csv(os.path.join(out_dir, "per_model_metrics.csv"), index=False)
        with open(os.path.join(out_dir, "per_model_metrics.json"), "w") as f:
            json.dump({"rows": per_model_rows}, f, indent=2)
    except Exception as _e:
        print(f"[warn] failed to write per-model metrics: {_e}")

    # print ensemble metrics
    print("\n[ensemble] frame:", {k: round(float(v), 4) for k, v in fmet.items()})
    print("[ensemble] video:", {k: round(float(v), 4) for k, v in video_flat.items()})


if __name__ == "__main__":
    main()
