#!/usr/bin/env python3
import os, sys, yaml, math, argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader

# data & models
from src.dataio.transforms import get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import evaluate  # not used here but keeps parity

import matplotlib.pyplot as plt

# ------------------------------ Helpers ------------------------------

def load_ckpt(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    fusion = ck["fusion"]
    tab_dim = ck["tab_dim"]
    cfg = ck["cfg"]
    backbone = cfg.get("backbone", "resnet50").lower()
    return ck, fusion, tab_dim, cfg, backbone

def build_test_loader(cfg, tab_dim_from_ckpt=None):
    """
    Recreate the test dataset with the SAME metadata subset (incl. text) used at train time.
    """
    te_csv = "outputs/index/test.csv"
    metadata_path = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")

    # numeric/categorical subset for vector building
    sel = cfg.get("metadata_features", {}) or {}
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}

    # explicit record saved at training time has priority (if present)
    mf_used = cfg.get("metadata_features_used", None)
    text_cols = (mf_used.get("text") if mf_used is not None else sel.get("text"))

    ds = CadicaDataset(
        te_csv,
        metadata_path,
        get_val_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    return ds, ld

def rebuild_model_from_cfg(cfg, fusion, tab_dim, device):
    """
    Rebuilds backbone + (optionally) clin MLP + fusion head
    to match the checkpoint fusion mode/backbone.
    Returns (model, clin_mlp, head, img_feat_dim).
    """
    backbone = cfg.get("backbone", "resnet50").lower()

    # ---------- LATE ----------
    if fusion == "late":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "effnet_b3":
            from src.models.effnet_b3_backbone import EfficientNetB3Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone as BB
            model = BB(variant="tiny", pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone as BB
            model = BB(variant="base", pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "densenet201":
            from src.models.densenet_backbone import DenseNet201Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7Backbone as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone as BB
            model = BB(pretrained=False, cbam=True).to(device); img_feat_dim = model.feat_dim
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM as BB
            model = BB(pretrained=False).to(device); img_feat_dim = model.feat_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        return model, clin_mlp, head, img_feat_dim

    # ---------- EARLY ----------
    elif fusion == "early":
        if backbone == "resnet50":
            from src.models.early_fusion_patient import ResNetEarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            from src.models.effnet_b3_backbone import EfficientNetB3EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient as EF
            model = EF(variant="tiny", tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient as EF
            model = EF(variant="base", tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3LargeEarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet201":
            from src.models.densenet_backbone import DenseNet201EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetEarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTEarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0EarlyFusionPatient as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2, cbam=True).to(device)
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseEarlyFusionPatientCBAM as EF
            model = EF(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        return model, None, None, getattr(model, "feat_dim", None)

    # ---------- IMAGE-ONLY ----------
    elif fusion == "imageonly":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b3":
            from src.models.effnet_b3_backbone import EfficientNetB3Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_tiny":
            from src.models.convnext_backbone import ConvNeXtBackbone as BB
            model = BB(variant="tiny", pretrained=False, out_dim=2).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone as BB
            model = BB(variant="base", pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v3_large":
            from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "mobilenet_v2":
            from src.models.mobilenetv2_backbone import MobileNetV2Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet201":
            from src.models.densenet_backbone import DenseNet201Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "densenet121":
            from src.models.densenet121_backbone import DenseNet121Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "alexnet":
            from src.models.alexnet_backbone import AlexNetBackbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "vit_b16":
            from src.models.vit_backbone import ViTBackbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b7":
            from src.models.effnet_b7_backbone import EfficientNetB7Backbone as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0_cbam":
            from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone as BB
            model = BB(pretrained=False, out_dim=2, cbam=True).to(device)
        elif backbone == "convnext_base_cbam":
            from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM as BB
            model = BB(pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        return model, None, None, getattr(model, "feat_dim", None)

    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")

def extract_embeddings_and_preds(model, head, loader, device, fusion, clin_mlp=None):
    """
    Returns:
      X [N,D], y [N], preds [N], pids [N], meta_rows [dict per item]
    meta_rows include: patient, video, frame, img_path plus any patient-level
    columns from dataset.meta_df (looked up by patient id).
    """
    model.eval()
    if clin_mlp: clin_mlp.eval()
    if head: head.eval()

    # Prepare patient->meta row lookup from the dataset attached to the loader
    ds = loader.dataset
    meta_df = getattr(ds, "meta_df", None)
    patient_meta = {}
    if meta_df is not None and "patient" in meta_df.columns:
        # use first row per patient
        pm = meta_df.groupby("patient").first().reset_index()
        patient_meta = {int(r.patient): r.to_dict() for _, r in pm.iterrows()}

    feats_all, y_all, pid_all, pred_all, meta_all = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            pids = batch["patient"].cpu().numpy()
            clin = batch["clin"].to(device)

            # ---- forward to get embeddings + logits ----
            if fusion == "late":
                f, _ = model(x)         # CNN feats
                z = clin_mlp(clin)      # tabular embed
                f_cat = torch.cat([f, z], dim=1)
                logits = head(f_cat)
            elif fusion == "early":
                f, logits = model(x, clin)
                f_cat = f
            elif fusion == "imageonly":
                f, logits = model(x)
                f_cat = f
            else:
                raise ValueError("fusion must be 'late','early','imageonly'")

            feats_all.append(f_cat.cpu().numpy())
            y_all.append(y)
            pid_all.append(pids)
            pred_all.append(torch.argmax(logits, dim=1).cpu().numpy())

            # assemble per-item metadata for coloring
            vids = batch["video"].cpu().numpy() if "video" in batch else np.zeros_like(pids)
            frms = batch["frame"].cpu().numpy() if "frame" in batch else np.zeros_like(pids)
            paths = batch.get("img_path", [""] * len(pids))
            if isinstance(paths, torch.Tensor):  # should not happen, but be safe
                paths = [str(p) for p in paths]

            for i in range(len(pids)):
                meta_row = {
                    "patient": int(pids[i]),
                    "video": int(vids[i]),
                    "frame": int(frms[i]),
                    "img_path": paths[i],
                }
                # enrich with patient-level meta (age, diabetes, etc.) if available
                pm = patient_meta.get(int(pids[i]), None)
                if pm is not None:
                    meta_row.update(pm)
                meta_all.append(meta_row)

    X = np.concatenate(feats_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    preds = np.concatenate(pred_all, axis=0)
    pids = np.concatenate(pid_all, axis=0)
    return X, y, preds, pids, meta_all

def reduce_2d(X, method="umap", random_state=42):
    method = method.lower()
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=random_state)
            return reducer.fit_transform(X)
        except Exception as e:
            print("[warn] UMAP not available; falling back to t-SNE:", e)
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=random_state).fit_transform(X)

    raise ValueError("method must be 'umap' or 'tsne'")

def _is_categorical(values):
    """Heuristic: object dtype or few unique values -> categorical"""
    values = np.asarray(values)
    if values.dtype.kind in ("U","S","O"):  # strings/objects
        return True
    uniq = np.unique(values)
    return len(uniq) <= 12  # small palette -> treat as categories

def plot_scatter(X2, values, title, out_path, is_categorical=None):
    if is_categorical is None:
        is_categorical = _is_categorical(values)

    plt.figure(figsize=(8,6))
    if is_categorical:
        unique_vals = pd.unique(values)
        cmap = plt.cm.get_cmap("tab10", max(2, len(unique_vals)))
        colors = {v: cmap(i % cmap.N) for i, v in enumerate(unique_vals)}
        c = [colors[v] for v in values]
        plt.scatter(X2[:,0], X2[:,1], c=c, alpha=0.80, edgecolors="none")
        handles = [plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=colors[v], label=str(v), markersize=8)
                   for v in unique_vals]
        plt.legend(handles=handles, title="Category", loc="best", frameon=True)
    else:
        sc = plt.scatter(X2[:,0], X2[:,1], c=values, cmap="viridis", alpha=0.85, edgecolors="none")
        cb = plt.colorbar(sc); cb.set_label("Value")

    plt.title(title)
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint")
    ap.add_argument("--method", default="umap", choices=["umap","tsne"], help="dimensionality reduction")
    ap.add_argument("--level", default="both", choices=["frame","patient","both"], help="granularity to visualize")
    ap.add_argument("--video_reduce", default="max", choices=["max","mean"], help="reserved for future video-level")
    ap.add_argument("--rand", type=int, default=42)
    ap.add_argument(
        "--color_by", default="label",
        help="Color points by 'label', 'pred', or any metadata column name "
             "(e.g. patient, video, frame, age, diabetes, another_comorb, indication, later_events, ...)"
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck, fusion, tab_dim, cfg, backbone = load_ckpt(args.ckpt, device="cpu")

    # dataset/loader
    ds, ld = build_test_loader(cfg)

    # model & heads
    model, clin_mlp, head, img_feat_dim = rebuild_model_from_cfg(cfg, fusion, tab_dim, device)
    # load weights
    model.load_state_dict(ck["model"])
    if fusion == "late":
        assert ck["clin_mlp"] is not None and ck["head"] is not None, \
            "Checkpoint missing late-fusion heads."
        clin_mlp.load_state_dict(ck["clin_mlp"])
        head.load_state_dict(ck["head"])

    # ---- FRAME-LEVEL EMBEDDINGS + PREDICTIONS ----
    X_frame, y_frame, preds_frame, pids_frame, meta_rows = extract_embeddings_and_preds(
        model, head, ld, device, fusion, clin_mlp
    )

    # choose coloring values (frame-level)
    color_source = args.color_by.strip().lower()
    if color_source == "label":
        color_vals_frame = y_frame
    elif color_source == "pred":
        color_vals_frame = preds_frame
    else:
        # try metadata column from meta_rows; if missing, raise
        if len(meta_rows) == 0 or (color_source not in meta_rows[0]):
            # try raw name as-is (without lowercasing)
            if len(meta_rows) and (args.color_by in meta_rows[0]):
                color_vals_frame = np.array([mr[args.color_by] for mr in meta_rows])
            else:
                raise ValueError(
                    f"Unknown color_by '{args.color_by}'. "
                    f"Available keys include: {sorted(list(meta_rows[0].keys())) if meta_rows else '[]'}"
                )
        else:
            color_vals_frame = np.array([mr[color_source] for mr in meta_rows])

    out_dir = "outputs/embeddings"
    os.makedirs(out_dir, exist_ok=True)

    # ---- FRAME-LEVEL PLOT ----
    if args.level in ("frame","both"):
        X2 = reduce_2d(X_frame, method=args.method, random_state=args.rand)
        plot_scatter(
            X2, color_vals_frame,
            title=f"{backbone} | {fusion} | frame-level ({args.method.upper()}) colored by {args.color_by}",
            out_path=os.path.join(out_dir, f"{backbone}_{fusion}_frame_{args.method}_{args.color_by}.png"),
        )
        print(f"[done] frame-level {args.method} plot -> {os.path.join(out_dir, f'{backbone}_{fusion}_frame_{args.method}_{args.color_by}.png')}")

    # ---- PATIENT-LEVEL (average embeddings across frames per patient) ----
    if args.level in ("patient","both"):
        df = pd.DataFrame(X_frame)
        df["pid"] = pids_frame
        df["y"] = y_frame
        df["pred"] = preds_frame

        # Embed per-patient = mean of frame embeddings
        X_patient = df.groupby("pid").mean().drop(columns=["y","pred"]).values

        # Patient label: max over frames (positive if any positive frame)
        y_patient = df.groupby("pid")["y"].max().values
        pred_patient = df.groupby("pid")["pred"].max().values

        # Choose patient-level coloring
        if color_source == "label":
            color_vals_patient = y_patient
        elif color_source == "pred":
            color_vals_patient = pred_patient
        else:
            # metadata column: use first available per patient from meta_rows
            # Build patient->value mapping from meta_rows (constant per patient)
            meta_df = pd.DataFrame(meta_rows)
            # prefer exact key match (original arg), else lowered key
            key = args.color_by if args.color_by in meta_df.columns else color_source
            if key not in meta_df.columns:
                raise ValueError(f"[patient] color_by '{args.color_by}' not found in metadata columns: {list(meta_df.columns)}")
            # take first occurrence per patient
            pmeta = meta_df.groupby("patient").agg("first").reset_index().set_index("patient")
            # align to the order of patients in X_patient
            patients_sorted = df.groupby("pid").size().index.values
            color_vals_patient = pmeta.loc[patients_sorted, key].values

        X2p = reduce_2d(X_patient, method=args.method, random_state=args.rand)
        plot_scatter(
            X2p, color_vals_patient,
            title=f"{backbone} | {fusion} | patient-level ({args.method.upper()}) colored by {args.color_by}",
            out_path=os.path.join(out_dir, f"{backbone}_{fusion}_patient_{args.method}_{args.color_by}.png"),
        )
        print(f"[done] patient-level {args.method} plot -> {os.path.join(out_dir, f'{backbone}_{fusion}_patient_{args.method}_{args.color_by}.png')}")

if __name__ == "__main__":
    main()