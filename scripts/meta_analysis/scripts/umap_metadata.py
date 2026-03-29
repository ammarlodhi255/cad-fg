# python meta_analysis/scripts/umap_metadata.py --level patient --supervised  --colors FEBD70,F99B9B --use_text

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.dataio.dataset import load_metadata_any, build_meta_vectors, CLIN_NUM, CLIN_CAT, build_text_embed_map


def _apply_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
    })
    palette = [
        "#A5CDE2",  # light blue
        "#2078B3",  # deep blue
        "#B2DF8A",  # light green
        "#FEBE70",  # orange
        "#FA9B9A",  # salmon
        "#CAB2D6",  # lavender
        "#8DD3C7",  # teal
    ]
    return palette


def _aggregate_patient_labels(df: pd.DataFrame, method: str = "max", mean_thresh: float = 0.5) -> pd.DataFrame:
    """Return DataFrame with one row per patient and an aggregated label.

    method: one of {max, any, min, all, mean, majority, mode}
        - max/any: patient positive if any frame is positive
        - min/all: patient positive only if all frames are positive
        - mean: patient positive if mean(frame label) >= mean_thresh
        - majority: patient positive if mean(frame label) > 0.5
        - mode: most frequent frame label per patient
    """
    g = df.groupby("patient")["label"]
    m = method.lower()
    if m in ("max", "any"):
        s = g.max()
    elif m in ("min", "all"):
        s = g.min()
    elif m == "mean":
        s = (g.mean() >= float(mean_thresh)).astype(int)
    elif m == "majority":
        s = (g.mean() > 0.5).astype(int)
    elif m == "mode":
        s = g.apply(lambda x: int(x.value_counts().idxmax()))
    else:
        raise ValueError(f"Unknown patient label method: {method}")
    out = s.reset_index().rename(columns={"label": "label"})
    return out


def _load_patient_level_matrix(index_csv: str, metadata_path: str, selected_features: dict | None = None,
                               patient_label_method: str = "max", mean_thresh: float = 0.5,
                               balance: bool = False,
                               text_features: list | None = None,
                               text_model: str | None = None,
                               text_cache_dir: str | None = None):
    df = pd.read_csv(index_csv)
    # collapse to one row per patient to avoid many near-duplicates per frame, with configurable label aggregation
    dfp = _aggregate_patient_labels(df, method=patient_label_method, mean_thresh=mean_thresh)

    meta_df = load_metadata_any(metadata_path)
    text_map = None
    if text_features:
        text_map = build_text_embed_map(meta_df, patient_col="patient",
                                        text_cols=text_features,
                                        model_name=(text_model or "sentence-transformers/all-MiniLM-L6-v2"),
                                        cache_dir=(text_cache_dir or "outputs/text_embeds"))
    meta_map = build_meta_vectors(meta_df, selected_features=selected_features, text_embed_map=text_map)

    X = []
    y = []
    pids = []
    for _, r in dfp.iterrows():
        pid = int(r["patient"])
        if pid not in meta_map:
            continue
        X.append(meta_map[pid])
        y.append(int(r["label"]))
        pids.append(pid)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    if balance and len(np.unique(y)) == 2:
        # Downsample majority class for a balanced plot
        cls0 = np.where(y == 0)[0]
        cls1 = np.where(y == 1)[0]
        if len(cls0) and len(cls1):
            k = min(len(cls0), len(cls1))
            rng = np.random.default_rng(1337)
            idx = np.concatenate([rng.choice(cls0, k, replace=False), rng.choice(cls1, k, replace=False)])
            X, y, pids = X[idx], y[idx], [pids[i] for i in idx]
    return X, y, pids


def _load_frame_level_matrix(index_csv: str, metadata_path: str, selected_features: dict | None = None,
                             sample_per_patient: int | None = None, jitter: float = 0.0,
                             balance: bool = False, seed: int = 1337,
                             text_features: list | None = None,
                             text_model: str | None = None,
                             text_cache_dir: str | None = None):
    """Frame-level matrix using per-patient metadata replicated for each frame.

    Options:
      - sample_per_patient: cap number of frames per patient to reduce heavy duplication.
      - jitter: add small Gaussian noise to metadata vectors to visually separate identical points.
      - balance: downsample majority class for clearer plots.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(index_csv)
    meta_df = load_metadata_any(metadata_path)
    text_map = None
    if text_features:
        text_map = build_text_embed_map(meta_df, patient_col="patient",
                                        text_cols=text_features,
                                        model_name=(text_model or "sentence-transformers/all-MiniLM-L6-v2"),
                                        cache_dir=(text_cache_dir or "outputs/text_embeds"))
    meta_map = build_meta_vectors(meta_df, selected_features=selected_features, text_embed_map=text_map)

    rows = []
    for pid, g in df.groupby("patient"):
        pid = int(pid)
        if pid not in meta_map:
            continue
        if sample_per_patient is not None and len(g) > sample_per_patient:
            g = g.sample(n=sample_per_patient, random_state=seed)
        for _, r in g.iterrows():
            rows.append((pid, int(r.get("label", 0))))

    X = []
    y = []
    for pid, lab in rows:
        v = meta_map[pid].astype(np.float32)
        if jitter and jitter > 0.0:
            v = v + rng.normal(0.0, float(jitter), size=v.shape).astype(np.float32)
        X.append(v)
        y.append(lab)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    if balance and len(np.unique(y)) == 2:
        cls0 = np.where(y == 0)[0]
        cls1 = np.where(y == 1)[0]
        if len(cls0) and len(cls1):
            k = min(len(cls0), len(cls1))
            idx = np.concatenate([rng.choice(cls0, k, replace=False), rng.choice(cls1, k, replace=False)])
            X, y = X[idx], y[idx]
    return X, y, None


def _plot_umap(emb, labels, title, save_path, palette_override=None):
    base_palette = _apply_plot_style()
    palette = palette_override if palette_override else base_palette
    fig, ax = plt.subplots(figsize=(8, 6))
    # Set background to requested light tone and use white grid lines
    bg = "#E8EAF3"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    classes = np.unique(labels)
    colors = (palette * ((len(classes) // len(palette)) + 1))[: len(classes)]
    def _label_name(c):
        try:
            ci = int(c)
        except Exception:
            return str(c)
        return "No-lesion" if ci == 0 else ("Lesion" if ci == 1 else str(c))

    for c, col in zip(classes, colors):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=36, edgecolor="#2b2b2b", linewidth=0.6, alpha=0.85, label=_label_name(c), color=col)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    # White grid lines over the colored background
    ax.grid(True, which="major", color="#FFFFFF", linewidth=1.0, alpha=1.0)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#FFFFFF", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Class")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="UMAP projection of metadata (patient- or frame-level)")
    parser.add_argument("--index_csv", default="outputs/index/trainval_pool.csv", help="CSV with patient, label columns")
    parser.add_argument("--metadata_path", default="data/CADICA/metadata.xlsx")
    parser.add_argument("--out_dir", default="meta_analysis/results/umap")
    parser.add_argument("--cfg", default="meta_analysis/meta_cfg.yaml", help="YAML config with metadata_features and text params")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--supervised", action="store_true", help="Use labels to guide UMAP (supervised UMAP)")
    parser.add_argument("--features", type=str, default="config", choices=["config","all", "numeric", "categorical"], help="Feature subset: from config or override")
    parser.add_argument("--level", type=str, default="patient", choices=["patient","frame"], help="Operate at patient- or frame-level")
    parser.add_argument("--patient_label", type=str, default="mean", choices=["max","any","min","all","mean","majority","mode"],
                        help="How to aggregate frame labels into a patient label")
    parser.add_argument("--mean_thresh", type=float, default=0.5, help="Threshold for --patient_label mean")
    parser.add_argument("--balance", action="store_true", help="Downsample majority class for visual balance")
    parser.add_argument("--sample_per_patient", type=int, default=50, help="Max frames per patient for frame-level plots (None for all)")
    parser.add_argument("--jitter", type=float, default=0.02, help="Std-dev of Gaussian noise added to metadata for frame-level visualization")
    parser.add_argument("--colors", type=str, default=None, help="Comma-separated hex colors (e.g., FEBD70,2078B3) to override class colors")
    parser.add_argument("--use_text", action="store_true", help="Include text features from config (requires cached or available encoder)")
    args = parser.parse_args()

    # Load config for feature selection
    try:
        with open(args.cfg, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    cfg_feats = (cfg.get("metadata_features") or {})
    cfg_num = list(cfg_feats.get("numeric", []) or [])
    cfg_cat = list(cfg_feats.get("categorical", []) or [])
    cfg_text = list(cfg_feats.get("text", []) or [])
    text_model = cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")
    text_cache_dir = cfg.get("text_cache_dir", "outputs/text_embeds")

    # Feature subset selection (default from config)
    if args.features == "config":
        selected = {"numeric": cfg_num, "categorical": cfg_cat}
    elif args.features == "all":
        selected = None
    elif args.features == "numeric":
        selected = {"numeric": list(CLIN_NUM), "categorical": []}
    else:
        selected = {"numeric": [], "categorical": list(CLIN_CAT)}

    text_cols = cfg_text if args.use_text else None

    if args.level == "patient":
        X, y, pids = _load_patient_level_matrix(
            args.index_csv,
            args.metadata_path,
            selected_features=selected,
            patient_label_method=args.patient_label,
            mean_thresh=args.mean_thresh,
            balance=args.balance,
            text_features=text_cols,
            text_model=text_model,
            text_cache_dir=text_cache_dir,
        )
        level_tag = "patient-level"
    else:
        spp = None if (args.sample_per_patient is None or int(args.sample_per_patient) <= 0) else int(args.sample_per_patient)
        X, y, pids = _load_frame_level_matrix(
            args.index_csv,
            args.metadata_path,
            selected_features=selected,
            sample_per_patient=spp,
            jitter=float(args.jitter) if args.jitter is not None else 0.0,
            balance=args.balance,
            seed=args.seed,
            text_features=text_cols,
            text_model=text_model,
            text_cache_dir=text_cache_dir,
        )
        level_tag = "frame-level"

    # Class distribution report
    uniq, cnts = np.unique(y, return_counts=True)
    print("Class counts:", {int(k): int(v) for k, v in zip(uniq, cnts)})
    if len(uniq) < 2:
        print("Warning: Only one class present after aggregation. Consider using --patient_label mean/majority/mode and a different index CSV, or disable filtering that drops negative patients.")

    # Color override parsing
    palette_override = None
    if args.colors:
        cols = [c.strip() for c in args.colors.split(",") if c.strip()]
        def norm_hex(h):
            h = h.lstrip('#').upper()
            return f"#{h}"
        if cols:
            palette_override = [norm_hex(c) for c in cols]

    # Save the matrix for reproducibility
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(out_dir / "X_meta.csv", index=False)
    pd.DataFrame({"patient": pids, "label": y}).to_csv(out_dir / "y_labels.csv", index=False)
    # Also save features actually used
    feat_spec = {"numeric": (selected.get("numeric") if isinstance(selected, dict) else "all"),
                 "categorical": (selected.get("categorical") if isinstance(selected, dict) else "all"),
                 "text": (text_cols or [])}
    with open(out_dir / "feature_spec.yaml", "w") as f:
        yaml.safe_dump(feat_spec, f)

    try:
        import umap
    except Exception as e:
        print("UMAP import failed. Please install with: pip install umap-learn")
        raise

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )

    # Unsupervised embedding
    emb_unsup = reducer.fit_transform(X)
    _plot_umap(emb_unsup, y, title=f"UMAP (unsupervised) on metadata [{level_tag}]", save_path=out_dir / f"umap_unsupervised_{args.level}", palette_override=palette_override)

    # Supervised embedding (if requested)
    if args.supervised:
        reducer_sup = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed,
        )
        emb_sup = reducer_sup.fit_transform(X, y=y)
        _plot_umap(emb_sup, y, title=f"UMAP (supervised) on metadata [{level_tag}]", save_path=out_dir / f"umap_supervised_{args.level}", palette_override=palette_override)


if __name__ == "__main__":
    main()
