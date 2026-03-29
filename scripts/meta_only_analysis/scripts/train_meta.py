import os, sys
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from torch.utils.data import Subset
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, DeepLift, GradientShap, FeatureAblation
import yaml
import json
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.dataio.dataset import load_metadata_any, build_meta_vectors, CLIN_NUM, CLIN_CAT, CAT_VALS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from meta_analysis.src.meta_model import MetadataFusionMLP, MetadataResidualMLP
from meta_analysis.src.text_encoder import process_text_features

class MetadataOnlyDataset(Dataset):
    def __init__(self, csv_path, metadata_path, selected_features=None, text_encoder=None):
        self.df = pd.read_csv(csv_path)
        self.meta_df = load_metadata_any(metadata_path)
        self.meta = build_meta_vectors(self.meta_df, selected_features=selected_features)
        self.selected_features = selected_features
        
        # Get feature dimension from first patient
        any_pid = int(self.df.iloc[0]["patient"])
        self.tab_dim = len(self.meta[any_pid])
        
        # Track which num/cat features are used
        if isinstance(self.selected_features, dict):
            self.used_num = [k for k in self.selected_features.get("numeric", CLIN_NUM) if k in CLIN_NUM]
            self.used_cat = [k for k in self.selected_features.get("categorical", CLIN_CAT) if k in CLIN_CAT]
        else:
            self.used_num = list(CLIN_NUM)
            self.used_cat = list(CLIN_CAT)

        # Setup text features if provided
        self.text_encoder = text_encoder
        self.text_fields = selected_features.get("text", []) if selected_features else []
        if self.text_encoder and self.text_fields:
            # Pre-compute text embeddings for each patient
            self.text_embeddings = {}
            for pid, rows in self.meta_df.groupby("patient"):
                row = rows.iloc[0]
                text_dict = {field: str(row.get(field, "")) for field in self.text_fields}
                emb = process_text_features(text_dict, self.text_encoder)
                # Ensure 1D vector [D]
                if emb.ndim > 1:
                    emb = emb.view(-1)
                self.text_embeddings[int(pid)] = emb.to(torch.float32)
            
            # Get text embedding dimension (vector length)
            any_emb = next(iter(self.text_embeddings.values()))
            self.text_dim = int(any_emb.numel())
            # Per-field embedding dim (assume equal per field from encoder)
            self.text_embed_dim = getattr(self.text_encoder, "embed_dim", None)
            if not self.text_embed_dim and len(self.text_fields) > 0:
                # Fallback if encoder doesn't expose dim
                self.text_embed_dim = self.text_dim // len(self.text_fields)
        else:
            self.text_dim = 0
            self.text_embed_dim = 0
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = int(r.patient)
        
        # Get tabular features
        x_tab = torch.tensor(self.meta[pid], dtype=torch.float32)
        
        # Get text features if available
        if self.text_encoder and self.text_fields:
            x_text = self.text_embeddings[pid]
        else:
            x_text = torch.zeros(self.text_dim, dtype=torch.float32)
        
        y = torch.tensor(int(r.label), dtype=torch.long)
        
        return {
            "x_tab": x_tab,
            "x_text": x_text,
            "y": y,
            "patient": pid,
            "video": int(r.video),
            "frame": int(r.frame)
        }

    def get_feature_names(self):
        """Return expanded feature names (numeric + one-hot categorical) matching vector order."""
        names = list(self.used_num)
        for k in self.used_cat:
            base = CAT_VALS.get(k, ["no", "yes"])
            names.extend([f"{k}={b}" for b in base])
        return names

    def get_text_field_slices(self):
        """Return list of (field_name, start, end) slices over the concatenated text embedding."""
        if not (self.text_fields and self.text_dim > 0 and self.text_embed_dim):
            return []
        slices = []
        for i, field in enumerate(self.text_fields):
            start = i * self.text_embed_dim
            end = start + self.text_embed_dim
            slices.append((field, start, end))
        return slices

    def get_feature_groups(self):
        return {"numeric": list(self.used_num), "categorical": list(self.used_cat)}

def _get_underlying_dataset(dataset):
    """Get the underlying dataset if wrapped in Subset."""
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset

def compute_feature_importance(model, loader, device, method: str = "ig"):
    """Compute importance over full validation set and group categoricals per base feature.

    Returns a Series with numeric names, categorical base names, and text:<field> entries.
    """
    model.eval()
    # Get underlying dataset (handle Subset wrapper)
    underlying_ds = _get_underlying_dataset(loader.dataset)
    has_text = getattr(underlying_ds, "text_dim", 0) > 0

    def forward_pair(xt, tt):
        return model(xt, tt)

    def forward_tab_only(xt):
        tt = xt.new_zeros((xt.size(0), 0))
        return model(xt, tt)

    method = (method or "ig").lower()
    if method == "ig":
        attr_obj = IntegratedGradients(forward_pair if has_text else forward_tab_only)
        def attr_fn(xt, tt, y):
            return attr_obj.attribute((xt, tt), target=y) if has_text else attr_obj.attribute(xt, target=y)
    elif method == "deeplift":
        attr_obj = DeepLift(forward_pair if has_text else forward_tab_only)
        def attr_fn(xt, tt, y):
            return attr_obj.attribute((xt, tt), target=y) if has_text else attr_obj.attribute(xt, target=y)
    elif method == "gradshap":
        attr_obj = GradientShap(forward_pair if has_text else forward_tab_only)
        def attr_fn(xt, tt, y):
            if has_text:
                baselines = (xt*0, tt*0)
                return attr_obj.attribute((xt, tt), target=y, baselines=baselines)
            else:
                baselines = xt*0
                return attr_obj.attribute(xt, target=y, baselines=baselines)
    elif method == "ablation":
        attr_obj = FeatureAblation(forward_pair if has_text else forward_tab_only)
        def attr_fn(xt, tt, y):
            return attr_obj.attribute((xt, tt), target=y) if has_text else attr_obj.attribute(xt, target=y)
    else:
        attr_obj = IntegratedGradients(forward_pair if has_text else forward_tab_only)
        def attr_fn(xt, tt, y):
            return attr_obj.attribute((xt, tt), target=y) if has_text else attr_obj.attribute(xt, target=y)

    feat_names = underlying_ds.get_feature_names()
    feat_dim = len(feat_names)
    tab_sum = torch.zeros(feat_dim, dtype=torch.float64)
    text_sum = None
    n_total = 0

    for batch in loader:
        xt = batch["x_tab"].to(device)
        tt = batch["x_text"].to(device)
        yb = batch["y"].to(device)
        attributions = attr_fn(xt, tt, yb)
        attr_tab = None; attr_text = None
        if isinstance(attributions, (list, tuple)):
            if len(attributions) >= 1:
                attr_tab = attributions[0]
            if len(attributions) >= 2 and attributions[1] is not None:
                attr_text = attributions[1]
        else:
            attr_tab = attributions

        if attr_tab is not None:
            tab_sum += attr_tab.abs().sum(0).detach().cpu().to(torch.float64)
        if has_text and attr_text is not None:
            if text_sum is None:
                text_sum = torch.zeros(attr_text.shape[-1], dtype=torch.float64)
            text_sum += attr_text.abs().sum(0).detach().cpu().to(torch.float64)
        n_total += xt.size(0)

    tab_mean = (tab_sum / max(n_total, 1)).numpy()
    text_mean = (text_sum / max(n_total, 1)).numpy() if text_sum is not None else None

    # Grouping
    groups = underlying_ds.get_feature_groups() if hasattr(underlying_ds, 'get_feature_groups') else {"numeric": [], "categorical": []}
    out = {}
    # Numeric direct
    for name in groups.get("numeric", []):
        if name in feat_names:
            out[name] = float(tab_mean[feat_names.index(name)])
    # Categorical aggregate
    for cat in groups.get("categorical", []):
        s = 0.0; found = False
        pref = f"{cat}="
        for i, n in enumerate(feat_names):
            if n.startswith(pref):
                s += float(tab_mean[i]); found = True
        if found:
            out[cat] = s
    # Any remaining numeric-like columns not in groups
    for i, n in enumerate(feat_names):
        if "=" not in n and n not in out:
            out[n] = float(tab_mean[i])
    # Text aggregated
    if text_mean is not None and hasattr(underlying_ds, 'get_text_field_slices'):
        for field, s, e in (underlying_ds.get_text_field_slices() or []):
            out[f"text:{field}"] = float(np.sum(text_mean[s:e]))

    return pd.Series(out).sort_values(ascending=False)


def _collect_val_arrays(val_loader, device):
    xs_tab = []
    xs_txt = []
    ys = []
    pats = []
    vids = []
    for batch in val_loader:
        xs_tab.append(batch["x_tab"])  # keep on CPU for permutation ops
        xs_txt.append(batch["x_text"])
        ys.append(batch["y"]) 
        pats.append(batch["patient"])
        vids.append(batch["video"])
    x_tab = torch.cat(xs_tab, dim=0).to(device)
    x_text = torch.cat(xs_txt, dim=0).to(device) if xs_txt and xs_txt[0].numel() > 0 else torch.zeros((len(torch.cat(ys)), 0), device=device)
    y = torch.cat(ys, dim=0).cpu().numpy()
    patient = torch.cat(pats, dim=0).cpu().numpy()
    video = torch.cat(vids, dim=0).cpu().numpy()
    return x_tab, x_text, y, patient, video


def _metrics_from_arrays(model, x_tab, x_text, y, patient, video, device):
    model.eval()
    with torch.no_grad():
        logits = model(x_tab, x_text)
        p = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
    y_pred_f = (p >= 0.5).astype(int)
    frame_metrics = {
        "acc": accuracy_score(y, y_pred_f),
        "bal_acc": balanced_accuracy_score(y, y_pred_f),
        "precision": precision_score(y, y_pred_f, zero_division=0),
        "recall": recall_score(y, y_pred_f, zero_division=0),
        "f1": f1_score(y, y_pred_f, zero_division=0),
        "auc": roc_auc_score(y, p) if len(np.unique(y))==2 else np.nan,
    }
    df = pd.DataFrame({"patient": patient, "video": video, "y_true": y, "y_prob": p})
    agg = df.groupby(["patient","video"]).agg(y_true=("y_true","max"), y_prob=("y_prob","max")).reset_index()
    y_pred_v = (agg.y_prob.values >= 0.5).astype(int)
    video_metrics = {
        "acc": accuracy_score(agg.y_true.values, y_pred_v),
        "bal_acc": balanced_accuracy_score(agg.y_true.values, y_pred_v),
        "precision": precision_score(agg.y_true.values, y_pred_v, zero_division=0),
        "recall": recall_score(agg.y_true.values, y_pred_v, zero_division=0),
        "f1": f1_score(agg.y_true.values, y_pred_v, zero_division=0),
        "auc": roc_auc_score(agg.y_true.values, agg.y_prob.values) if len(np.unique(agg.y_true.values))==2 else np.nan,
    }
    return frame_metrics, video_metrics


def compute_permutation_importance(model, val_loader, device, metric: str = "frame_auc", repeats: int = 1, seed: int = 1337):
    """Permutation importance by feature GROUP (numeric singletons, categorical one-hot groups, text fields).

    metric: which frame-level metric to use for scoring drops; one of
        {"frame_auc", "frame_bal_acc", "frame_acc"}
    """
    # Collect arrays
    x_tab, x_text, y, patient, video = _collect_val_arrays(val_loader, device)

    # Baseline
    base_frame, base_video = _metrics_from_arrays(model, x_tab, x_text, y, patient, video, device)
    if metric == "frame_auc":
        base = float(base_frame.get("auc", np.nan))
    elif metric == "frame_bal_acc":
        base = float(base_frame.get("bal_acc", 0.0))
    else:
        base = float(base_frame.get("acc", 0.0))

    # Build tabular index groups - get underlying dataset (handle Subset wrapper)
    underlying_ds = _get_underlying_dataset(val_loader.dataset)
    feat_names = underlying_ds.get_feature_names()
    name_to_idx = {n:i for i,n in enumerate(feat_names)}
    groups = []
    # numeric
    for n in underlying_ds.get_feature_groups().get("numeric", []):
        if n in name_to_idx:
            groups.append((n, [name_to_idx[n]], "tab"))
    # categorical groups
    for cat in underlying_ds.get_feature_groups().get("categorical", []):
        idxs = [i for i,n in enumerate(feat_names) if n.startswith(f"{cat}=")]
        if idxs:
            groups.append((cat, idxs, "tab"))
    # text groups
    for field, s, e in getattr(underlying_ds, "get_text_field_slices", lambda:[])() or []:
        groups.append((f"text:{field}", (s, e), "text"))

    rng = np.random.default_rng(seed)
    scores = {}

    for name, idxs, kind in groups:
        vals = []
        for _ in range(max(1, int(repeats))):
            if kind == "tab":
                x_tab_perm = x_tab.clone()
                perm = torch.from_numpy(rng.permutation(x_tab_perm.size(0))).to(x_tab_perm.device)
                x_tab_perm[:, idxs] = x_tab_perm[perm][:, idxs]
                frame_m, _ = _metrics_from_arrays(model, x_tab_perm, x_text, y, patient, video, device)
            else:
                x_text_perm = x_text.clone()
                s, e = idxs
                if x_text_perm.numel() == 0 or e <= s:
                    continue
                perm = torch.from_numpy(rng.permutation(x_text_perm.size(0))).to(x_text_perm.device)
                x_text_perm[:, s:e] = x_text_perm[perm][:, s:e]
                frame_m, _ = _metrics_from_arrays(model, x_tab, x_text_perm, y, patient, video, device)
            if metric == "frame_auc":
                val = float(frame_m.get("auc", np.nan))
            elif metric == "frame_bal_acc":
                val = float(frame_m.get("bal_acc", 0.0))
            else:
                val = float(frame_m.get("acc", 0.0))
            vals.append(val)
        # If AUC is NaN (e.g., single-class), treat drop as zero by using base
        vals_clean = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        mean_val = float(np.mean(vals_clean)) if vals_clean else base
        # For AUC, larger is better; drop = base - mean_val
        scores[name] = float(max(0.0, base - mean_val))

    return pd.Series(scores).sort_values(ascending=False)

def _compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute sensitivity (recall for positive class) and specificity (recall for negative class)."""
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return float(sens), float(spec)


def evaluate(model, loader, device, compute_importance=False):
    model.eval()
    probs, ys = [], []
    patients, videos = [], []
    
    # Get feature importance if requested
    importance = None
    if compute_importance:
        try:
            importance = compute_feature_importance(model, loader, device)
        except Exception as e:
            print(f"Warning: Could not compute feature importance: {e}")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x_tab = batch["x_tab"].to(device)
            x_text = batch["x_text"].to(device)
            y = batch["y"].cpu().numpy()
            logits = model(x_tab, x_text)
            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            
            probs.append(p)
            ys.append(y)
            patients.append(batch["patient"].numpy())
            videos.append(batch["video"].numpy())
    
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    
    # Frame-level metrics
    y_pred_f = (p >= 0.5).astype(int)
    sens_f, spec_f = _compute_sensitivity_specificity(y, y_pred_f)
    frame_metrics = {
        "acc": accuracy_score(y, y_pred_f),
        "bal_acc": balanced_accuracy_score(y, y_pred_f),
        "precision": precision_score(y, y_pred_f, zero_division=0),
        "recall": recall_score(y, y_pred_f, zero_division=0),
        "sensitivity": sens_f,
        "specificity": spec_f,
        "f1": f1_score(y, y_pred_f, zero_division=0),
        "auc": roc_auc_score(y, p)
    }
    
    # Video-level metrics (max pooling)
    df = pd.DataFrame({
        "patient": np.concatenate(patients),
        "video": np.concatenate(videos),
        "y_true": y,
        "y_prob": p
    })
    
    agg = df.groupby(["patient", "video"]).agg(
        y_true=("y_true", "max"),
        y_prob=("y_prob", "max")
    ).reset_index()
    
    y_pred_v = (agg.y_prob.values >= 0.5).astype(int)
    sens_v, spec_v = _compute_sensitivity_specificity(agg.y_true.values, y_pred_v)
    video_metrics = {
        "acc": accuracy_score(agg.y_true.values, y_pred_v),
        "bal_acc": balanced_accuracy_score(agg.y_true.values, y_pred_v),
        "precision": precision_score(agg.y_true.values, y_pred_v, zero_division=0),
        "recall": recall_score(agg.y_true.values, y_pred_v, zero_division=0),
        "sensitivity": sens_v,
        "specificity": spec_v,
        "f1": f1_score(agg.y_true.values, y_pred_v, zero_division=0),
        "auc": roc_auc_score(agg.y_true.values, agg.y_prob.values)
    }
    
    return frame_metrics, video_metrics, agg, importance

def _apply_plot_style():
    """Apply figure style consistent with fig_comorbidities_bars.py and return palette."""
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

def plot_feature_importance(importance: pd.Series, epoch: int, save_dir: Path, top_k: int = 20, filename_stem: str | None = None):
    """Plot and save feature importance as horizontal bars with consistent styling.

    - Normalizes to percentages and annotates values outside bars.
    - Shows top_k most important features to keep the figure readable.
    """
    palette = _apply_plot_style()

    # Normalize to percent
    imp = importance.copy()
    total = imp.abs().sum()
    if total > 0:
        imp = (imp.abs() / total) * 100.0

    # Sort descending and keep top_k
    imp = imp.sort_values(ascending=False)
    if top_k and len(imp) > top_k:
        imp = imp.iloc[:top_k]

    labels = imp.index.tolist()
    values = imp.values.tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(labels))
    colors = palette[: len(labels)]
    bars = ax.barh(y, values, color=colors, edgecolor="#2b2b2b", linewidth=1.2)

    ax.set_xlabel("Importance (%)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="major", linewidth=0.8, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, axis="x", which="minor", linewidth=0.4, alpha=0.15)
    ax.set_axisbelow(True)
    max_v = max(values) if len(values) else 1.0
    ax.set_xlim(0, max_v * 1.20)

    for rect, val in zip(bars, values):
        text_x = rect.get_width() + max_v * 0.01
        ax.text(text_x, rect.get_y() + rect.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=13)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.35, right=0.96, top=0.92, bottom=0.10)
    ax.set_title(f"Feature Importance (Epoch {epoch+1})")

    save_dir.mkdir(parents=True, exist_ok=True)
    if filename_stem:
        png = save_dir / f"{filename_stem}.png"
        pdf = save_dir / f"{filename_stem}.pdf"
        svg = save_dir / f"{filename_stem}.svg"
    else:
        png = save_dir / f"feature_importance_epoch_{epoch+1}.png"
        pdf = save_dir / f"feature_importance_epoch_{epoch+1}.pdf"
        svg = save_dir / f"feature_importance_epoch_{epoch+1}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)

    # Save raw values (both absolute and percent)
    raw_path = save_dir / (f"{filename_stem}.csv" if filename_stem else f"feature_importance_epoch_{epoch+1}.csv")
    pd.DataFrame({
        "feature": importance.index,
        "raw": importance.values,
        "percent": ((importance.abs() / (importance.abs().sum() or 1.0)) * 100.0).values,
    }).to_csv(raw_path, index=False)

def main():
    # Parse lightweight CLI override for selection metric
    parser = argparse.ArgumentParser(description="Train metadata-only models", add_help=False)
    parser.add_argument(
        "--select_metric",
        default=None,
        help="Metric to select best model. Aliases: bal-acc->frame_bal_acc, acc->frame_acc, auc->frame_auc",
    )
    parser.add_argument(
        "--attr_method",
        default=None,
        help="Attribution method: ig, deeplift, gradshap, ablation",
    )
    parser.add_argument(
        "--perm_repeats",
        type=int,
        default=None,
        help="Number of repeats for permutation importance (default 5)",
    )
    parser.add_argument(
        "--perm_metric",
        default=None,
        help="Metric for permutation importance: frame_auc (default), frame_bal_acc, or frame_acc",
    )
    parser.add_argument(
        "--kfold_root",
        default=None,
        help="Path to folds root (e.g., outputs/folds/patient_stratified_5fold). If set and exists, run cross-validation.",
    )
    parser.add_argument(
        "--cv_outroot",
        default="meta_analysis/results_cv",
        help="Output root for cross-validation runs (when --kfold_root is used)",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip training; aggregate existing per-fold outputs under --cv_outroot into summaries and final feature importance figures.",
    )
    cli_args, _ = parser.parse_known_args()

    # Load config
    with open("/cluster/home/ammaa/Downloads/meta-cadica/scripts/meta_analysis/meta_cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 1e-3)
    patience = cfg.get("early_stopping_patience", 10)
    # Determine selection metric (CLI overrides config). Default: frame_bal_acc
    def _canon_metric(s: str | None) -> str | None:
        if not s:
            return None
        m = str(s).strip().lower().replace(" ", "_")
        alias = {
            "bal-acc": "frame_bal_acc",
            "bal_acc": "frame_bal_acc",
            "balanced_accuracy": "frame_bal_acc",
            "acc": "frame_acc",
            "auc": "frame_auc",
            "video-bal-acc": "video_bal_acc",
            "video_bal_acc": "video_bal_acc",
            "video-acc": "video_acc",
            "video_auc": "video_auc",
        }
        return alias.get(m, m)

    selection_metric = _canon_metric(cli_args.select_metric) or _canon_metric(cfg.get("selection_metric")) or "frame_bal_acc"
    attr_method = (cli_args.attr_method or cfg.get("attr_method") or "ig").lower()
    perm_repeats = int(cli_args.perm_repeats or cfg.get("perm_repeats", 5))
    perm_metric = (cli_args.perm_metric or cfg.get("perm_metric") or "frame_auc").lower()
    
    # Data paths
    metadata_path = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    train_csv = "outputs/index/single_split/train.csv"
    val_csv = "outputs/index/single_split/val.csv"
    
    # Initialize text encoder if text features are specified
    text_encoder = None
    if cfg.get("metadata_features", {}).get("text"):
        from meta_analysis.src.text_encoder import TextFeatureEncoder
        text_encoder = TextFeatureEncoder(
            model_name=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds")
        )
    
    # Get feature selection from config
    selected_features = cfg.get("metadata_features", {})
    if not selected_features:
        selected_features = {
            "numeric": ["age", "bmi", "height", "weight", "evol_dm_years", "lvef_pct", "n_vessels", "max_involvement_pct"],
            "categorical": ["sex", "diabetes", "smoker", "hypertension", "dyslipidemia", "kidney_failure", "heart_failure", "atrial_fib"]
        }
    
    def _evaluate_ckpt_on_outer_val(ckpt_path: Path, val_csv_path: str, device, metadata_path: str, selected_features: dict, text_encoder):
        """Load checkpoint and evaluate on outer validation set."""
        if not ckpt_path.exists():
            print(f"[_evaluate_ckpt] Checkpoint not found: {ckpt_path}")
            return None
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_name = ckpt.get("model_name", "residual")
        ckpt_features = ckpt.get("selected_features", selected_features)
        
        # Create dataset for outer validation
        val_ds = MetadataOnlyDataset(val_csv_path, metadata_path, ckpt_features, text_encoder)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Create model
        if model_name == "mlp":
            model = MetadataFusionMLP(
                tabular_dim=val_ds.tab_dim,
                text_dim=val_ds.text_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.3
            )
        elif model_name == "residual":
            model = MetadataResidualMLP(
                tabular_dim=val_ds.tab_dim,
                text_dim=val_ds.text_dim,
                hidden_dim=128,
                n_blocks=3,
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Load state dict
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()
        
        # Evaluate on outer validation
        frame_metrics, video_metrics, agg, _ = evaluate(model, val_loader, device, compute_importance=False)
        
        print(f"[_evaluate_ckpt] {model_name} on outer validation: frame_bal_acc={frame_metrics.get('bal_acc', 0):.4f}, video_bal_acc={video_metrics.get('bal_acc', 0):.4f}")
        
        return {
            "model": model_name,
            "best_epoch": ckpt.get("epoch", None),
            "sel_metric": "frame_bal_acc",
            "best_score": float(ckpt.get("frame_metrics", {}).get("bal_acc", 0.0)),
            "frame_metrics": frame_metrics,
            "video_metrics": video_metrics,
        }
    
    def _train_once(train_csv_path: str, val_csv_path: str, out_root: Path):
        out_root.mkdir(parents=True, exist_ok=True)

        # Read train CSV for splitting
        train_df = pd.read_csv(train_csv_path)
        
        # Split train.csv into inner train/val by patient (like 02_train_kfold.py)
        inner_val_frac = cfg.get("inner_val_frac", 0.3)
        seed_val = cfg.get("seed", 1337)
        
        # Patient-level stratified split if possible
        pl_tr = train_df.groupby("patient")["label"].max().reset_index().rename(columns={"label": "patient_label"})
        y_p = pl_tr["patient_label"].astype(int).values
        if len(np.unique(y_p)) >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed_val)
            tr_pidx, va_pidx = next(sss.split(np.zeros(len(pl_tr)), y_p))
            tr_pats = set(pl_tr.iloc[tr_pidx]["patient"].tolist())
            va_pats = set(pl_tr.iloc[va_pidx]["patient"].tolist())
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed_val)
            tr_idx_tmp, va_idx_tmp = next(gss.split(train_df, groups=train_df["patient"]))
            tr_pats = set(train_df.iloc[tr_idx_tmp]["patient"].unique().tolist())
            va_pats = set(train_df.iloc[va_idx_tmp]["patient"].unique().tolist())
        
        inner_tr_idx = np.where(train_df["patient"].isin(tr_pats))[0]
        inner_va_idx = np.where(train_df["patient"].isin(va_pats))[0]
        
        print(f"\n[Inner split] inner_train: {len(inner_tr_idx)} samples, inner_val: {len(inner_va_idx)} samples")
        
        # Create full dataset from train_csv (for inner split)
        base_train_ds = MetadataOnlyDataset(train_csv_path, metadata_path, selected_features, text_encoder)
        # Outer validation dataset
        outer_val_ds = MetadataOnlyDataset(val_csv_path, metadata_path, selected_features, text_encoder)
        
        # Create inner train/val subsets
        inner_train_ds = Subset(base_train_ds, inner_tr_idx)
        inner_val_ds = Subset(base_train_ds, inner_va_idx)

        print("\nFeature configuration requested (from config):")
        print("Numeric:", selected_features.get("numeric", []))
        print("Categorical:", selected_features.get("categorical", []))
        if selected_features.get("text"):
            print("Text fields:", selected_features.get("text", []))
        # What is actually used after dataset defaults/validation
        try:
            used_groups = base_train_ds.get_feature_groups()
            used_names = base_train_ds.get_feature_names()
            print("\nResolved features used (after defaults):")
            print("Numeric used:", used_groups.get("numeric", []))
            print("Categorical used:", used_groups.get("categorical", []))
            print(f"Expanded feature count (tabular): {len(used_names)}")
        except Exception:
            pass
        print("\nFeature dimensions:")
        print(f"Tabular features: {base_train_ds.tab_dim}")
        print(f"Text features: {base_train_ds.text_dim}")

        # Use inner train/val for training and selection
        train_loader = DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True)
        inner_val_loader = DataLoader(inner_val_ds, batch_size=batch_size)
        # Outer validation loader (for final evaluation after training)
        outer_val_loader = DataLoader(outer_val_ds, batch_size=batch_size)

        print(f"Feature dimension (tabular only): {base_train_ds.tab_dim}")

        # Model variants
        models = {
            "mlp": MetadataFusionMLP(
                tabular_dim=base_train_ds.tab_dim,
                text_dim=base_train_ds.text_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.3
            ),
            # "residual": MetadataResidualMLP(
            #     tabular_dim=base_train_ds.tab_dim,
            #     text_dim=base_train_ds.text_dim,
            #     hidden_dim=128,
            #     n_blocks=3,
            #     dropout=0.3
            # )
        }

        os.makedirs("meta_analysis/checkpoints", exist_ok=True)

        fold_rows = []

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # Use non-weighted CrossEntropyLoss
            criterion = torch.nn.CrossEntropyLoss()

            # Always use frame_bal_acc for selection (as per user request)
            selection_metric_actual = "frame_bal_acc"
            direction = "max"
            best_score = -float("inf")
            epochs_no_improve = 0
            best_state = None
            best_frame_metrics = None
            best_video_metrics = None
            best_val_loss_metric = None
            best_video_val_loss_metric = None
            best_epoch = None

            # Per-epoch history for curves
            history = []  # list of dicts: epoch, train_loss, val_loss, frame_bal_acc, frame_auc, video_bal_acc, video_auc

            # Accumulators for final aggregated feature importances
            ig_accum = None; ig_count = 0
            perm_accum = None; perm_count = 0

            pbar = trange(epochs, desc=f"Training {name}")
            for epoch in pbar:
                # Train
                model.train()
                train_loss = 0
                n_batches = 0
            
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
                    x_tab = batch["x_tab"].to(device)
                    x_text = batch["x_text"].to(device)
                    y = batch["y"].to(device)

                    optimizer.zero_grad()
                    logits = model(x_tab, x_text)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    n_batches += 1
            
                avg_loss = train_loss / n_batches
            
            # Validate on inner validation set and compute feature importance every 5 epochs
                compute_importance = (epoch + 1) % 5 == 0
                frame_metrics, video_metrics, agg, importance = evaluate(
                    model, inner_val_loader, device, compute_importance=False
                )
                # Compute attribution importance on schedule (separate from evaluate)
                if compute_importance:
                    try:
                        importance = compute_feature_importance(model, inner_val_loader, device, method=attr_method)
                    except Exception as e:
                        print(f"Warning: Could not compute feature importance ({attr_method}): {e}")
                        importance = None

            # Validation loss on inner validation (for selection if desired)
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for vbatch in inner_val_loader:
                        vx_tab = vbatch["x_tab"].to(device)
                        vx_text = vbatch["x_text"].to(device)
                        vy = vbatch["y"].to(device)
                        vlogits = model(vx_tab, vx_text)
                        vloss = criterion(vlogits, vy)
                        val_loss += vloss.item()
                        n_val += 1
                val_loss = val_loss / max(n_val, 1)

                # Video-level validation loss (BCE on aggregated probabilities)
                # Use aggregated per-video probabilities from `agg`
                eps = 1e-7
                yv = agg["y_true"].values.astype(np.float32)
                pv = agg["y_prob"].values.astype(np.float32)
                pv = np.clip(pv, eps, 1.0 - eps)
                video_val_loss = float(-(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean())
            
            # Save feature importance if computed
                if importance is not None:
                    importance_dir = out_root / name / "feature_importance" / attr_method
                    importance_dir.mkdir(parents=True, exist_ok=True)
                    plot_feature_importance(importance, epoch, importance_dir)
                    # Normalize to percent and accumulate for final aggregation
                    imp_norm = importance.abs() / max(importance.abs().sum(), 1e-9)
                    if ig_accum is None:
                        ig_accum = imp_norm.copy()
                    else:
                        ig_accum = ig_accum.add(imp_norm, fill_value=0.0)
                    ig_count += 1

            # Also compute permutation importance on the same schedule
                if compute_importance:
                    # Compare using configured permutation metric (default AUC)
                    perm_series = compute_permutation_importance(model, inner_val_loader, device, metric=perm_metric, repeats=perm_repeats)
                    perm_dir = out_root / name / "feature_importance" / "permutation"
                    perm_dir.mkdir(parents=True, exist_ok=True)
                    plot_feature_importance(perm_series, epoch, perm_dir)
                    # Normalize to percent and accumulate for final aggregation
                    perm_norm = perm_series.abs() / max(perm_series.abs().sum(), 1e-9)
                    if perm_accum is None:
                        perm_accum = perm_norm.copy()
                    else:
                        perm_accum = perm_accum.add(perm_norm, fill_value=0.0)
                    perm_count += 1

                    # Save side-by-side CSV comparing IG (or selected method) and permutation
                    if importance is not None:
                        compare = pd.DataFrame({
                            "ig": importance,
                            "permutation": perm_series,
                        })
                        # Ranks (1 = highest)
                        compare["ig_rank"] = compare["ig"].rank(ascending=False, method="min")
                        compare["perm_rank"] = compare["permutation"].rank(ascending=False, method="min")
                        compare.sort_values(["ig_rank","perm_rank"], inplace=True)
                        compare.to_csv(perm_dir / f"ig_vs_permutation_epoch_{epoch+1}.csv")
            
                # Update progress bar
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'acc': f"{frame_metrics['acc']:.4f}",
                    'bal_acc': f"{frame_metrics['bal_acc']:.4f}",
                    'vid_acc': f"{video_metrics['acc']:.4f}",
                    'vid_bal_acc': f"{video_metrics['bal_acc']:.4f}",
                    'vid_val_loss': f"{video_val_loss:.4f}"
                })
            
                # Log history row
                history.append({
                    "epoch": int(epoch + 1),
                    "train_loss": float(avg_loss),
                    "val_loss": float(val_loss),
                    "frame_acc": float(frame_metrics.get("acc", np.nan)),
                    "frame_bal_acc": float(frame_metrics.get("bal_acc", np.nan)),
                    "frame_precision": float(frame_metrics.get("precision", np.nan)),
                    "frame_recall": float(frame_metrics.get("recall", np.nan)),
                    "frame_f1": float(frame_metrics.get("f1", np.nan)),
                    "frame_auc": float(frame_metrics.get("auc", np.nan)),
                    "video_bal_acc": float(video_metrics.get("bal_acc", np.nan)),
                    "video_acc": float(video_metrics.get("acc", np.nan)),
                    "video_precision": float(video_metrics.get("precision", np.nan)),
                    "video_recall": float(video_metrics.get("recall", np.nan)),
                    "video_f1": float(video_metrics.get("f1", np.nan)),
                    "video_auc": float(video_metrics.get("auc", np.nan)),
                    "video_val_loss": float(video_val_loss),
                })

            # Select best model based on frame_bal_acc (always, as per user request)
                current = float(frame_metrics.get("bal_acc", 0.0))
                improved = current > best_score
                if improved:
                    best_score = current
                    best_state = model.state_dict().copy()
                    print(f"\n[ckpt] New best frame_bal_acc (inner_val): {best_score:.4f}")
                    epochs_no_improve = 0
                    best_frame_metrics = dict(frame_metrics)
                    best_video_metrics = dict(video_metrics)
                    best_val_loss_metric = float(val_loss)
                    best_video_val_loss_metric = float(video_val_loss)
                    best_epoch = int(epoch + 1)
                    
                    # Save checkpoint
                    ckpt = {
                        "model_name": name,
                        "state_dict": best_state,
                        "frame_metrics": frame_metrics,
                        "video_metrics": video_metrics,
                        "val_loss": val_loss,
                        "video_val_loss": video_val_loss,
                        "feature_dim": base_train_ds.tab_dim,
                        "selected_features": selected_features,
                        "epoch": epoch,
                    }
                    ckpt_path = out_root / name / "best_model.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(ckpt, ckpt_path)
                else:
                    epochs_no_improve += 1
            
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break
            
            # After training: restore best model and evaluate on outer validation
            # Always evaluate on outer validation, regardless of whether we have a best_state
            if best_state is not None:
                model.load_state_dict(best_state)
                print(f"\nEvaluating best model (epoch {best_epoch}, inner_val frame_bal_acc={best_score:.4f}) on outer validation...")
                best_epoch_to_report = best_epoch
            else:
                print("\nWarning: No best model found, using final epoch model for outer validation")
                best_epoch_to_report = epochs if epochs > 0 else 1
            
            # Always evaluate on outer validation (using best model if available, otherwise final model)
            outer_frame_metrics, outer_video_metrics, outer_agg, _ = evaluate(
                model, outer_val_loader, device, compute_importance=False
            )
            
            print("[Outer validation/frame]", {k: round(float(v), 4) for k, v in outer_frame_metrics.items()})
            print("[Outer validation/video]", {k: round(float(v), 4) for k, v in outer_video_metrics.items()})
            
            # Update metrics to outer validation metrics for reporting
            best_frame_metrics = dict(outer_frame_metrics)
            best_video_metrics = dict(outer_video_metrics)
            best_epoch = best_epoch_to_report
        
        # Save curves and history
            model_dir = out_root / name
            model_dir.mkdir(parents=True, exist_ok=True)
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(model_dir / "training_history.csv", index=False)

        def plot_loss_curves(df: pd.DataFrame, out_dir: Path):
            palette = _apply_plot_style()
            epochs_x = df["epoch"].tolist()
            tr = df["train_loss"].tolist()
            vl = df["val_loss"].tolist()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs_x, tr, label="Train loss", color=palette[1], linewidth=2.0)
            ax.plot(epochs_x, vl, label="Val loss", color=palette[4], linewidth=2.0)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
            ax.grid(True, which="major", linewidth=0.8, alpha=0.25)
            ax.minorticks_on()
            ax.grid(True, which="minor", linewidth=0.4, alpha=0.15)
            ax.set_axisbelow(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / "loss_curves.pdf")
            fig.savefig(out_dir / "loss_curves.svg")
            plt.close(fig)

            plot_loss_curves(hist_df, model_dir)

            # Final aggregated feature importance bar charts
            feat_out_dir = out_root / name / "feature_importance"
            feat_out_dir.mkdir(parents=True, exist_ok=True)
            ig_dir = feat_out_dir / attr_method
            ig_dir.mkdir(parents=True, exist_ok=True)
            perm_dir = feat_out_dir / "permutation"
            perm_dir.mkdir(parents=True, exist_ok=True)

            # IG/attr: use aggregated if available; otherwise compute once now
            ig_final = None
            if ig_accum is not None and ig_count > 0:
                ig_final = (ig_accum / float(ig_count)).sort_values(ascending=False)
            else:
                try:
                    ig_final = compute_feature_importance(model, inner_val_loader, device, method=attr_method).sort_values(ascending=False)
                except Exception:
                    ig_final = None
            if ig_final is not None:
                plot_feature_importance(ig_final, epoch=9999, save_dir=ig_dir)
                ig_final.to_csv(ig_dir / "final_aggregated.csv")

            # Permutation: use aggregated if available; otherwise compute once now
            perm_final = None
            if perm_accum is not None and perm_count > 0:
                perm_final = (perm_accum / float(perm_count)).sort_values(ascending=False)
            else:
                try:
                    perm_final = compute_permutation_importance(model, inner_val_loader, device, metric=perm_metric, repeats=perm_repeats).sort_values(ascending=False)
                except Exception:
                    perm_final = None
            if perm_final is not None:
                plot_feature_importance(perm_final, epoch=9999, save_dir=perm_dir)
                perm_final.to_csv(perm_dir / "final_aggregated.csv")
            # Side-by-side final comparison CSV
            if (ig_accum is not None and ig_count > 0) or (perm_accum is not None and perm_count > 0):
                df_final = pd.DataFrame({
                    attr_method: (ig_accum / max(ig_count, 1)) if ig_accum is not None else None,
                    "permutation": (perm_accum / max(perm_count, 1)) if perm_accum is not None else None,
                })
                df_final.to_csv(feat_out_dir / "final_aggregated_ig_vs_permutation.csv")

        # Save final and best metrics in one JSON
            final_summary = {
                "selection_metric": "frame_bal_acc",
                "best": {
                    "epoch": best_epoch,
                    "score": float(best_score),
                    "frame": best_frame_metrics if best_frame_metrics else {},
                    "video": best_video_metrics if best_video_metrics else {},
                    "val_loss": best_val_loss_metric,
                    "video_val_loss": best_video_val_loss_metric,
                },
                "final": {
                    "epoch": int(history[-1]["epoch"]) if history else None,
                    "frame": {
                        "acc": float(frame_metrics.get("acc", np.nan)),
                        "bal_acc": float(frame_metrics.get("bal_acc", np.nan)),
                        "precision": float(frame_metrics.get("precision", np.nan)),
                        "recall": float(frame_metrics.get("recall", np.nan)),
                        "sensitivity": float(frame_metrics.get("sensitivity", np.nan)),
                        "specificity": float(frame_metrics.get("specificity", np.nan)),
                        "f1": float(frame_metrics.get("f1", np.nan)),
                        "auc": float(frame_metrics.get("auc", np.nan)),
                    },
                    "video": {
                        "acc": float(video_metrics.get("acc", np.nan)),
                        "bal_acc": float(video_metrics.get("bal_acc", np.nan)),
                        "precision": float(video_metrics.get("precision", np.nan)),
                        "recall": float(video_metrics.get("recall", np.nan)),
                        "sensitivity": float(video_metrics.get("sensitivity", np.nan)),
                        "specificity": float(video_metrics.get("specificity", np.nan)),
                        "f1": float(video_metrics.get("f1", np.nan)),
                        "auc": float(video_metrics.get("auc", np.nan)),
                    },
                    "val_loss": float(val_loss),
                    "video_val_loss": float(video_val_loss),
                }
            }
            with open(model_dir / "final_metrics.json", "w") as f:
                json.dump(final_summary, f, indent=2)

            print(f"\nBest frame_bal_acc (inner_val) ({name}): {best_score:.4f}")

            # Summarize best for this model/run (outer validation metrics)
            if best_video_metrics is not None and best_frame_metrics is not None:
                row = {"model": name, "best_epoch": best_epoch, "sel_metric": "frame_bal_acc", "best_score": float(best_score)}
                for k, v in best_video_metrics.items():
                    row[f"video_{k}"] = float(v)
                for k, v in best_frame_metrics.items():
                    row[f"frame_{k}"] = float(v)
                fold_rows.append(row)
                print(f"[_train_once] Added row for model {name}: {len(row)} metrics")
            else:
                print(f"[_train_once] Warning: best_video_metrics or best_frame_metrics is None for model {name}")

        result_df = pd.DataFrame(fold_rows)
        print(f"[_train_once] Returning DataFrame with {len(result_df)} rows")
        return result_df

    # Cross-validation mode if folds root provided and exists, OR if --aggregate_only and cv_outroot contains fold_* dirs
    if (cli_args.kfold_root and os.path.isdir(cli_args.kfold_root)) or (
        cli_args.aggregate_only and os.path.isdir(cli_args.cv_outroot) and any(str(d).startswith("fold_") for d in os.listdir(cli_args.cv_outroot))
    ):
        cv_out_root = Path(cli_args.cv_outroot)
        cv_out_root.mkdir(parents=True, exist_ok=True)
        if cli_args.kfold_root and os.path.isdir(cli_args.kfold_root):
            folds = [d for d in sorted(os.listdir(cli_args.kfold_root)) if d.startswith("fold_")]
        else:
            folds = [d for d in sorted(os.listdir(cv_out_root)) if d.startswith("fold_")]
        if not folds:
            raise RuntimeError(f"No folds found under {cli_args.kfold_root or cli_args.cv_outroot}")

        all_rows = []
        # Accumulate feature importances across ALL folds (not per-model)
        # This will be aggregated across all folds regardless of model
        all_folds_imp_sum_ig = None
        all_folds_imp_count_ig = 0
        all_folds_imp_sum_perm = None
        all_folds_imp_count_perm = 0
        # Also keep per-model aggregates for per-model figures
        per_model_imp_sum_ig = {}
        per_model_imp_count_ig = {}
        per_model_imp_sum_perm = {}
        per_model_imp_count_perm = {}

        for fdir in folds:
            fold_idx = int(fdir.split("_")[-1])
            fold_root = os.path.join(cli_args.kfold_root, fdir) if cli_args.kfold_root else None
            out_dir = cv_out_root / fdir

            # Either train this fold or load checkpoint and evaluate on outer validation
            if not cli_args.aggregate_only:
                if not fold_root:
                    raise RuntimeError("--kfold_root must be provided for training folds")
                tr_csv = os.path.join(fold_root, "train.csv")
                va_csv = os.path.join(fold_root, "val.csv")
                df = _train_once(tr_csv, va_csv, out_dir)
            else:
                # Load best.pt from each fold and evaluate on outer validation
                if not fold_root:
                    raise RuntimeError("--kfold_root must be provided for aggregate_only mode to access val.csv")
                va_csv = os.path.join(fold_root, "val.csv")
                if not os.path.exists(va_csv):
                    print(f"[CV] Warning: val.csv not found for fold {fold_idx}: {va_csv}")
                    df = pd.DataFrame()
                else:
                    rows = []
                    for model_name in ("mlp", "residual"):
                        # Try best_model.pt first, then best.pt
                        ckpt_path = out_dir / model_name / "best_model.pt"
                        if not ckpt_path.exists():
                            ckpt_path = out_dir / model_name / "best.pt"
                        
                        if ckpt_path.exists():
                            print(f"[CV] Fold {fold_idx}: Loading {ckpt_path} and evaluating on outer validation...")
                            result = _evaluate_ckpt_on_outer_val(
                                ckpt_path, va_csv, device, metadata_path, selected_features, text_encoder
                            )
                            if result is not None:
                                row = {
                                    "model": result["model"],
                                    "best_epoch": result["best_epoch"],
                                    "sel_metric": result["sel_metric"],
                                    "best_score": result["best_score"],
                                }
                                # Add frame metrics
                                for k, v in result["frame_metrics"].items():
                                    row[f"frame_{k}"] = float(v)
                                # Add video metrics
                                for k, v in result["video_metrics"].items():
                                    row[f"video_{k}"] = float(v)
                                rows.append(row)
                        else:
                            print(f"[CV] Warning: No checkpoint found for fold {fold_idx}, model {model_name}")
                    df = pd.DataFrame(rows)

            if not df.empty:
                df.insert(0, "fold", fold_idx)
                all_rows.append(df)
                print(f"[CV] Fold {fold_idx}: Collected {len(df)} model(s) results")
            else:
                print(f"[CV] Warning: Fold {fold_idx} returned empty DataFrame - no metrics collected")

            # Read per-model feature importances for this fold and add (both IG/attr and permutation if present)
            # Helper: fallback aggregation from per-epoch CSVs if final_aggregated.csv is missing (older runs)
            def _fallback_dir_series(dir_path: Path) -> pd.Series | None:
                if not dir_path.exists():
                    return None
                csvs = sorted([p for p in dir_path.glob("*.csv") if p.is_file()])
                if not csvs:
                    return None
                acc = None; n = 0
                for pth in csvs:
                    try:
                        df = pd.read_csv(pth)
                        if "feature" in df.columns and ("percent" in df.columns or "raw" in df.columns):
                            ser = df.set_index("feature")["percent"] if "percent" in df.columns else df.set_index("feature")["raw"].abs()
                            # Normalize to percent if raw provided
                            if ser.name == "raw":
                                denom = float(ser.abs().sum() or 1.0)
                                ser = (ser.abs() / denom) * 100.0
                            acc = ser if acc is None else acc.add(ser, fill_value=0.0)
                            n += 1
                    except Exception:
                        continue
                if acc is None or n == 0:
                    return None
                return (acc / float(n))
            
            for model_name in ("mlp", "residual"):
                ig_csv = out_dir / model_name / "feature_importance" / attr_method / "final_aggregated.csv"
                perm_csv = out_dir / model_name / "feature_importance" / "permutation" / "final_aggregated.csv"
                
                if ig_csv.exists():
                    ser_ig = pd.read_csv(ig_csv).set_index("feature")["raw"]
                else:
                    ser_ig = _fallback_dir_series(out_dir / model_name / "feature_importance" / attr_method)
                
                if ser_ig is not None and len(ser_ig) > 0:
                    # Add to per-model aggregate
                    per_model_imp_sum_ig[model_name] = per_model_imp_sum_ig.get(model_name, pd.Series(dtype=float)).add(ser_ig, fill_value=0.0)
                    per_model_imp_count_ig[model_name] = per_model_imp_count_ig.get(model_name, 0) + 1
                    # Add to all-folds aggregate (combines across all models)
                    all_folds_imp_sum_ig = ser_ig if all_folds_imp_sum_ig is None else all_folds_imp_sum_ig.add(ser_ig, fill_value=0.0)
                    all_folds_imp_count_ig += 1
                
                if perm_csv.exists():
                    ser_perm = pd.read_csv(perm_csv).set_index("feature")["raw"]
                else:
                    ser_perm = _fallback_dir_series(out_dir / model_name / "feature_importance" / "permutation")
                
                if ser_perm is not None and len(ser_perm) > 0:
                    # Add to per-model aggregate
                    per_model_imp_sum_perm[model_name] = per_model_imp_sum_perm.get(model_name, pd.Series(dtype=float)).add(ser_perm, fill_value=0.0)
                    per_model_imp_count_perm[model_name] = per_model_imp_count_perm.get(model_name, 0) + 1
                    # Add to all-folds aggregate (combines across all models)
                    all_folds_imp_sum_perm = ser_perm if all_folds_imp_sum_perm is None else all_folds_imp_sum_perm.add(ser_perm, fill_value=0.0)
                    all_folds_imp_count_perm += 1

        if all_rows:
            df_all = pd.concat(all_rows, ignore_index=True)
            df_all.to_csv(cv_out_root / "cv_val_folds.csv", index=False)
            # summary mean/std per model
            summary = {}
            for model_name in df_all["model"].unique():
                sdf = df_all[df_all["model"] == model_name]
                stats = {}
                for col in sdf.columns:
                    if col in ("fold", "model", "best_epoch", "sel_metric"): continue
                    vals = pd.to_numeric(sdf[col], errors="coerce")
                    stats[col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "n": int(vals.count()),
                    }
                summary[model_name] = stats
            with open(cv_out_root / "cv_val_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Aggregate feature importances across folds, and save figures for IG and permutation separately
        # IG / chosen attribution method
        for model_name, ssum in per_model_imp_sum_ig.items():
            cnt = per_model_imp_count_ig.get(model_name, 1)
            agg = (ssum / max(cnt, 1)).sort_values(ascending=False)
            out_dir = cv_out_root / model_name / "feature_importance_cv" / attr_method
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save clear-named figure + CSV
            plot_feature_importance(agg, epoch=9999, save_dir=out_dir, filename_stem="final_aggregated_cv")
            pd.DataFrame({
                "feature": agg.index,
                "raw": agg.values,
                "percent": ((agg.abs() / (agg.abs().sum() or 1.0)) * 100.0).values,
            }).to_csv(out_dir / "final_aggregated_cv.csv", index=False)
            # Write top-k text for quick scanning
            top_k = min(20, len(agg))
            with open(out_dir / "top_features.txt", "w") as tf:
                tf.write("Top features (IG/attribution, aggregated across folds)\n")
                for i, (feat, val) in enumerate(agg.iloc[:top_k].items(), 1):
                    pct = (abs(val) / (abs(agg).sum() or 1.0)) * 100.0
                    tf.write(f"{i:2d}. {feat}: {val:.6f} ({pct:.2f}%)\n")
        # Permutation
        for model_name, ssum in per_model_imp_sum_perm.items():
            cnt = per_model_imp_count_perm.get(model_name, 1)
            agg = (ssum / max(cnt, 1)).sort_values(ascending=False)
            out_dir = cv_out_root / model_name / "feature_importance_cv" / "permutation"
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_feature_importance(agg, epoch=9999, save_dir=out_dir, filename_stem="final_aggregated_cv")
            pd.DataFrame({
                "feature": agg.index,
                "raw": agg.values,
                "percent": ((agg.abs() / (agg.abs().sum() or 1.0)) * 100.0).values,
            }).to_csv(out_dir / "final_aggregated_cv.csv", index=False)
            top_k = min(20, len(agg))
            with open(out_dir / "top_features.txt", "w") as tf:
                tf.write("Top features (Permutation, aggregated across folds)\n")
                for i, (feat, val) in enumerate(agg.iloc[:top_k].items(), 1):
                    pct = (abs(val) / (abs(agg).sum() or 1.0)) * 100.0
                    tf.write(f"{i:2d}. {feat}: {val:.6f} ({pct:.2f}%)\n")

        # Write a human-readable summary.txt per model with outer-validation metrics
        if all_rows:
            df_all = pd.concat(all_rows, ignore_index=True)
            for model_name in df_all["model"].unique():
                sdf = df_all[df_all["model"] == model_name]
                lines = []
                lines.append(f"Model: {model_name}")
                lines.append(f"Folds: {sdf['fold'].nunique()}")
                # pick key columns to summarize
                keys = [
                    # frame
                    "frame_acc","frame_bal_acc","frame_precision","frame_recall","frame_sensitivity","frame_specificity","frame_f1","frame_auc",
                    # video
                    "video_acc","video_bal_acc","video_precision","video_recall","video_sensitivity","video_specificity","video_f1","video_auc",
                    # losses
                    "val_loss","video_val_loss",
                ]
                for k in keys:
                    if k in sdf.columns:
                        s = pd.to_numeric(sdf[k], errors="coerce")
                        lines.append(f"{k}: {s.mean():.4f} ± {s.std():.4f} (min={s.min():.4f}, max={s.max():.4f}, n={s.count()})")
                # Best epoch stats (informative only)
                if "best_epoch" in sdf.columns:
                    try:
                        be = pd.to_numeric(sdf["best_epoch"], errors="coerce")
                        lines.append(f"Avg best_epoch: {be.mean():.2f} ± {be.std():.2f}")
                    except Exception:
                        pass
                out_txt = cv_out_root / model_name / "summary.txt"
                out_txt.parent.mkdir(parents=True, exist_ok=True)
                with open(out_txt, "w") as f:
                    f.write("\n".join(lines) + "\n")

        # Produce TWO summary figures at the cv_outroot level
        #  - One for IG/attribution aggregated across all folds (combines all models)
        #  - One for permutation aggregated across all folds (combines all models)
        # These are aggregated across ALL folds regardless of model
        if all_folds_imp_sum_ig is not None and all_folds_imp_count_ig > 0:
            ig_agg_all = (all_folds_imp_sum_ig / max(all_folds_imp_count_ig, 1)).sort_values(ascending=False)
            plot_feature_importance(ig_agg_all, epoch=9999, save_dir=cv_out_root, filename_stem=f"feature_importance_cv_{attr_method}")
            # Also save CSV
            pd.DataFrame({
                "feature": ig_agg_all.index,
                "raw": ig_agg_all.values,
                "percent": ((ig_agg_all.abs() / (ig_agg_all.abs().sum() or 1.0)) * 100.0).values,
            }).to_csv(cv_out_root / f"feature_importance_cv_{attr_method}.csv", index=False)
        
        if all_folds_imp_sum_perm is not None and all_folds_imp_count_perm > 0:
            perm_agg_all = (all_folds_imp_sum_perm / max(all_folds_imp_count_perm, 1)).sort_values(ascending=False)
            plot_feature_importance(perm_agg_all, epoch=9999, save_dir=cv_out_root, filename_stem="feature_importance_cv_permutation")
            # Also save CSV
            pd.DataFrame({
                "feature": perm_agg_all.index,
                "raw": perm_agg_all.values,
                "percent": ((perm_agg_all.abs() / (perm_agg_all.abs().sum() or 1.0)) * 100.0).values,
            }).to_csv(cv_out_root / "feature_importance_cv_permutation.csv", index=False)

        # Also emit a single consolidated metrics text at the root (outer folds summary)
        # If multiple models exist, include a block per model in one file.
        if all_rows:
            df_all = pd.concat(all_rows, ignore_index=True)
            root_lines = ["Cross-Validation Outer Fold Metrics Summary"]
            # Overall set of keys to report
            keys = [
                "frame_acc","frame_bal_acc","frame_precision","frame_recall","frame_sensitivity","frame_specificity","frame_f1","frame_auc",
                "video_acc","video_bal_acc","video_precision","video_recall","video_sensitivity","video_specificity","video_f1","video_auc",
            ]
            for model_name in df_all["model"].unique():
                sdf = df_all[df_all["model"] == model_name]
                root_lines.append("")
                root_lines.append(f"Model: {model_name}")
                root_lines.append(f"Folds: {sdf['fold'].nunique()}")
                for k in keys:
                    if k in sdf.columns:
                        s = pd.to_numeric(sdf[k], errors="coerce")
                        root_lines.append(f"{k}: {s.mean():.4f} ± {s.std():.4f} (min={s.min():.4f}, max={s.max():.4f}, n={s.count()})")
            summary_path = cv_out_root / "summary.txt"
            with open(summary_path, "w") as f:
                f.write("\n".join(root_lines) + "\n")
            print(f"\n[CV] Wrote summary.txt to {summary_path}")
        else:
            print(f"\n[CV] Warning: No rows collected, summary.txt not created. Check if training completed successfully.")
            # Create empty summary.txt with warning
            summary_path = cv_out_root / "summary.txt"
            with open(summary_path, "w") as f:
                f.write("Cross-Validation Outer Fold Metrics Summary\n")
                f.write("\nWarning: No metrics collected. Training may have failed or no models completed.\n")
            print(f"[CV] Created empty summary.txt at {summary_path}")

        print(f"\n[CV] Completed {len(folds)} folds. Outputs at {cv_out_root}")
        return

    # Single split (default)
    _ = _train_once(train_csv, val_csv, Path("/cluster/home/ammaa/Downloads/meta-cadica/scripts/meta_analysis/results"))

if __name__ == "__main__":
    main()
