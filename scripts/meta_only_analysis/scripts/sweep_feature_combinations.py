#!/usr/bin/env python3
import os
import sys
import itertools
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from meta_analysis.scripts.train_meta import MetadataOnlyDataset, evaluate
from meta_analysis.src.meta_model import MetadataResidualMLP


def set_seed(s: int = 1337):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_selected_features(combo):
    """combo: list of tuples (kind, name) where kind in {numeric, categorical, text}"""
    sel = {"numeric": [], "categorical": [], "text": []}
    for kind, name in combo:
        if kind == "numeric":
            sel["numeric"].append(name)
        elif kind == "categorical":
            sel["categorical"].append(name)
        elif kind == "text":
            sel["text"].append(name)
    # Remove empties if not used
    if not sel["numeric"]:
        sel["numeric"] = []
    if not sel["categorical"]:
        sel["categorical"] = []
    if not sel["text"]:
        sel["text"] = []
    return sel


def train_and_eval_residual(train_ds, val_ds, device, epochs=20, lr=1e-3, batch_size=64, patience=5):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MetadataResidualMLP(
        tabular_dim=train_ds.tab_dim,
        text_dim=train_ds.text_dim,
        hidden_dim=128,
        n_blocks=3,
        dropout=0.3,
        out_dim=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_metric = -float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in trange(epochs, desc="train", leave=False):
        model.train()
        tr_loss = 0.0; n = 0
        for batch in train_loader:
            x_tab = batch["x_tab"].to(device)
            x_text = batch["x_text"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(x_tab, x_text)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item(); n += 1
        # evaluate
        frame_metrics, video_metrics, _, _ = evaluate(model, val_loader, device, compute_importance=False)
        # Use AUC (frame) as default ranking metric
        metric = float(frame_metrics.get("auc", 0.0))
        if metric > best_metric:
            best_metric = metric
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    frame_metrics, video_metrics, _, _ = evaluate(model, val_loader, device, compute_importance=False)
    return frame_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep feature combinations for residual metadata model")
    parser.add_argument("--cfg", default="meta_analysis/meta_cfg.yaml")
    parser.add_argument("--train_csv", default="outputs/index/single_split/train.csv")
    parser.add_argument("--val_csv", default="outputs/index/single_split/val.csv")
    parser.add_argument("--metadata_path", default="data/CADICA/metadata.xlsx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--metric", default="frame_auc", choices=["frame_auc","video_auc"], help="Metric to rank combinations")
    parser.add_argument("--out_dir", default="meta_analysis/results/combination_sweep")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target features
    FEATS = [
        ("text", "another_comorb"),
        ("text", "indication"),
        ("categorical", "hypertension"),
        ("categorical", "diabetes"),
        ("numeric", "age"),
    ]

    # Load config (for text encoder settings)
    try:
        with open(args.cfg, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    results = []

    # Iterate all non-empty combinations
    for k in range(1, len(FEATS) + 1):
        for combo in itertools.combinations(FEATS, k):
            selected = build_selected_features(combo)

            # Text encoder only if text fields present
            text_encoder = None
            if selected.get("text"):
                from meta_analysis.src.text_encoder import TextFeatureEncoder
                text_encoder = TextFeatureEncoder(
                    model_name=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
                )

            # Build datasets
            train_ds = MetadataOnlyDataset(args.train_csv, args.metadata_path, selected, text_encoder)
            val_ds = MetadataOnlyDataset(args.val_csv, args.metadata_path, selected, text_encoder)

            # Train + eval
            frame_m, video_m = train_and_eval_residual(
                train_ds, val_ds, device,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, patience=args.patience,
            )

            row = {
                "combo": ",".join([f"{k}:{n}" for k, n in combo]),
                "n_features_tab": train_ds.tab_dim,
                "n_features_text": train_ds.text_dim,
                "frame_auc": float(frame_m.get("auc", 0.0)),
                "video_auc": float(video_m.get("auc", 0.0)),
                "frame_acc": float(frame_m.get("acc", 0.0)) if "acc" in frame_m else np.nan,
                "video_acc": float(video_m.get("acc", 0.0)) if "acc" in video_m else np.nan,
            }
            results.append(row)

    df = pd.DataFrame(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "residual_combinations_auc.csv"
    df.sort_values(args.metric, ascending=False).to_csv(out_csv, index=False)

    # Print top-5
    print("Top 5 combinations by", args.metric)
    print(df.sort_values(args.metric, ascending=False).head(5).to_string(index=False))
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

