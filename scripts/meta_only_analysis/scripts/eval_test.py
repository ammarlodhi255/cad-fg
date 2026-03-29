import os, sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from meta_analysis.src.meta_model import MetadataFusionMLP, MetadataResidualMLP
from meta_analysis.scripts.train_meta import MetadataOnlyDataset, evaluate

def _build_text_encoder_if_needed(cfg: dict, selected_features: dict):
    if selected_features and selected_features.get("text"):
        from meta_analysis.src.text_encoder import TextFeatureEncoder
        return TextFeatureEncoder(
            model_name=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds")
        )
    return None


def _evaluate_ckpt_on_test(ckpt_path: Path, device, cfg: dict, test_csv: str, metadata_path: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    selected_features = ckpt.get("selected_features", {})
    text_encoder = _build_text_encoder_if_needed(cfg, selected_features)

    test_ds = MetadataOnlyDataset(test_csv, metadata_path, selected_features, text_encoder)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

    if model_name == "mlp":
        model = MetadataFusionMLP(
            tabular_dim=test_ds.tab_dim,
            text_dim=test_ds.text_dim,
            hidden_dims=[128, 64, 32],
            dropout=0.3,
        )
    elif model_name == "residual":
        model = MetadataResidualMLP(
            tabular_dim=test_ds.tab_dim,
            text_dim=test_ds.text_dim,
            hidden_dim=128,
            n_blocks=3,
            dropout=0.3,
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    frame_metrics, video_metrics, video_df, _ = evaluate(model, test_loader, device, compute_importance=False)
    return model_name, selected_features, frame_metrics, video_metrics, video_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_root", default="meta_analysis/results_cv", help="Root containing per-fold outputs (for CV evaluation)")
    ap.add_argument("--single_root", default="meta_analysis/results", help="Root containing single-split outputs")
    ap.add_argument("--test_csv", default="outputs/index/test.csv")
    ap.add_argument("--metadata_path", default="data/CADICA/metadata.xlsx")
    ap.add_argument("--mode", choices=["auto","single","cv"], default="auto", help="Which results to evaluate")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open("meta_analysis/meta_cfg.yaml"))

    # Auto mode: prefer CV if there are folds present
    do_cv = False
    if args.mode == "cv":
        do_cv = True
    elif args.mode == "single":
        do_cv = False
    else:
        # auto
        cv_root = Path(args.cv_root)
        do_cv = cv_root.exists() and any((cv_root / d).is_dir() and str(d).startswith("fold_") for d in os.listdir(cv_root))

    results = []

    if do_cv:
        cv_root = Path(args.cv_root)
        folds = [d for d in sorted(os.listdir(cv_root)) if str(d).startswith("fold_")]
        if not folds:
            print(f"No folds found under {cv_root}. Nothing to evaluate.")
            return
        print(f"Evaluating CV outputs under {cv_root} on test set…")
        for fdir in tqdm(folds, desc="Folds"):
            froot = cv_root / fdir
            for model_name in ("mlp", "residual"):
                ckpt_path = froot / model_name / "best_model.pt"
                if not ckpt_path.exists():
                    continue
                mname, sel, fm, vm, vdf = _evaluate_ckpt_on_test(ckpt_path, device, cfg, args.test_csv, args.metadata_path)
                results.append({
                    "fold": int(str(fdir).split("_")[-1]),
                    "model": mname,
                    "frame": fm,
                    "video": vm,
                })
        if not results:
            print("No CV checkpoints found to evaluate.")
            return
        # Aggregate by model
        df_rows = []
        for r in results:
            row = {"fold": r["fold"], "model": r["model"]}
            for k, v in r["frame"].items():
                row[f"frame_{k}"] = v
            for k, v in r["video"].items():
                row[f"video_{k}"] = v
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        out_dir = cv_root
        df.to_csv(out_dir / "test_cv_folds.csv", index=False)
        summary = {}
        for model in df["model"].unique():
            sdf = df[df["model"] == model]
            stats = {}
            for col in sdf.columns:
                if col in ("fold","model"): continue
                s = pd.to_numeric(sdf[col], errors="coerce")
                stats[col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "n": int(s.count()),
                }
            summary[model] = stats
        with open(out_dir / "test_cv_summary.json", "w") as f:
            import json
            json.dump(summary, f, indent=2)
        print("Saved CV test metrics to:")
        print(f"- {out_dir/'test_cv_folds.csv'}")
        print(f"- {out_dir/'test_cv_summary.json'}")
        return

    # Single-split evaluation (default)
    results_dir = Path(args.single_root)
    model_paths = list(results_dir.glob("*/best_model.pt"))
    if not model_paths:
        # Backward-compat: older naming
        model_paths = list(results_dir.glob("best_*_model.pt"))
    if not model_paths:
        print(f"No saved models found in {results_dir}")
        return

    for model_path in tqdm(model_paths, desc="Evaluating models"):
        print(f"\nEvaluating {model_path}")
        mname, selected_features, frame_metrics, video_metrics, video_df = _evaluate_ckpt_on_test(model_path, device, cfg, args.test_csv, args.metadata_path)

        print("\nTest Set Results:")
        print("\nFrame-level metrics:")
        for k in ("acc","bal_acc","precision","recall","sensitivity","specificity","f1","auc"):
            if k in frame_metrics:
                print(f"  {k:12s}: {frame_metrics[k]:.4f}")
        print("\nVideo-level metrics:")
        for k in ("acc","bal_acc","precision","recall","sensitivity","specificity","f1","auc"):
            if k in video_metrics:
                print(f"  {k:12s}: {video_metrics[k]:.4f}")

        # Save detailed results
        out = {
            "model_name": mname,
            "frame_metrics": frame_metrics,
            "video_metrics": video_metrics,
            "selected_features": selected_features,
        }
        torch.save(out, results_dir / f"test_metrics_{mname}.pt")
        video_df.to_csv(results_dir / f"test_video_preds_{mname}.csv", index=False)

        if selected_features:
            print("\nFeature subset used:")
            print("Numeric:", selected_features.get("numeric"))
            print("Categorical:", selected_features.get("categorical"))
            if selected_features.get("text"):
                print("Text:", selected_features.get("text"))
        else:
            print("\nUsing all available features")

if __name__ == "__main__":
    main()
