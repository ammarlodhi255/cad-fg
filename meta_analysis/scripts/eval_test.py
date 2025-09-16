import os, sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from meta_analysis.src.meta_model import MetadataOnlyMLP, MetadataOnlyResidualMLP
from meta_analysis.scripts.train_meta import MetadataOnlyDataset, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("meta_analysis/results")
    
    # Load test set
    test_csv = "outputs/index/test.csv"
    metadata_path = "data/CADICA/metadata.xlsx"
    
    # Evaluate all saved models
    model_paths = list(results_dir.glob("best_*_model.pt"))
    if not model_paths:
        print("No saved models found in meta_analysis/results/")
        return
        
    for model_path in tqdm(model_paths, desc="Evaluating models"):
        print(f"\nEvaluating {model_path.name}")
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=False)  # Keep False as we need metadata too
        model_name = ckpt["model_name"]
        selected_features = ckpt["selected_features"]
        
        # Create test dataset with same feature selection
        test_ds = MetadataOnlyDataset(test_csv, metadata_path, selected_features)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
        
        # Initialize correct model architecture
        if model_name == "mlp":
            model = MetadataOnlyMLP(
                in_dim=test_ds.tab_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.3
            )
        elif model_name == "residual":
            model = MetadataOnlyResidualMLP(
                in_dim=test_ds.tab_dim,
                hidden_dim=128,
                n_blocks=3,
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Load weights and evaluate
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        frame_metrics, video_metrics, video_df, _ = evaluate(model, test_loader, device, compute_importance=False)
        
        print("\nTest Set Results:")
        print("\nFrame-level metrics:")
        for k, v in frame_metrics.items():
            print(f"  {k:8s}: {v:.4f}")
        
        print("\nVideo-level metrics:")
        for k, v in video_metrics.items():
            print(f"  {k:8s}: {v:.4f}")
        
        # Save detailed results
        results = {
            "model_name": model_name,
            "frame_metrics": frame_metrics,
            "video_metrics": video_metrics,
            "selected_features": selected_features
        }
        
        # Save test metrics
        torch.save(results, results_dir / f"test_metrics_{model_name}.pt")
        
        # Save video-level predictions
        video_df.to_csv(results_dir / f"test_video_preds_{model_name}.csv", index=False)
        
        # Print feature information
        if selected_features:
            print("\nFeature subset used:")
            print("Numeric:", selected_features["numeric"])
            print("Categorical:", selected_features["categorical"])
        else:
            print("\nUsing all available features")

if __name__ == "__main__":
    main()