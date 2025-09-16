import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from meta_analysis.src.feature_analysis import FeatureImportanceAnalyzer, suggest_feature_selection
from meta_analysis.scripts.train_meta import MetadataOnlyDataset

def main():
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("meta-analysis/results")
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Load best model
    model_paths = list(results_dir.glob("model_*.pt"))
    best_auc = -1
    best_model_info = None
    
    for mp in model_paths:
        ckpt = torch.load(mp, map_location="cpu")
        auc = ckpt["video_metrics"]["auc"]
        if auc > best_auc:
            best_auc = auc
            best_model_info = ckpt
    
    if best_model_info is None:
        print("No models found! Run train_meta.py first.")
        return
    
    print(f"Analyzing best model ({best_model_info['model_name']}) with AUC = {best_auc:.4f}")
    
    # Load validation dataset for analysis
    val_ds = MetadataOnlyDataset(
        "outputs/index/val.csv",
        "data/CADICA/metadata.xlsx",
        best_model_info.get("selected_features", None)
    )
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        model=best_model_info["model"],
        dataset=val_ds,
        device=device
    )
    
    # 1. Compute feature importance
    print("Computing feature importance...")
    importance_df = analyzer.get_all_importances(n_samples=200)
    importance_df.to_csv(analysis_dir / "feature_importance.csv")
    
    # 2. Plot feature importance heatmap
    print("Plotting feature importance...")
    analyzer.plot_feature_importance(
        importance_df,
        save_path=analysis_dir / "feature_importance_heatmap.png"
    )
    
    # 3. Plot feature correlations
    print("Plotting feature correlations...")
    analyzer.plot_feature_correlations(
        save_path=analysis_dir / "feature_correlations.png"
    )
    
    # 4. Analyze prediction confidence
    print("Analyzing prediction confidence...")
    confidence_df = analyzer.analyze_prediction_confidence(
        save_path=analysis_dir / "prediction_confidence.png"
    )
    confidence_df.to_csv(analysis_dir / "prediction_confidence.csv")
    
    # 5. Suggest feature selection
    print("\nSuggesting feature selection...")
    suggested_features = suggest_feature_selection(
        importance_df,
        correlation_threshold=0.8,
        top_k=10
    )
    
    print("\nRecommended feature subset:")
    print("Numeric features:", suggested_features["numeric"])
    print("Categorical features:", suggested_features["categorical"])
    
    # Save suggestions
    with open(analysis_dir / "suggested_features.txt", "w") as f:
        f.write("Recommended feature subset based on importance analysis:\n\n")
        f.write("Numeric features:\n")
        f.write("\n".join(f"- {f}" for f in suggested_features["numeric"]))
        f.write("\n\nCategorical features:\n")
        f.write("\n".join(f"- {f}" for f in suggested_features["categorical"]))

if __name__ == "__main__":
    main()
