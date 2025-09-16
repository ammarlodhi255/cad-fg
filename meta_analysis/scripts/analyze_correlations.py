import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from meta_analysis.scripts.train_meta import MetadataOnlyDataset

def analyze_feature_correlations(dataset, save_dir):
    """Analyze and visualize feature correlations"""
    # Collect all feature vectors
    features = []
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        features.append(sample["x"].numpy())
        labels.append(sample["y"].item())
    
    features = np.stack(features)
    labels = np.array(labels)
    
    # Get feature names
    numeric_features = dataset.meta_df.columns.intersection([
        "age", "height", "weight", "bmi", "evol_dm_years", 
        "lvef_pct", "n_vessels", "max_involvement_pct"
    ]).tolist()
    
    categorical_features = dataset.meta_df.columns.intersection([
        "sex", "diabetes", "smoker", "hypertension", "dyslipidemia",
        "kidney_failure", "heart_failure", "atrial_fib"
    ]).tolist()
    
    # Create correlation matrix
    df = pd.DataFrame(features, columns=numeric_features + categorical_features)
    corr = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation matrix
    corr.to_csv(save_dir / 'feature_correlations.csv')
    
    # Analyze correlation with target
    target_corr = pd.DataFrame({
        'feature': df.columns,
        'correlation': [np.corrcoef(df[col], labels)[0,1] for col in df.columns]
    }).sort_values('correlation', key=abs, ascending=False)
    
    # Plot target correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(data=target_corr, x='correlation', y='feature')
    plt.title('Feature Correlations with Target')
    plt.tight_layout()
    plt.savefig(save_dir / 'target_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save target correlations
    target_corr.to_csv(save_dir / 'target_correlations.csv', index=False)
    
    # Print top correlations
    print("\nTop positive correlations with target:")
    print(target_corr[target_corr['correlation'] > 0].head())
    print("\nTop negative correlations with target:")
    print(target_corr[target_corr['correlation'] < 0].head())
    
    # Find highly correlated feature pairs
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i,j]) > 0.7:  # threshold for high correlation
                high_corr.append({
                    'feature1': corr.columns[i],
                    'feature2': corr.columns[j],
                    'correlation': corr.iloc[i,j]
                })
    
    if high_corr:
        print("\nHighly correlated feature pairs (|r| > 0.7):")
        for pair in sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")

def main():
    # Settings
    save_dir = Path("meta_analysis/results/correlations")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data for analysis
    train_csv = "outputs/index/train.csv"
    metadata_path = "data/CADICA/metadata.xlsx"
    
    # Create dataset with all features
    train_ds = MetadataOnlyDataset(train_csv, metadata_path, selected_features=None)
    
    print("Analyzing feature correlations...")
    analyze_feature_correlations(train_ds, save_dir)
    print(f"\nResults saved to {save_dir}")

if __name__ == "__main__":
    main()
