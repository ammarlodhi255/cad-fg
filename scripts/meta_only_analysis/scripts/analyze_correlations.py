import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from meta_analysis.scripts.train_meta import MetadataOnlyDataset


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

def analyze_feature_correlations(dataset, save_dir):
    """Analyze and visualize feature correlations"""
    # Collect all feature vectors
    features = []
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        features.append(sample["x_tab"].numpy())
        labels.append(sample["y"].item())
    
    features = np.stack(features)
    labels = np.array(labels)
    
    # Use feature names matching vector order
    feature_names = dataset.get_feature_names()

    # Create correlation matrix
    df = pd.DataFrame(features, columns=feature_names)
    corr = df.corr()

    # Plot correlation heatmap (keep diverging scheme, apply fonts)
    _apply_plot_style()
    fig_hm, ax_hm = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax_hm)
    ax_hm.set_title('Feature Correlations')
    fig_hm.tight_layout()
    fig_hm.savefig(save_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    fig_hm.savefig(save_dir / 'feature_correlations.pdf')
    fig_hm.savefig(save_dir / 'feature_correlations.svg')
    plt.close(fig_hm)
    
    # Save correlation matrix
    corr.to_csv(save_dir / 'feature_correlations.csv')
    
    # Analyze correlation with target
    target_corr = pd.DataFrame({
        'feature': df.columns,
        'correlation': [np.corrcoef(df[col], labels)[0,1] for col in df.columns]
    }).sort_values('correlation', key=abs, ascending=False)

    # Styled horizontal bar plot for target correlations
    palette = _apply_plot_style()
    top = min(len(target_corr), 30)
    tc = target_corr.iloc[:top]
    labels_tc = tc['feature'].tolist()
    values_tc = tc['correlation'].tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(labels_tc))
    colors = palette[: len(labels_tc)]
    bars = ax.barh(y, values_tc, color=colors, edgecolor="#2b2b2b", linewidth=1.2)

    ax.set_xlabel('Correlation with target')
    ax.set_yticks(y, labels_tc)
    ax.invert_yaxis()
    ax.grid(True, axis='x', which='major', linewidth=0.8, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, axis='x', which='minor', linewidth=0.4, alpha=0.15)
    ax.set_axisbelow(True)
    max_abs = max(abs(v) for v in values_tc) if values_tc else 1.0
    ax.set_xlim(-max_abs * 1.20, max_abs * 1.20)

    for rect, val in zip(bars, values_tc):
        # Place labels just outside the bar on the right/left depending on sign
        offset = max_abs * 0.01
        text_x = rect.get_width() + offset if val >= 0 else rect.get_width() - offset
        ha = 'left' if val >= 0 else 'right'
        ax.text(text_x, rect.get_y() + rect.get_height() / 2, f"{val:.2f}", va='center', ha=ha, fontsize=13)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.35, right=0.96, top=0.92, bottom=0.10)
    ax.set_title('Feature Correlations with Target')

    fig.savefig(save_dir / 'target_correlations.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'target_correlations.pdf')
    fig.savefig(save_dir / 'target_correlations.svg')
    plt.close(fig)
    
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
