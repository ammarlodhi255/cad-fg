import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    Occlusion,
    FeatureAblation
)

class FeatureImportanceAnalyzer:
    def __init__(self, model, dataset, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self):
        """Get expanded feature names (numeric + one-hot categories)"""
        if hasattr(self.dataset, 'get_feature_names'):
            return self.dataset.get_feature_names()
        
        # Fallback to basic numeric + categorical from dataset.py
        from src.dataio.dataset import CLIN_NUM, CLIN_CAT, CAT_VALS
        names = list(CLIN_NUM)
        for k in CLIN_CAT:
            base = CAT_VALS.get(k, ["no","yes"])
            names.extend([f"{k}={b}" for b in base])
        return names
    
    def _get_batch(self, n_samples=100):
        """Get a representative batch of samples"""
        indices = np.random.choice(len(self.dataset), n_samples, replace=False)
        xs, ys = [], []
        for i in indices:
            sample = self.dataset[i]
            xs.append(sample["x"])
            ys.append(sample["y"])
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)
    
    def integrated_gradients(self, n_samples=100, n_steps=50):
        """Compute feature importance using Integrated Gradients"""
        ig = IntegratedGradients(self.model)
        X, y = self._get_batch(n_samples)
        
        # Compute attributions
        attributions = ig.attribute(X, target=y, n_steps=n_steps)
        attr_mean = attributions.abs().mean(0).cpu().numpy()
        
        return pd.Series(attr_mean, index=self.feature_names)
    
    def deep_lift(self, n_samples=100):
        """Compute feature importance using DeepLift"""
        dl = DeepLift(self.model)
        X, y = self._get_batch(n_samples)
        
        attributions = dl.attribute(X, target=y)
        attr_mean = attributions.abs().mean(0).cpu().numpy()
        
        return pd.Series(attr_mean, index=self.feature_names)
    
    def feature_ablation(self, n_samples=100):
        """Compute feature importance using Feature Ablation"""
        fa = FeatureAblation(self.model)
        X, y = self._get_batch(n_samples)
        
        attributions = fa.attribute(X, target=y)
        attr_mean = attributions.abs().mean(0).cpu().numpy()
        
        return pd.Series(attr_mean, index=self.feature_names)
    
    def get_all_importances(self, n_samples=100):
        """Get feature importance from multiple methods"""
        importances = pd.DataFrame({
            'integrated_gradients': self.integrated_gradients(n_samples),
            'deep_lift': self.deep_lift(n_samples),
            'feature_ablation': self.feature_ablation(n_samples)
        })
        
        # Normalize each method's scores
        for col in importances.columns:
            importances[col] = importances[col] / importances[col].sum()
        
        return importances
    
    def plot_feature_importance(self, importance_df, save_path=None, figsize=(12, 8)):
        """Plot feature importance heatmap"""
        plt.figure(figsize=figsize)
        sns.heatmap(
            importance_df,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            yticklabels=self.feature_names
        )
        plt.title('Feature Importance by Different Methods')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_correlations(self, save_path=None, figsize=(10, 10)):
        """Plot feature correlation matrix"""
        # Get all features as a matrix
        features = []
        for i in range(len(self.dataset)):
            features.append(self.dataset[i]["x"].numpy())
        features = np.stack(features)
        
        # Compute correlations
        corr = pd.DataFrame(
            features,
            columns=self.feature_names
        ).corr()
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True
        )
        plt.title('Feature Correlations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_prediction_confidence(self, save_path=None):
        """Analyze prediction confidence vs feature values"""
        all_data = []
        model = self.model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                x = sample["x"].to(self.device).unsqueeze(0)
                logits = model(x)
                prob = torch.softmax(logits, 1)[0, 1].item()  # prob of class 1
                
                features = {
                    name: val.item() 
                    for name, val in zip(self.feature_names, sample["x"])
                }
                features.update({
                    "prediction": prob,
                    "true_label": sample["y"].item()
                })
                all_data.append(features)
        
        df = pd.DataFrame(all_data)
        
        # Create confidence analysis plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        # Plot for numeric features
        numeric_features = [f for f in self.feature_names if not "=" in f]
        for i, feat in enumerate(numeric_features[:9]):  # First 9 features
            sns.scatterplot(
                data=df,
                x=feat,
                y="prediction",
                hue="true_label",
                ax=axes[i]
            )
            axes[i].set_title(f"{feat} vs Prediction Confidence")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return df

def suggest_feature_selection(importance_df, correlation_threshold=0.8, top_k=None):
    """
    Suggest feature selection based on importance and correlations
    
    Args:
        importance_df: DataFrame with feature importance scores
        correlation_threshold: Remove highly correlated features above this threshold
        top_k: Keep only top K most important features
    """
    # Average importance across methods
    mean_importance = importance_df.mean(axis=1)
    
    # Sort features by importance
    sorted_features = mean_importance.sort_values(ascending=False)
    
    if top_k:
        sorted_features = sorted_features[:top_k]
    
    selected_features = {
        "numeric": [],
        "categorical": []
    }
    
    for feat in sorted_features.index:
        if "=" in feat:  # categorical
            cat = feat.split("=")[0]
            if cat not in selected_features["categorical"]:
                selected_features["categorical"].append(cat)
        else:  # numeric
            selected_features["numeric"].append(feat)
    
    return selected_features
