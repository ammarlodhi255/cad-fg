import os, sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.dataio.dataset import load_metadata_any, build_meta_vectors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from meta_analysis.src.meta_model import MetadataOnlyMLP, MetadataOnlyResidualMLP

class MetadataOnlyDataset(Dataset):
    def __init__(self, csv_path, metadata_path, selected_features=None):
        self.df = pd.read_csv(csv_path)
        self.meta_df = load_metadata_any(metadata_path)
        self.meta = build_meta_vectors(self.meta_df, selected_features=selected_features)
        
        # Get feature dimension from first patient
        any_pid = int(self.df.iloc[0]["patient"])
        self.tab_dim = len(self.meta[any_pid])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = int(r.patient)
        x = torch.tensor(self.meta[pid], dtype=torch.float32)
        y = torch.tensor(int(r.label), dtype=torch.long)
        return {
            "x": x, "y": y,
            "patient": pid,
            "video": int(r.video),
            "frame": int(r.frame)
        }

def compute_feature_importance(model, loader, device, feature_names):
    """Compute feature importance using Integrated Gradients"""
    model.eval()
    ig = IntegratedGradients(model)
    
    # Get a batch of data
    batch = next(iter(loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    
    # Compute attributions
    attributions = ig.attribute(x, target=y)
    attr_mean = attributions.abs().mean(0).cpu().numpy()
    
    return pd.Series(attr_mean, index=feature_names)

def evaluate(model, loader, device, compute_importance=False):
    model.eval()
    probs, ys = [], []
    patients, videos = [], []
    
    # Get feature importance if requested
    importance = None
    if compute_importance:
        try:
            importance = compute_feature_importance(model, loader, device, loader.dataset.get_feature_names())
        except Exception as e:
            print(f"Warning: Could not compute feature importance: {e}")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            logits = model(x)
            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            
            probs.append(p)
            ys.append(y)
            patients.append(batch["patient"].numpy())
            videos.append(batch["video"].numpy())
    
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    
    # Frame-level metrics
    frame_metrics = {
        "bal_acc": balanced_accuracy_score(y, p >= 0.5),
        "f1": f1_score(y, p >= 0.5),
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
    
    video_metrics = {
        "bal_acc": balanced_accuracy_score(agg.y_true, agg.y_prob >= 0.5),
        "f1": f1_score(agg.y_true, agg.y_prob >= 0.5),
        "auc": roc_auc_score(agg.y_true, agg.y_prob)
    }
    
    return frame_metrics, video_metrics, agg, importance

def plot_feature_importance(importance, epoch, save_dir):
    """Plot and save feature importance"""
    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind='barh')
    plt.title(f'Feature Importance (Epoch {epoch+1})')
    plt.tight_layout()
    plt.savefig(save_dir / f'feature_importance_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw values
    importance.to_csv(save_dir / f'feature_importance_epoch_{epoch+1}.csv')

def main():
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 50
    lr = 1e-3
    patience = 10  # early stopping
    
    # Data
    metadata_path = "data/CADICA/metadata.xlsx"
    train_csv = "outputs/index/train.csv"
    val_csv = "outputs/index/val.csv"
    
    # Optional: feature selection
    selected_features = {
        "numeric": ["age", "bmi", "height", "weight", "evol_dm_years", "lvef_pct", "n_vessels", "max_involvement_pct"],
        "categorical": ["sex", "diabetes", "smoker", "hypertension", "dyslipidemia", "kidney_failure", "heart_failure", "atrial_fib"]
    }
    
    # Datasets
    train_ds = MetadataOnlyDataset(train_csv, metadata_path, selected_features)
    val_ds = MetadataOnlyDataset(val_csv, metadata_path, selected_features)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    print(f"Feature dimension: {train_ds.tab_dim}")
    
    # Model variants
    models = {
        "mlp": MetadataOnlyMLP(
            in_dim=train_ds.tab_dim,
            hidden_dims=[128, 64, 32],
            dropout=0.3
        ),
        "residual": MetadataOnlyResidualMLP(
            in_dim=train_ds.tab_dim,
            hidden_dim=128,
            n_blocks=3,
            dropout=0.3
        )
    }
    
    os.makedirs("meta_analysis/results", exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_bal_acc = 0
        epochs_no_improve = 0
        best_state = None
        
        pbar = trange(epochs, desc=f"Training {name}")
        for epoch in pbar:
            # Train
            model.train()
            train_loss = 0
            n_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_loss = train_loss / n_batches
            
            # Validate and compute feature importance every 5 epochs
            compute_importance = (epoch + 1) % 5 == 0
            frame_metrics, video_metrics, _, importance = evaluate(
                model, val_loader, device, compute_importance=compute_importance
            )
            
            # Save feature importance if computed
            if importance is not None:
                importance_dir = Path("meta_analysis/results") / name / "feature_importance"
                importance_dir.mkdir(parents=True, exist_ok=True)
                plot_feature_importance(importance, epoch, importance_dir)
            
            # Update progress bar
            pbar.set_postfix({
                'epoch': f"{epoch+1}/{epochs}",
                'loss': f"{avg_loss:.4f}",
                'bal_acc': f"{frame_metrics['bal_acc']:.4f}",
                'vid_bal_acc': f"{video_metrics['bal_acc']:.4f}"
            })
            
            # Save best model based on validation balanced accuracy
            if frame_metrics["bal_acc"] > best_val_bal_acc:
                best_val_bal_acc = frame_metrics["bal_acc"]
                best_state = model.state_dict().copy()
                print(f"\n[ckpt] New best balanced accuracy: {best_val_bal_acc:.4f}")
                epochs_no_improve = 0
                
                # Save checkpoint
                torch.save({
                    "model_name": name,
                    "state_dict": best_state,
                    "frame_metrics": frame_metrics,
                    "video_metrics": video_metrics,
                    "feature_dim": train_ds.tab_dim,
                    "selected_features": selected_features,
                    "epoch": epoch
                }, f"meta_analysis/results/best_{name}_model.pt")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        print(f"\nBest validation balanced accuracy ({name}): {best_val_bal_acc:.4f}")

if __name__ == "__main__":
    main()