#!/usr/bin/env python3
import os, yaml, random, numpy as np, torch, pandas as pd
import sys
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, Subset
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.trainloop.engine import train_one_epoch, evaluate

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

class AugmentedCadicaDataset(Dataset):
    """Extended CadicaDataset with clinically appropriate augmentation for minority class"""
    
    def __init__(self, base_dataset: CadicaDataset, minority_class: int = 0, augment_factor: int = 3):
        self.base_dataset = base_dataset
        self.minority_class = minority_class
        self.augment_factor = augment_factor
        
        # Find minority class samples
        self.minority_indices = [
            i for i in range(len(base_dataset)) 
            if base_dataset[i]['y'].item() == minority_class
        ]
        
        # Create clinically appropriate augmentation transform
        self.minority_transform = self.get_safe_augment_tf(base_dataset.tfm.transforms[0].height)
        
        print(f"[dataset] Original size: {len(base_dataset)}")
        print(f"[dataset] Minority class ({minority_class}) samples: {len(self.minority_indices)}")
        print(f"[dataset] Will create {augment_factor-1} additional samples per minority sample")
        print(f"[dataset] Final size: {len(self)}")
        
    def get_safe_augment_tf(self, size=224):
        """
        Clinically appropriate augmentation pipeline for coronary angiography:
        - Only safe photometric augmentations that mimic acquisition variability
        - Very limited geometric transformations within clinical plausibility
        - No anatomically impossible transformations
        """
        return A.Compose([
            A.Resize(size, size),  # Maintain aspect ratio
            
            # Photometric augmentations (mimic fluoroscopy variations)
            A.OneOf([
                # Contrast/brightness to simulate dose variations
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,  # Mild brightness adjustment
                    contrast_limit=0.1,    # Mild contrast adjustment
                    p=1
                ),
                # CLAHE to normalize contrast across acquisitions
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=1
                ),
            ], p=0.8),
            
            # Acquisition noise simulation
            A.GaussNoise(
                var_limit=(2.0, 10.0),  # Mild noise, won't obscure details
                mean=0,
                p=0.5
            ),
            
            # Very mild blur to simulate minor focus variations
            A.GaussianBlur(
                blur_limit=(3, 3),  # Minimal blur
                p=0.3
            ),
            
            # Minimal geometric augmentations
            A.ShiftScaleRotate(
                shift_limit=0.02,      # Tiny translation (2%)
                scale_limit=0.05,      # Small zoom variation (5%)
                rotate_limit=5,        # Very limited rotation (±5°)
                border_mode=0,         # Border_mode=0 for zero padding
                p=0.5
            ),
            
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.base_dataset) + len(self.minority_indices) * (self.augment_factor - 1)
    
    def __getitem__(self, idx):
        base_len = len(self.base_dataset)
        
        if idx < base_len:
            return self.base_dataset[idx]
        
        aug_idx = (idx - base_len) // (self.augment_factor - 1)
        orig_idx = self.minority_indices[aug_idx]
        orig_sample = self.base_dataset[orig_idx]
        
        img = orig_sample['x'].numpy().transpose(1, 2, 0)
        aug_data = self.minority_transform(image=img)
        aug_img = aug_data["image"]
        
        return {
            'x': aug_img,
            'y': orig_sample['y'],
            'clin': orig_sample['clin'],
            'patient': orig_sample['patient'],
            'video': orig_sample['video'],
            'frame': orig_sample['frame'],
            'img_path': orig_sample['img_path']
        }

def create_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Creates a balanced sampler for the dataset"""
    labels = [dataset[i]['y'].item() for i in range(len(dataset))]
    
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    print(f"[sampler] Class counts: {class_counts}")
    print(f"[sampler] Class weights: {class_weights}")
    
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

def load_all_data(cfg):
    """Load and combine training and validation data for cross-validation"""
    # Load both train and val CSVs
    train_df = pd.read_csv("outputs/index/train.csv")
    val_df = pd.read_csv("outputs/index/val.csv")
    
    # Combine them for cross-validation
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)} samples")
    
    # Save temporary combined CSV for dataset loading
    combined_csv = "outputs/index/combined_train_val.csv"
    os.makedirs(os.path.dirname(combined_csv), exist_ok=True)
    combined_df.to_csv(combined_csv, index=False)
    
    # Create dataset from combined data
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    sel = cfg.get("metadata_features", {})
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)
    
    combined_ds = CadicaDataset(
        combined_csv,
        metadata,
        get_train_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    
    # Also load test set for final evaluation
    test_ds = CadicaDataset(
        "outputs/index/test.csv",
        metadata,
        get_val_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    
    return combined_ds, test_ds, combined_df

def train_fold(
    fold: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    combined_ds: CadicaDataset,
    cfg: dict,
    device: str,
    save_dir: str
) -> Tuple[float, dict]:
    """Train and evaluate a single fold"""
    
    # Create train/val subsets
    train_subset = Subset(combined_ds, train_indices)
    val_subset = Subset(combined_ds, val_indices)
    
    # Create augmented training dataset
    train_ds = AugmentedCadicaDataset(
        train_subset,
        minority_class=0,
        augment_factor=3
    )
    
    # Create balanced sampler
    train_sampler = create_balanced_sampler(train_ds)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Build model
    backbone = cfg.get("backbone", "resnet50").lower()
    fusion = cfg["fusion"]
    
    if fusion == "late":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            img_model = ResNetBackbone(pretrained=True).to(device)
            img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            img_model = ResNet18Backbone(pretrained=True).to(device)
            img_feat_dim = img_model.feat_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=combined_ds.tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
    
    elif fusion == "early":
        if backbone == "resnet50":
            from src.models.early_fusion_patient import ResNetEarlyFusionPatient
            img_model = ResNetEarlyFusionPatient(
                tab_dim=combined_ds.tab_dim,
                c_pat=cfg["pat_channels"],
                pretrained=True,
                out_dim=2
            ).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        clin_mlp = head = None
        params = list(img_model.parameters())
    
    else:
        raise ValueError("fusion must be 'late' or 'early'")
    
    # Optimizer
    opt_name = cfg["optimizer"].lower()
    if opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif opt_name in ("sgd", "sgdm"):
        optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    # Training loop
    best_val_acc = 0.0
    best_metrics = None
    patience = cfg.get("early_stopping_patience", 20)
    epochs_no_improve = 0
    fold_dir = os.path.join(save_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    ckpt_path = os.path.join(fold_dir, "best.pt")
    
    for epoch in range(cfg["epochs"]):
        print(f"\nFold {fold}, Epoch {epoch+1}/{cfg['epochs']}")
        
        train_one_epoch(
            img_model, optimizer, train_loader, device,
            mode=fusion, clin_mlp=clin_mlp, late_head=head
        )
        
        frame_metrics, y, p, video_metrics, video_df = evaluate(
            img_model, val_loader, device,
            mode=fusion, clin_mlp=clin_mlp, late_head=head,
            also_video=True, video_reduce="all"
        )
        
        print(f"=== Fold {fold} Validation Metrics (Frame) ===")
        for k, v in frame_metrics.items():
            print(f"{k:10s}: {v:.4f}")
        print(f"=== Fold {fold} Validation Metrics (Video) ===")
        for k, v in video_metrics.items():
            if k == "_multi":
                continue
            print(f"{k:10s}: {v:.4f}")
        
        # Save best model
        if frame_metrics["bal_acc"] > best_val_acc:
            best_val_acc = frame_metrics["bal_acc"]
            best_metrics = {"frame": frame_metrics, "video": video_metrics}
            torch.save({
                "fold": fold,
                "model": img_model.state_dict(),
                "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
                "head": head.state_dict() if head else None,
                "metrics": best_metrics,
                "epoch": epoch
            }, ckpt_path)
            print(f"[ckpt] New best model saved for fold {fold} (bal_acc={best_val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[early stopping] No improvement for {epochs_no_improve} epochs")
            
        if patience and epochs_no_improve >= patience:
            print(f"[early stopping] Stopping fold {fold} training after {epoch+1} epochs")
            break
    
    return best_val_acc, best_metrics

def main():
    # Load config and setup
    cfg = yaml.safe_load(open("configs/default.yaml"))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and combine data
    combined_ds, test_ds, combined_df = load_all_data(cfg)
    
    # Setup cross-validation
    n_splits = 5  # 5-fold CV
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    
    # Get groups (patients) and labels for stratification
    groups = [combined_ds[i]['patient'] for i in range(len(combined_ds))]
    labels = [combined_ds[i]['y'].item() for i in range(len(combined_ds))]
    
    # Create output directory
    save_dir = f"outputs/checkpoints/cv_balanced_{cfg['backbone']}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Cross-validation loop
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(combined_ds)), labels, groups)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        # Train the fold
        best_val_acc, best_metrics = train_fold(
            fold=fold,
            train_indices=train_idx,
            val_indices=val_idx,
            combined_ds=combined_ds,
            cfg=cfg,
            device=device,
            save_dir=save_dir
        )
        
        # Store fold results
        row = {
            "fold": fold,
            "val_balanced_acc": best_val_acc,
            **{f"val_frame_{k}": float(v) for k, v in best_metrics["frame"].items()},
            **{f"val_video_{k}": float(v) for k, v in best_metrics["video"].items() if k != "_multi"},
        }
        mv = best_metrics["video"].get("_multi") if isinstance(best_metrics["video"], dict) else None
        if isinstance(mv, dict):
            for agg_name, md in mv.items():
                for mk, mvv in md.items():
                    row[f"val_video_{agg_name}_{mk}"] = float(mvv)
        cv_results.append(row)
    
    # Save cross-validation results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(save_dir, "cv_results.csv"), index=False)
    
    # Print CV summary
    print("\nCross-validation Results:")
    print(f"Mean Balanced Accuracy: {cv_df['val_balanced_acc'].mean():.4f} ± {cv_df['val_balanced_acc'].std():.4f}")
    
    # Final evaluation on test set using each fold's best model
    test_results = []
    
    for fold in range(n_splits):
        print(f"\nEvaluating fold {fold} model on test set...")
        
        # Load best model for this fold
        ckpt = torch.load(os.path.join(save_dir, f"fold_{fold}", "best.pt"))
        
        # Rebuild model and load weights
        backbone = cfg.get("backbone", "resnet50").lower()
        fusion = cfg["fusion"]
        
        if fusion == "late":
            if backbone == "resnet50":
                from src.models.resnet_backbone import ResNetBackbone
                img_model = ResNetBackbone(pretrained=True).to(device)
                img_feat_dim = 2048
                
                from src.models.late_fusion import TabularMLP, LateFusionClassifier
                clin_mlp = TabularMLP(in_dim=combined_ds.tab_dim, hid=64, out_dim=32).to(device)
                head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
                
                img_model.load_state_dict(ckpt["model"])
                clin_mlp.load_state_dict(ckpt["clin_mlp"])
                head.load_state_dict(ckpt["head"])
            
        elif fusion == "early":
            if backbone == "resnet50":
                from src.models.early_fusion_patient import ResNetEarlyFusionPatient
                img_model = ResNetEarlyFusionPatient(
                    tab_dim=combined_ds.tab_dim,
                    c_pat=cfg["pat_channels"],
                    pretrained=True,
                    out_dim=2
                ).to(device)
                img_model.load_state_dict(ckpt["model"])
                clin_mlp = head = None
        
        # Evaluate on test set
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

        frame_metrics, y, p, video_metrics, video_df = evaluate(
            img_model, test_loader, device,
            mode=fusion, clin_mlp=clin_mlp, late_head=head,
            also_video=True, video_reduce="all"
        )

        # Save video-level predictions
        os.makedirs(os.path.join(save_dir, f"fold_{fold}"), exist_ok=True)
        video_df.to_csv(os.path.join(save_dir, f"fold_{fold}", "test_video_probs.csv"), index=False)

        test_results.append({
            "fold": fold,
            **{f"test_frame_{k}": float(v) for k, v in frame_metrics.items()},
            **{f"test_video_{k}": float(v) for k, v in video_metrics.items()},
        })

        print(f"Fold {fold} Test Metrics (Frame):")
        for k, v in frame_metrics.items():
            print(f"{k:10s}: {v:.4f}")
        print(f"Fold {fold} Test Metrics (Video):")
        for k, v in video_metrics.items():
            if k == "_multi":
                continue
            print(f"{k:10s}: {v:.4f}")
    
    # Save and print test results
    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(save_dir, "test_results.csv"), index=False)
    
    print("\nFinal Test Results (across all folds):")
    for col in test_df.columns:
        if col != "fold":
            mean = test_df[col].mean()
            std = test_df[col].std()
            print(f"{col}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
