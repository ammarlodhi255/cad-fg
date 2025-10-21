import os, yaml, random, numpy as np, torch, pandas as pd
import sys
from typing import Dict, List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
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
    
    def __init__(self, base_dataset: CadicaDataset, minority_class: int = 0, augment_factor: int = 1):
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
        # Original samples + (augment_factor-1) additional samples for each minority sample
        return len(self.base_dataset) + len(self.minority_indices) * (self.augment_factor - 1)
    
    def __getitem__(self, idx):
        base_len = len(self.base_dataset)
        
        # Return original sample if idx is within base dataset
        if idx < base_len:
            return self.base_dataset[idx]
        
        # For additional samples, generate augmented versions
        aug_idx = (idx - base_len) // (self.augment_factor - 1)
        orig_idx = self.minority_indices[aug_idx]
        orig_sample = self.base_dataset[orig_idx]
        
        # Apply clinically appropriate augmentation to image
        img = orig_sample['x'].numpy().transpose(1, 2, 0)  # Convert to HWC for albumentations
        aug_data = self.minority_transform(image=img)
        aug_img = aug_data["image"]  # Already converted to tensor by ToTensorV2
        
        # Return augmented sample with same metadata
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
    
    # Calculate class weights
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    print(f"[sampler] Class counts: {class_counts}")
    print(f"[sampler] Class weights: {class_weights}")
    
    # Assign weights to samples
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

def main():
    # Load config and setup
    cfg = yaml.safe_load(open("configs/default.yaml"))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv, val_csv = "outputs/index/train.csv", "outputs/index/val.csv"
    metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")

    sel = cfg.get("metadata_features", {})
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)

    # Create base datasets
    base_train_ds = CadicaDataset(
        train_csv,
        metadata,
        get_train_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )
    
    val_ds = CadicaDataset(
        val_csv,
        metadata,
        get_val_tf(cfg["img_size"]),
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
    )

    print("[meta] Using features:", base_train_ds.get_feature_groups(), f"tab_dim={base_train_ds.tab_dim}")
    print("[meta] Expanded order:", base_train_ds.get_feature_names())

    # Create augmented training dataset with clinically appropriate augmentations
    train_ds = AugmentedCadicaDataset(
        base_train_ds,
        minority_class=0,  # Class 0 is the minority (1,730 vs 3,624 samples)
        augment_factor=2   # Creates 2 additional samples for each minority class sample
    )

    # Create balanced sampler for the augmented dataset
    train_sampler = create_balanced_sampler(train_ds)

    # Create data loaders
    tr = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    va = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # Model setup and training
    fusion = cfg["fusion"]
    os.makedirs("outputs/checkpoints", exist_ok=True)
    backbone = cfg.get("backbone", "resnet50").lower()

    if fusion == "late":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            img_model = ResNetBackbone(pretrained=True).to(device)
            img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            img_model = ResNet18Backbone(pretrained=True).to(device)
            img_feat_dim = img_model.feat_dim
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            img_model = EfficientNetB0Backbone(pretrained=True).to(device)
            img_feat_dim = img_model.feat_dim
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtBackbone
            img_model = ConvNeXtBackbone(variant="base", pretrained=True).to(device)
            img_feat_dim = img_model.feat_dim  # 1024
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=train_ds.base_dataset.tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())

    elif fusion == "early":
        if backbone == "resnet50":
            from src.models.early_fusion_patient import ResNetEarlyFusionPatient
            img_model = ResNetEarlyFusionPatient(
                tab_dim=train_ds.base_dataset.tab_dim,
                c_pat=cfg["pat_channels"],
                pretrained=True,
                out_dim=2
            ).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18EarlyFusionPatient
            img_model = ResNet18EarlyFusionPatient(
                tab_dim=train_ds.base_dataset.tab_dim,
                c_pat=cfg["pat_channels"],
                pretrained=True,
                out_dim=2
            ).to(device)
        elif backbone == "convnext_base":
            from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
            img_model = ConvNeXtEarlyFusionPatient(
                variant="base", tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
                pretrained=True, out_dim=2
            ).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        clin_mlp = head = None
        params = list(img_model.parameters())

    else:
        raise ValueError("fusion must be 'late' or 'early'")

    # Optimizer setup
    opt_name = cfg["optimizer"].lower()
    if opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif opt_name in ("sgd", "sgdm"):
        optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Training loop
    ROOT_DIR = f"outputs/checkpoints/balanced_augmented-{backbone}"
    os.makedirs(ROOT_DIR, exist_ok=True)
    best = 0.0
    ckpt_path = f"{ROOT_DIR}/best_{backbone}_{fusion}.pt"
    patience = cfg.get("early_stopping_patience", None)
    epochs_no_improve = 0

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

        train_one_epoch(
            img_model, optimizer, tr, device,
            mode=fusion, clin_mlp=clin_mlp, late_head=head
        )

        metrics, y, p = evaluate(
            img_model, va, device,
            mode=fusion, clin_mlp=clin_mlp, late_head=head
        )
        
        print("=== Evaluation metrics ===")
        for k, v in metrics.items():
            print(f"{k:10s}: {v:.4f}")

        if metrics["bal_acc"] > best:
            best = metrics["bal_acc"]
            torch.save({
                "fusion": fusion,
                "tab_dim": train_ds.base_dataset.tab_dim,
                "cfg": {
                    **cfg,
                    "metadata_features_used": {
                        **train_ds.base_dataset.get_feature_groups(),
                        "text": text_cols,
                    },
                },
                "model": img_model.state_dict(),
                "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
                "head": head.state_dict() if head else None,
            }, ckpt_path)
            print(f"[ckpt] new best saved ({best:.3f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[early stopping] no improvement for {epochs_no_improve} epochs")

        if patience and epochs_no_improve >= patience:
            print(f"[early stopping] stopping early after {epoch+1} epochs")
            break

    print(f"Best val balanced acc: {best:.3f} -> {ckpt_path}")

if __name__ == "__main__":
    main()