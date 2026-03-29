#!/usr/bin/env python3
"""
Train final models on combined train+val data (all folds) for each split type.
Test on the common held-out test set.

Compares three splitting methodologies:
- Patient-stratified
- Video-stratified  
- Random (no stratification)
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
from ultralytics import YOLO
import yaml


# =========================
# CONFIG
# =========================
BASE_DIR = Path("/cluster/home/ammaa/Downloads/meta-cadica")

# Fold directories and corresponding test sets for each split type
SPLIT_CONFIGS = {
    "patient_strat": {
        "folds": BASE_DIR / "data/det_folds",
        "test": BASE_DIR / "data/cadica_yolo_det/test",  # Patient-stratified test
    },
    "video_strat": {
        "folds": BASE_DIR / "data/det_video_stratified_folds",
        "test": BASE_DIR / "data/det_video_stratified_folds/test",  # Video-stratified test
    },
    "random_strat": {
        "folds": BASE_DIR / "data/det_no_patient_stratification",
        "test": BASE_DIR / "data/det_no_patient_stratification/test",  # Random test
    },
}

# Output directory
OUTPUT_DIR = BASE_DIR / "outputs/final_models"

# Training config
MODELS = ["yolov8x"]  # Can add more: ["yolov8x", "yolo11x", "yolo12x"]
EPOCHS = 100
BATCH = 16
IMGSZ = 640


# =========================
# COMBINE FOLDS
# =========================
def combine_all_folds(fold_base_dir: Path, output_dir: Path) -> Path:
    """
    Combine all train+inner_val+outer_val from all folds into one dataset.
    Returns path to combined trainval directory.
    """
    combined_dir = output_dir / "trainval"
    (combined_dir / "images").mkdir(parents=True, exist_ok=True)
    (combined_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    
    # Iterate through all folds
    for fold_dir in sorted(fold_base_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
            
        # Combine all splits: train, inner_val, outer_val
        for split in ["train", "inner_val", "outer_val"]:
            img_dir = fold_dir / split / "images"
            lbl_dir = fold_dir / split / "labels"
            
            if not img_dir.exists():
                continue
                
            # Copy all images and labels
            for img_path in img_dir.glob("*"):
                img_name = img_path.name
                lbl_name = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
                lbl_path = lbl_dir / lbl_name
                
                # Copy (skip if already exists from another fold/split)
                dst_img = combined_dir / "images" / img_name
                dst_lbl = combined_dir / "labels" / lbl_name
                
                if not dst_img.exists() and img_path.exists():
                    shutil.copy(img_path, dst_img)
                if not dst_lbl.exists() and lbl_path.exists():
                    shutil.copy(lbl_path, dst_lbl)
                    total_images += 1
    
    print(f"  Combined images: {total_images}")
    return combined_dir


# =========================
# CREATE YAML CONFIG
# =========================
def create_yaml_config(trainval_dir: Path, test_dir: Path, output_path: Path):
    """Create YOLO dataset YAML config."""
    config = {
        "path": str(trainval_dir.parent.absolute()),
        "train": str((trainval_dir / "images").absolute()),
        "val": str((trainval_dir / "images").absolute()),  # Same as train (no validation)
        "test": str((test_dir / "images").absolute()),
        "nc": 1,
        "names": ["lesion"],
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path


# =========================
# TRAIN MODEL
# =========================
def train_model(
    model_name: str,
    yaml_path: Path,
    output_dir: Path,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
):
    """Train YOLO model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Initialize model
    try:
        model = YOLO(f"{model_name}.pt")
    except Exception as e:
        print(f"❌ Model {model_name} not available: {e}")
        return None
    
    # Train
    start_time = datetime.now()
    
    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(output_dir.absolute()),
        name="train",
        exist_ok=True,
        save=True,
        patience=20,
    )
    
    train_time = (datetime.now() - start_time).total_seconds() / 3600
    
    # Load best model
    best_weights = output_dir / "train/weights/best.pt"
    if not best_weights.exists():
        print(f"❌ Best weights not found: {best_weights}")
        return None
    
    model = YOLO(str(best_weights))
    
    # Test on held-out test set
    print(f"\n{'='*60}")
    print(f"Testing {model_name} on held-out test set")
    print(f"{'='*60}")
    
    test_results = model.val(
        data=str(yaml_path.absolute()),
        split="test",
    )
    
    # Extract metrics
    metrics = {
        "model": model_name,
        "mAP50": float(test_results.box.map50),
        "mAP50-95": float(test_results.box.map),
        "precision": float(test_results.box.mp),
        "recall": float(test_results.box.mr),
        "f1_score": 2 * (test_results.box.mp * test_results.box.mr) / (test_results.box.mp + test_results.box.mr + 1e-9),
        "training_time_hours": train_time,
    }
    
    return metrics


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("="*70)
    print("Training Final Models on Combined Train+Val Data")
    print("="*70)
    
    all_results = []
    
    # Process each split type
    for split_type, config in SPLIT_CONFIGS.items():
        fold_dir = config["folds"]
        test_dir = config["test"]
        
        print(f"\n{'#'*70}")
        print(f"# Split Type: {split_type.upper()}")
        print(f"{'#'*70}")
        
        if not fold_dir.exists():
            print(f"⚠️  Fold directory not found: {fold_dir}")
            continue
        
        if not test_dir.exists():
            print(f"⚠️  Test directory not found: {test_dir}")
            continue
        
        # Create output directory for this split type
        split_output_dir = OUTPUT_DIR / split_type
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all folds
        print(f"\nCombining all folds from: {fold_dir}")
        combined_dir = combine_all_folds(fold_dir, split_output_dir)
        
        # Create YAML config with correct test set
        yaml_path = split_output_dir / "dataset.yaml"
        create_yaml_config(combined_dir, test_dir, yaml_path)
        print(f"  Created config: {yaml_path}")
        print(f"  Test set: {test_dir}")
        
        # Train each model
        for model_name in MODELS:
            model_output_dir = split_output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            metrics = train_model(
                model_name=model_name,
                yaml_path=yaml_path,
                output_dir=model_output_dir,
                epochs=EPOCHS,
                batch=BATCH,
                imgsz=IMGSZ,
            )
            
            if metrics:
                metrics["split_type"] = split_type
                all_results.append(metrics)
                
                # Save individual metrics
                metrics_path = model_output_dir / "test_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✅ Saved metrics: {metrics_path}")
    
    # =========================
    # SAVE COMPARISON
    # =========================
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save full results
        results_csv = OUTPUT_DIR / "final_results_comparison.csv"
        df.to_csv(results_csv, index=False)
        print(f"\n{'='*70}")
        print(f"✅ Final Results: {results_csv}")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        # Save summary by split type
        summary = df.groupby("split_type")[["mAP50", "mAP50-95", "f1_score", "precision", "recall"]].mean()
        summary_csv = OUTPUT_DIR / "split_type_summary.csv"
        summary.to_csv(summary_csv)
        print(f"\n{'='*70}")
        print(f"Split Type Summary:")
        print(f"{'='*70}")
        print(summary)
        
    print(f"\n{'='*70}")
    print("🎉 All training completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
