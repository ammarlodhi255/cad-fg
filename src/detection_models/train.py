#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.detection_models.build_yolo_detection_dataset import (  # type: ignore
    build_detection_index,
    collect_classes,
)
from src.dataio.det_dataset import CadicaDetDataset  # type: ignore
from src.dataio.dataset import load_metadata_any  # type: ignore
from src.cadica.splits import split_by_patient  # type: ignore
from src.detection_models.yolov5_det_mid_fusion import YOLOv5DetMidFusionCLSCrossAttn  # type: ignore


def set_seed(s: int) -> None:
    import random
    import torch

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_folds_by_patient(
    df: pd.DataFrame,
    n_splits: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Use GroupKFold on patient IDs to build CV splits."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["patient"].values
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)
        folds.append((df_tr, df_va))
    return folds


def build_image_only_datasets(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    metadata_path: str,
    cfg: Dict,
    class_map: Dict[str, int],
    single_class: bool,
):
    """
    For image-only detection, we still use CadicaDetDataset but ignore the metadata
    at model level. This keeps indexing, augmentation, and GT handling unified.
    """
    from src.dataio.transforms import get_train_tf_basic, get_val_tf  # lazy import

    sel = cfg.get("metadata_features", {}) or {}
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)
    img_size = cfg.get("img_size", 640)

    tf_train = get_train_tf_basic(img_size)
    tf_val = get_val_tf(img_size)

    # For image-only we can fit metadata stats on train patients, but we won't use clin in the model.
    fit_pids = sorted(df_tr["patient"].unique().tolist())

    train_ds = CadicaDetDataset(
        df_tr,
        metadata_path=metadata_path,
        tfm=tf_train,
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
        fit_patients=fit_pids,
        stats=None,
        class_map=class_map,
        single_class=single_class,
    )

    # reuse stats for val to avoid leakage
    val_ds = CadicaDetDataset(
        df_va,
        metadata_path=metadata_path,
        tfm=tf_val,
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
        fit_patients=fit_pids,
        stats=train_ds.stats,
        class_map=class_map,
        single_class=single_class,
    )
    return train_ds, val_ds


def train_image_only_fold(
    fold_id: int,
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    out_root: str,
    cfg: Dict,
    class_map: Dict[str, int],
    single_class: bool,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
):
    """
    Image-only baseline using Ultralytics YOLO training API.
    We still use the detection index, but rely on build_yolo_detection_dataset
    to materialize fold-specific images/labels + YAML.
    """
    from src.detection_models.build_yolo_detection_dataset import write_split, write_yaml  # type: ignore

    if YOLO is None:
        raise ImportError("ultralytics package is required. Install via 'pip install ultralytics'.")

    fold_root = os.path.join(out_root, f"fold{fold_id}")
    os.makedirs(fold_root, exist_ok=True)

    write_split(df_tr, "train", fold_root, class_map, single_class=single_class)
    write_split(df_va, "val", fold_root, class_map, single_class=single_class)
    yaml_path = write_yaml(
        fold_root,
        class_map,
        single_class=single_class,
        yaml_name=f"cadica_fold{fold_id}.yaml",
    )

    model = YOLO("yolov5l.pt")

    run_name = f"fold{fold_id}_image_only"
    print(f"[train_image_only_fold] Starting training for {run_name} with data={yaml_path}")

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=os.path.join(out_root, "runs"),
        name=run_name,
    )


def train_cross_attn_fold(
    fold_id: int,
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    out_root: str,
    cfg: Dict,
    class_map: Dict[str, int],
    single_class: bool,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
):
    """
    Cross-attention fusion training using local YOLOv5 loss implementation.
    This sets up:
      - CadicaDetDataset with metadata vectors respecting configs/default.yaml.
      - YOLOv5DetMidFusionCLSCrossAttn model with tab_dim matching dataset.tab_dim.
      - YOLOv5 ComputeLoss on the underlying YOLOv5 detection model.
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # ---- Import YOLOv5 local loss ----
    # We use torch.hub to load the base model (already working),
    # but import ComputeLoss from local YOLOv5 repo for training
    yv5_root = os.path.join(os.path.dirname(__file__), "yolov5")
    yv5_root = os.path.abspath(yv5_root)
    if not os.path.isdir(yv5_root):
        raise FileNotFoundError(
            f"YOLOv5 repo not found at {yv5_root}. "
            "Please ensure yolov5 is cloned to src/detection_models/yolov5/"
        )
    
    # Add YOLOv5 root to path (YOLOv5 uses relative imports)
    if yv5_root not in sys.path:
        sys.path.insert(0, yv5_root)
    
    # Import ComputeLoss - it needs utils.metrics and utils.torch_utils which use relative imports
    try:
        from utils.loss import ComputeLoss  # type: ignore
    except ImportError as e:
        # Provide helpful error message
        raise ImportError(
            f"Failed to import ComputeLoss from YOLOv5 at {yv5_root}. "
            f"Error: {e}. "
            "Ensure: "
            "1) yolov5 repo is properly cloned to src/detection_models/yolov5/ "
            "2) ultralytics is installed: pip install ultralytics"
        ) from e

    from src.dataio.transforms import get_train_tf_basic, get_val_tf  # type: ignore

    metadata_path = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    sel = cfg.get("metadata_features", {}) or {}
    sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
    text_cols = sel.get("text", None)

    tf_train = get_train_tf_basic(imgsz)
    tf_val = get_val_tf(imgsz)

    fit_pids = sorted(df_tr["patient"].unique().tolist())

    train_ds = CadicaDetDataset(
        df_tr,
        metadata_path=metadata_path,
        tfm=tf_train,
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
        fit_patients=fit_pids,
        stats=None,
        class_map=class_map,
        single_class=single_class,
    )

    val_ds = CadicaDetDataset(
        df_va,
        metadata_path=metadata_path,
        tfm=tf_val,
        selected_features=sel_nc,
        text_features=text_cols,
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
        fit_patients=fit_pids,
        stats=train_ds.stats,
        class_map=class_map,
        single_class=single_class,
    )

    def collate_fn(batch: List[Dict[str, Any]]):
        imgs = torch.stack([b["x"] for b in batch], dim=0)
        clins = torch.stack([b["clin"] for b in batch], dim=0)
        targets_list = []
        for i, b in enumerate(batch):
            t = b["targets"]
            if t.numel():
                t = t.clone()
                t[:, 0] = i  # set image index
                targets_list.append(t)
        if targets_list:
            targets = torch.cat(targets_list, dim=0)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)
        return imgs, clins, targets, batch

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    tab_dim = int(train_ds.tab_dim)

    # Load YOLOv5l base model via torch.hub (same as fusion wrapper uses)
    import types
    if not hasattr(torch, "hub") or not isinstance(torch.hub, types.ModuleType):
        raise ImportError("torch.hub is required to load YOLOv5 detection models")
    
    # Try local file first, then hub
    w_path = "yolov5l.pt"
    if not os.path.isfile(w_path):
        alt = os.path.join(yv5_root, "yolov5l.pt")
        if os.path.isfile(alt):
            w_path = alt
    
    if os.path.isfile(w_path):
        base_model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=w_path,
            autoshape=False,
            trust_repo=True,
            verbose=False,
        )
    else:
        # Fallback to hub pretrained
        base_model = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5l",
            pretrained=True,
            autoshape=False,
            trust_repo=True,
            verbose=False,
        )
    
    # torch.hub.load returns a YOLOv5 wrapper; extract the actual model
    # The wrapper has .model attribute which is the BaseModel
    hub_wrapper = base_model
    actual_model = base_model.model if hasattr(base_model, "model") else base_model
    actual_model = actual_model.to(device)
    
    # For ComputeLoss, we need the model that has .hyp and .model[-1] (Detect head)
    # ComputeLoss expects the BaseModel (actual_model), not the wrapper
    # But we also need to ensure it has .hyp attribute
    if not hasattr(actual_model, "hyp"):
        # Try to get hyp from wrapper or set defaults
        if hasattr(hub_wrapper, "hyp"):
            actual_model.hyp = hub_wrapper.hyp
        else:
            # Use default YOLOv5 hyperparameters
            actual_model.hyp = {
                "lr0": 0.01, "lrf": 0.1, "momentum": 0.937, "weight_decay": 0.0005,
                "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
                "box": 0.05, "cls": 0.5, "cls_pw": 1.0, "obj": 1.0, "obj_pw": 1.0,
                "iou_t": 0.20, "anchor_t": 4.0, "fl_gamma": 0.0, "hsv_h": 0.015,
                "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1,
                "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
                "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
                "label_smoothing": 0.0,
            }

    model = YOLOv5DetMidFusionCLSCrossAttn(
        model_name="yolov5l",
        tab_dim=tab_dim,
        pretrained=False,
        attn_dim=256,
        drop=0.1,
        nsites=2,
        base_model=actual_model,
    )
    model = model.to(device)
    
    # Ensure model is in training mode for proper forward pass
    model.train()

    # YOLOv5 loss on the underlying detection model
    criterion = ComputeLoss(actual_model)  # type: ignore
    
    # Import validation utilities for mAP computation
    from utils.metrics import ap_per_class, ConfusionMatrix  # type: ignore
    from utils.general import non_max_suppression, xywh2xyxy  # type: ignore

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    fold_root = os.path.join(out_root, f"fold{fold_id}_cross_attn")
    os.makedirs(fold_root, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_lbox = 0.0
        train_lobj = 0.0
        train_lcls = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_id} [cross_attn] Epoch {epoch+1}/{epochs}")
        for imgs, clins, targets, _meta in pbar:
            imgs = imgs.to(device, non_blocking=True)
            clins = clins.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            # YOLOv5 returns feature maps in training mode (list of [B,C,H,W] tensors)
            preds = model(imgs, clins)
            # Ensure preds is a list (YOLOv5 training mode returns list of feature maps)
            if not isinstance(preds, (list, tuple)):
                preds = [preds] if torch.is_tensor(preds) else preds
            loss, loss_items = criterion(preds, targets)  # type: ignore
            loss.backward()
            optimizer.step()

            # Extract individual loss components
            lbox, lobj, lcls = loss_items[0].item(), loss_items[1].item(), loss_items[2].item()
            train_loss += float(loss.detach().cpu())
            train_lbox += lbox
            train_lobj += lobj
            train_lcls += lcls
            n_batches += 1
            
            # Update progress bar with losses
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{lbox:.4f}',
                'obj': f'{lobj:.4f}',
                'cls': f'{lcls:.4f}'
            })

        train_loss /= max(1, n_batches)
        train_lbox /= max(1, n_batches)
        train_lobj /= max(1, n_batches)
        train_lcls /= max(1, n_batches)

        # Validation with mAP computation
        model.train()  # Need train mode for feature maps (loss computation)
        val_loss = 0.0
        val_lbox = 0.0
        val_lobj = 0.0
        val_lcls = 0.0
        n_val_batches = 0
        
        # For mAP computation
        stats = []
        seen = 0
        names = {k: v for k, v in enumerate(class_map.keys())} if not single_class else {0: "lesion"}
        
        print(f"\n{'='*80}")
        print(f"Validating fold {fold_id} epoch {epoch+1}/{epochs}...")
        
        with torch.no_grad():
            for batch_idx, (imgs, clins, targets, _meta) in enumerate(val_loader):
                imgs = imgs.to(device, non_blocking=True)
                clins = clins.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass for loss
                preds = model(imgs, clins)
                if not isinstance(preds, (list, tuple)):
                    preds = [preds] if torch.is_tensor(preds) else preds
                loss, loss_items = criterion(preds, targets)  # type: ignore
                
                # Extract individual loss components
                lbox, lobj, lcls = loss_items[0].item(), loss_items[1].item(), loss_items[2].item()
                val_loss += float(loss.detach().cpu())
                val_lbox += lbox
                val_lobj += lobj
                val_lcls += lcls
                n_val_batches += 1
                
                # For mAP: run inference (NMS) on predictions
                # Convert feature maps to detections
                model.eval()  # Switch to eval for NMS
                with torch.no_grad():
                    # Run through model again in eval mode to get detections
                    detections = model(imgs, clins)
                    # If detections is a tuple/list, take first element (detections)
                    if isinstance(detections, (list, tuple)):
                        detections = detections[0] if len(detections) > 0 else detections
                    
                    # Apply NMS
                    detections = non_max_suppression(
                        detections,
                        conf_thres=0.001,
                        iou_thres=0.6,
                        max_det=300,
                    )
                
                model.train()  # Back to train mode
                
                # Process targets and predictions for mAP
                for i, det in enumerate(detections):
                    seen += 1
                    if targets.shape[0] == 0:
                        if det is not None and len(det):
                            # No targets but detections found
                            stats.append((torch.zeros(0, 5, device=device), torch.Tensor(), torch.Tensor(), torch.Tensor()))
                        continue
                    
                    # Get targets for this image
                    img_targets = targets[targets[:, 0] == i]
                    if len(img_targets) == 0:
                        if det is not None and len(det):
                            stats.append((torch.zeros(0, 5, device=device), torch.Tensor(), torch.Tensor(), torch.Tensor()))
                        continue
                    
                    # Convert targets from [img_idx, cls, xc, yc, w, h] to [cls, x1, y1, x2, y2]
                    target_cls = img_targets[:, 1].long()
                    target_boxes = img_targets[:, 2:6]  # normalized xywh
                    # Convert xywh to xyxy
                    target_boxes_xyxy = torch.zeros_like(target_boxes)
                    target_boxes_xyxy[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2  # x1
                    target_boxes_xyxy[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2  # y1
                    target_boxes_xyxy[:, 2] = target_boxes[:, 0] + target_boxes[:, 2] / 2  # x2
                    target_boxes_xyxy[:, 3] = target_boxes[:, 1] + target_boxes[:, 3] / 2  # y2
                    
                    # Scale to image size
                    h, w = imgs[i].shape[1:3]
                    target_boxes_xyxy[:, [0, 2]] *= w
                    target_boxes_xyxy[:, [1, 3]] *= h
                    
                    # Process detections
                    if det is not None and len(det):
                        # det is [x1, y1, x2, y2, conf, cls]
                        det_boxes = det[:, :4]
                        det_conf = det[:, 4]
                        det_cls = det[:, 5].long()
                    else:
                        det_boxes = torch.zeros(0, 4, device=device)
                        det_conf = torch.zeros(0, device=device)
                        det_cls = torch.zeros(0, dtype=torch.long, device=device)
                    
                    stats.append((target_boxes_xyxy.cpu(), target_cls.cpu(), det_boxes.cpu(), det_conf.cpu(), det_cls.cpu()))

        val_loss /= max(1, n_val_batches)
        val_lbox /= max(1, n_val_batches)
        val_lobj /= max(1, n_val_batches)
        val_lcls /= max(1, n_val_batches)
        
        # Compute mAP using YOLOv5's utilities
        map50 = 0.0
        map50_95 = 0.0
        if len(stats) > 0:
            try:
                # Collect all targets and predictions
                target_boxes_list = []
                target_cls_list = []
                det_boxes_list = []
                det_conf_list = []
                det_cls_list = []
                
                for s in stats:
                    if s[0].numel() > 0:
                        target_boxes_list.append(s[0])
                        target_cls_list.append(s[1])
                    if s[2].numel() > 0:
                        det_boxes_list.append(s[2])
                        det_conf_list.append(s[3])
                        det_cls_list.append(s[4])
                
                if len(target_boxes_list) > 0 and len(det_boxes_list) > 0:
                    all_target_boxes = torch.cat(target_boxes_list, dim=0)
                    all_target_cls = torch.cat(target_cls_list, dim=0)
                    all_det_boxes = torch.cat(det_boxes_list, dim=0)
                    all_det_conf = torch.cat(det_conf_list, dim=0)
                    all_det_cls = torch.cat(det_cls_list, dim=0)
                    
                    # Compute mAP using YOLOv5's ap_per_class
                    iouv = torch.linspace(0.5, 0.95, 10)  # IoU thresholds
                    from utils.metrics import box_iou  # type: ignore
                    
                    # Match detections to targets
                    correct = torch.zeros(len(all_det_boxes), len(iouv), dtype=torch.bool)
                    if len(all_target_boxes) > 0:
                        ious = box_iou(all_det_boxes, all_target_boxes)
                        for i in range(len(all_det_boxes)):
                            cls = all_det_cls[i]
                            # Find targets of same class
                            same_cls = all_target_cls == cls
                            if same_cls.any():
                                iou_vals = ious[i][same_cls]
                                if len(iou_vals) > 0:
                                    best_iou = iou_vals.max()
                                    correct[i] = best_iou > iouv
                    
                    # Compute AP per class (YOLOv5 returns precision, recall, ap, f1, ap_class)
                    _, _, ap, _, ap_class = ap_per_class(
                        correct.numpy(),
                        all_det_conf.numpy(),
                        all_det_cls.numpy(),
                        all_target_cls.numpy(),
                        plot=False,
                        save_dir=None,
                        names=names,
                    )
                    map50 = float(ap[:, 0].mean()) if len(ap) > 0 else 0.0
                    map50_95 = float(ap.mean()) if len(ap) > 0 else 0.0
            except Exception as e:
                import traceback
                print(f"[warn] Could not compute mAP: {e}")
                traceback.print_exc()
                map50 = 0.0
                map50_95 = 0.0

        # Print formatted output similar to YOLOv5
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'Train:':<12} {'loss':<10} {'box':<10} {'obj':<10} {'cls':<10}")
        print(f"{'':<12} {train_loss:<10.4f} {train_lbox:<10.4f} {train_lobj:<10.4f} {train_lcls:<10.4f}")
        print(f"{'Val:':<12} {'loss':<10} {'box':<10} {'obj':<10} {'cls':<10}")
        print(f"{'':<12} {val_loss:<10.4f} {val_lbox:<10.4f} {val_lobj:<10.4f} {val_lcls:<10.4f}")
        print(f"{'Metrics:':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<10}")
        print(f"{'':<12} {map50:<10.4f} {map50_95:<10.4f}")
        print(f"{'='*80}\n")

        # save best model per fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(fold_root, "best_cross_attn.pt")
            torch.save({"model": model.state_dict(), "val_loss": val_loss}, ckpt_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="5-fold CV training for YOLOv5l detection on CADICA.")
    ap.add_argument(
        "--cadica-root",
        type=str,
        default="data/CADICA",
        help="Path to CADICA root folder (containing metadata.xlsx and selectedVideos/).",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="runs/det_yolov5l_cv",
        help="Output root for fold datasets and training runs.",
    )
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    ap.add_argument("--fusion", type=str, default="image_only", choices=["image_only", "cross_attn"])
    ap.add_argument("--single-class", action="store_true", help="Collapse all lesion categories into a single class.")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs per fold.")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for YOLO training.")
    ap.add_argument("--device", type=str, default="cuda", help="Device for training, e.g. 'cuda', 'cpu', '0', '0,1'.")
    ap.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Build global detection index (only frames with GT, including patient ids)
    df = build_detection_index(args.cadica_root)

    # Class mapping
    if args.single_class:
        class_map = {"lesion": 0}
    else:
        class_map = collect_classes(df)
        if not class_map:
            print("[warn] No classes discovered; falling back to single-class lesion.")
            class_map = {"lesion": 0}
            args.single_class = True

    print(f"[main] Classes: {class_map}")

    # Load config for metadata settings (default.yaml)
    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default.yaml")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    # Patient-wise GroupKFold
    folds = build_folds_by_patient(df, n_splits=args.folds)
    os.makedirs(args.out_root, exist_ok=True)

    for k, (df_tr, df_va) in enumerate(folds):
        print(
            f"[main] Fold {k}: train frames={len(df_tr)} (patients={df_tr['patient'].nunique()}), "
            f"val frames={len(df_va)} (patients={df_va['patient'].nunique()})"
        )
        if args.fusion == "image_only":
            train_image_only_fold(
                fold_id=k,
                df_tr=df_tr,
                df_va=df_va,
                out_root=args.out_root,
                cfg=cfg,
                class_map=class_map,
                single_class=args.single_class,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
            )
        elif args.fusion == "cross_attn":
            train_cross_attn_fold(
                fold_id=k,
                df_tr=df_tr,
                df_va=df_va,
                out_root=args.out_root,
                cfg=cfg,
                class_map=class_map,
                single_class=args.single_class,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
            )


if __name__ == "__main__":
    main()


