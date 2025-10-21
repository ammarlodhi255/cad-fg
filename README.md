CADICA cross-validation utilities
=================================

This repository now includes scripts to create patient-stratified 5-fold cross-validation splits and to train/evaluate models (image-only, late fusion, early fusion) across those folds.

Quick start
- Build the frame index and the standard train/val/test split:
  - `python scripts/01_make_index.py`
- Create 5-fold patient-stratified splits under `outputs/folds/patient_stratified_5fold/`:
  - `python scripts/01b_make_patient_stratified_folds.py`
- Run k-fold training for chosen backbones and fusion modes (defaults shown):
  - `python scripts/02_train_kfold.py --cfg configs/default.yaml --fusions imageonly,late,early --backbones resnet50,resnet18,effnet_b0,convnext_tiny,inception_v3,densenet121,mobilenet_v2,vit_b16`

Notes
- Folds are stratified by patient-level labels (positive if any video positive) to avoid leakage and preserve class balance across folds.
- The CV trainer saves per-fold checkpoints and validation metrics (frame- and video-level) under `outputs/checkpoints/cv_runs/<backbone>_<fusion>/fold_k/` and aggregates summaries to `cv_val_folds.csv` and a global `cv_summary.csv`.
- Training uses the metadata configuration from `configs/default.yaml` (including optional text features if enabled) and balanced sampling on the training split.


Training kfold:
python scripts/02_train_kfold.py --cfg configs/default.yaml --fusions imageonly,late,early --backbones resnet50,resnet18,effnet_b0,convnext_tiny,inception_v3,densenet121,mobilenet_v2,vit_b16 