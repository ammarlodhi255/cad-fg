#!/usr/bin/env python3
import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dataio.dataset import CadicaDataset
from src.dataio.transforms import get_val_tf


SETS = {"development": "outputs/index/trainval_pool.csv", "test": "outputs/index/test_pristine.csv"}


def load_build_models():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_train_kfold.py")
    spec = importlib.util.spec_from_file_location("train_kfold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models


def make_dataset(csv, cfg):
    sel = cfg.get("metadata_features", {}) or {}
    return CadicaDataset(
        csv, cfg.get("metadata_path", "data/CADICA/metadata.xlsx"), get_val_tf(cfg.get("img_size", 224)),
        selected_features={"numeric": sel.get("numeric"), "categorical": sel.get("categorical")},
        text_features=sel.get("text", None),
        text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"))


def metrics(y, p, patients):
    yhat = (p >= 0.5).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    within = [roc_auc_score(y[patients == q], p[patients == q])
              for q in np.unique(patients) if len(np.unique(y[patients == q])) == 2]
    s = pd.Series(p)
    eta2 = 1 - ((s - s.groupby(patients).transform("mean")) ** 2).sum() / ((s - s.mean()) ** 2).sum()
    return {"auroc": roc_auc_score(y, p), "bal_acc": (sens + spec) / 2, "sensitivity": sens,
            "specificity": spec, "f1": f1, "acc": (tp + tn) / len(y),
            "within_patient_auroc": float(np.mean(within)) if within else float("nan"),
            "n_patients_mixed": len(within), "patient_variance_share": float(eta2)}


@torch.no_grad()
def predict(model, X, patients, meta, device, bs=128):
    probs = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device).float()
        clin = torch.tensor(np.stack([meta[int(q)] for q in patients[i:i + bs]]), dtype=torch.float32, device=device)
        _, logits = model(x, clin)
        probs.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


def derangement(ids, rng):
    while True:
        perm = rng.permutation(ids)
        if not np.any(perm == ids):
            return dict(zip(ids, perm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="final_full_train.pt of one or more fusion models")
    ap.add_argument("--out_dir", default="outputs/analysis")
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_models = load_build_models()

    cfg0 = torch.load(args.ckpt[0], map_location="cpu", weights_only=False).get("cfg") \
        or yaml.safe_load(open("configs/default.yaml"))
    data = {}
    for name, csv in SETS.items():
        ds = make_dataset(csv, cfg0)
        # images are cached once so that every condition sees identical inputs
        data[name] = {"ds": ds, "X": torch.stack([ds[i]["x"].float() for i in range(len(ds))]),
                      "y": ds.df["label"].to_numpy().astype(int),
                      "patients": ds.df["patient"].to_numpy().astype(int)}
        print(f"{name}: {len(ds)} frames, {len(np.unique(data[name]['patients']))} patients", flush=True)

    meta = data["test"]["ds"].meta
    names = data["test"]["ds"].get_feature_names()
    text_idx = np.array([n.startswith("text[") for n in names])
    dev_ids = np.unique(data["development"]["patients"])
    mean_vec = np.mean([meta[int(q)] for q in dev_ids], axis=0).astype(np.float32)

    rows = []
    for ck_path in args.ckpt:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        img, _, _, _ = build_models(backbone=ck["backbone"], fusion=ck["fusion"], tab_dim=int(ck["tab_dim"]),
                                    pat_channels=ck["cfg"].get("pat_channels", 4), device=args.device)
        img.load_state_dict(ck["model"]); img.eval()
        tag = os.path.basename(os.path.dirname(ck_path)) + "/" + os.path.basename(ck_path)
        rng = np.random.default_rng(args.seed)
        for split, d in data.items():
            ids = np.unique(d["patients"])
            # true: own metadata; constant: development mean; mask_*: one feature group set to its mean;
            # shuffled: metadata of another patient (derangements)
            conds = {"true": [meta],
                     "constant": [{int(q): mean_vec for q in ids}],
                     "mask_text": [{int(q): np.where(text_idx, mean_vec, meta[int(q)]) for q in ids}],
                     "mask_tabular": [{int(q): np.where(~text_idx, mean_vec, meta[int(q)]) for q in ids}],
                     "shuffled": []}
            for _ in range(args.n_perm):
                perm = derangement(ids, rng)
                conds["shuffled"].append({int(q): meta[int(perm[q])] for q in ids})
            for cond, metas in conds.items():
                ms = [metrics(d["y"], predict(img, d["X"], d["patients"], m, args.device), d["patients"]) for m in metas]
                row = {"checkpoint": tag, "set": split, "condition": cond, "n_runs": len(ms)}
                for k in ms[0]:
                    vals = [m[k] for m in ms]
                    row[k] = float(np.mean(vals))
                    if len(ms) > 1:
                        row[k + "_min"], row[k + "_max"] = float(np.min(vals)), float(np.max(vals))
                rows.append(row)
                print(f"{tag:45s} {split:11s} {cond:12s} AUROC {row['auroc']:.4f} BA {row['bal_acc']:.4f} "
                      f"within-AUROC {row['within_patient_auroc']:.4f} patient-var {row['patient_variance_share']:.3f}",
                      flush=True)
        del img

    with open(os.path.join(args.out_dir, "metadata_permutation.json"), "w") as f:
        json.dump({"n_perm": args.n_perm, "seed": args.seed, "feature_groups": {
            "text_dims": int(text_idx.sum()), "tabular_dims": int((~text_idx).sum())}, "rows": rows}, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metadata_permutation.csv"), index=False)


if __name__ == "__main__":
    main()
