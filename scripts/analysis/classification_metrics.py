#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


FRAME = ["acc", "sens", "prec", "f1", "spec", "bal_acc", "auc", "mcc", "ap"]
SAVED = {"acc": "val_frame_acc", "sens": "val_frame_recall", "prec": "val_frame_precision",
         "f1": "val_frame_f1", "spec": "val_frame_specificity", "bal_acc": "val_frame_bal_acc",
         "auc": "val_frame_auc", "ap": "val_frame_ap"}
VIDEO = {"video_topk3_acc": "val_video_topk3_acc", "video_logit_mean_acc": "val_video_logit_mean_acc",
         "video_tsmooth_max_acc": "val_video_tsmooth_max_acc"}
BASELINES = ["Always-positive", "Prevalence-matched random", "Per-patient majority (oracle)"]


def mcc(tp, fp, tn, fn):
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def ms(values):
    # mean and SD in %, over the folds with a defined value
    v = 100 * np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std(ddof=1)), int(len(v))


def fold_counts(folds_root):
    out = []
    for i in range(5):
        tr = pd.read_csv(os.path.join(folds_root, f"fold_{i}", "train.csv"))
        va = pd.read_csv(os.path.join(folds_root, f"fold_{i}", "val.csv"))
        vid = lambda d: d.groupby(["patient", "video"])["label"].max()
        out.append({"P": int(va["label"].sum()), "N": int((va["label"] == 0).sum()),
                    "q": float(va["label"].mean()), "pi": float(tr["label"].mean()),
                    "qv": float(vid(va).mean()), "piv": float(vid(tr).mean()),
                    "n_val": int(len(va)), "val_patients": int(va["patient"].nunique()), "val": va})
    return out


def run_metrics(df, folds):
    df = df.sort_values("fold").reset_index(drop=True)
    per = {k: [] for k in FRAME + list(VIDEO)}
    worst = 0.0
    for _, r in df.iterrows():
        f = folds[int(r["fold"])]
        # confusion matrix reconstructed from the fold sizes and the saved sensitivity / specificity
        tp, tn = r["val_frame_recall"] * f["P"], r["val_frame_specificity"] * f["N"]
        fn, fp = f["P"] - tp, f["N"] - tn
        worst = max(worst, abs((tp + tn) / (f["P"] + f["N"]) - r["val_frame_acc"]))
        for k, col in SAVED.items():
            per[k].append(r[col])
        per["mcc"].append(mcc(tp, fp, tn, fn))
        for k, col in VIDEO.items():
            per[k].append(r[col] if col in r else np.nan)
    return per, worst


def baselines(folds):
    out = {b: {k: [] for k in FRAME + ["video_acc"]} for b in BASELINES}
    for f in folds:
        q, pi, qv, piv = f["q"], f["pi"], f["qv"], f["piv"]
        for k, v in dict(acc=q, sens=1, prec=q, f1=2 * q / (1 + q), spec=0, bal_acc=.5, mcc=0, auc=.5, ap=q,
                         video_acc=qv).items():
            out["Always-positive"][k].append(v)
        # expected values of a classifier that predicts "lesion" with the training-fold prevalence
        for k, v in dict(acc=q * pi + (1 - q) * (1 - pi), sens=pi, prec=q, f1=2 * q * pi / (q + pi), spec=1 - pi,
                         bal_acc=.5, mcc=0, auc=.5, ap=q, video_acc=qv * piv + (1 - qv) * (1 - piv)).items():
            out["Prevalence-matched random"][k].append(v)
        # oracle: every frame gets the majority label of its validation patient
        va = f["val"]
        frac = va.groupby("patient")["label"].transform("mean")
        pred = (frac >= 0.5).astype(int)
        y = va["label"].to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        vid = va.groupby(["patient", "video"])["label"].max().reset_index()
        vpred = (vid.groupby("patient")["label"].transform("mean") >= 0.5).astype(int)
        for k, v in dict(acc=(tp + tn) / len(y), sens=sens, prec=prec,
                         f1=2 * prec * sens / (prec + sens) if prec + sens else 0.0, spec=spec,
                         bal_acc=(sens + spec) / 2, mcc=mcc(tp, fp, tn, fn), auc=roc_auc_score(y, frac),
                         ap=average_precision_score(y, frac),
                         video_acc=float((vpred == vid["label"]).mean())).items():
            out["Per-patient majority (oracle)"][k].append(v)
    return out


def test_metrics(pred_csv):
    p = pd.read_csv(pred_csv)
    y, s = p["y_true"].astype(int).to_numpy(), p["p_lesion"].to_numpy()
    pr = (s >= 0.5).astype(int)
    tp, tn = int(((pr == 1) & (y == 1)).sum()), int(((pr == 0) & (y == 0)).sum())
    fp, fn = int(((pr == 1) & (y == 0)).sum()), int(((pr == 0) & (y == 1)).sum())
    sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
    v = {"acc": (tp + tn) / len(y), "sens": sens, "prec": prec, "f1": 2 * prec * sens / (prec + sens),
         "spec": spec, "bal_acc": (sens + spec) / 2, "auc": roc_auc_score(y, s), "mcc": mcc(tp, fp, tn, fn),
         "ap": average_precision_score(y, s)}
    return {k: 100 * float(x) for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="outputs/checkpoints/new")
    ap.add_argument("--folds_root", default="outputs/folds/patient_stratified_5fold")
    ap.add_argument("--test_pred", nargs="*", default=[],
                    help="name=path/to/frame_predictions.csv of final models on the held-out test set")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    folds = fold_counts(args.folds_root)
    rows, worst_acc = [], 0.0
    for path in sorted(glob.glob(os.path.join(args.runs_root, "*", "*", "cv_val_folds.csv"))):
        df = pd.read_csv(path)
        if len(df) != 5 or not set(SAVED.values()) <= set(df.columns):
            continue
        per, worst = run_metrics(df, folds)
        worst_acc = max(worst_acc, worst)
        row = {"run": os.path.relpath(os.path.dirname(path), args.runs_root).replace("\\", "/")}
        for k in FRAME + list(VIDEO):
            row[f"{k}_mean"], row[f"{k}_sd"], row[f"{k}_folds"] = ms(per[k])
        rows.append(row)
    print(f"{len(rows)} runs; max |reconstructed - saved accuracy| over all folds: {worst_acc:.2e}")
    if worst_acc > 1e-4:
        raise SystemExit("Stopping: fold counts do not match the saved results.")

    base = baselines(folds)
    for b in BASELINES:
        row = {"run": f"baseline: {b}"}
        for k in FRAME:
            row[f"{k}_mean"], row[f"{k}_sd"], row[f"{k}_folds"] = ms(base[b][k])
        for k in VIDEO:
            row[f"{k}_mean"], row[f"{k}_sd"], row[f"{k}_folds"] = ms(base[b]["video_acc"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "classification_metrics.csv"), index=False)

    res = {"folds": [{k: v for k, v in f.items() if k != "val"} for f in folds],
           "baselines_mean_sd": {b: {k: ms(v)[:2] for k, v in base[b].items()} for b in BASELINES},
           "test": {}}
    for item in args.test_pred:
        name, path = item.split("=", 1)
        res["test"][name] = test_metrics(path)
    with open(os.path.join(args.out_dir, "classification_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("Per-fold validation prevalence (%):", [round(100 * f["q"], 2) for f in folds],
          "| frames:", [f["n_val"] for f in folds], "| patients:", [f["val_patients"] for f in folds])
    print("Per-fold training prevalence (%):", [round(100 * f["pi"], 2) for f in folds])
    for b in BASELINES:
        print(f"{b:32s} " + "  ".join(f"{k} {m:.2f}+-{s:.2f}" for k, (m, s) in res["baselines_mean_sd"][b].items()))
    for name, v in res["test"].items():
        print(f"test {name:20s} " + "  ".join(f"{k} {x:.2f}" for k, x in v.items()))
    print(f"Saved classification_metrics.csv and classification_metrics.json in {args.out_dir}")


if __name__ == "__main__":
    main()
