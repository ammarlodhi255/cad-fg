#!/usr/bin/env python3
import os
import sys
import glob
import json
import time
import shutil
import argparse
import platform
import subprocess


MODELS = [("yolov5l-cls", "imageonly"), ("yolov5l_cls_xattn_mid", "early")]
RECIPE = ["--cfg", "configs/default.yaml",
          "--folds_root", "outputs/folds/patient_stratified_5fold",
          "--epochs", "100", "--patience", "20",
          "--select_domain", "frame", "--select_metric", "bal_acc",
          "--amp", "off"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="outputs/seed_repeats")
    ap.add_argument("--seeds", default="1337,1,2")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    assert os.path.isfile("scripts/02_train_kfold.py"), "run from the repository root"
    assert len(glob.glob("outputs/folds/patient_stratified_5fold/fold_*")) == 5, \
        "copy splits/classification/patient_stratified_5fold to outputs/folds/ first"
    os.makedirs(os.path.join(args.save_dir, "logs"), exist_ok=True)

    import torch
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    commands = {}

    for seed in [int(s) for s in args.seeds.split(",")]:
        for backbone, fusion in MODELS:
            name = f"{backbone}_{fusion}"
            dst = os.path.join(args.save_dir, f"seed_{seed}", name)
            if os.path.isfile(os.path.join(dst, "cv_val_folds.csv")):
                print(f"[skip] seed {seed} {name}: already done")
                continue
            outroot = os.path.join("outputs", "seed_runs", f"seed_{seed}")
            cmd = [sys.executable, "scripts/02_train_kfold.py", *RECIPE,
                   "--backbones", backbone, "--fusions", fusion, "--seed", str(seed),
                   "--num_workers", str(args.num_workers), "--outroot", outroot]
            commands[f"seed_{seed}/{name}"] = " ".join(cmd[1:])
            print(f"\n=== seed {seed} {name} ===\n{' '.join(cmd[1:])}", flush=True)
            t0 = time.time()
            log_path = os.path.join(args.save_dir, "logs", f"seed_{seed}_{name}.log")
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise SystemExit(f"run failed (exit {proc.returncode}); see {log_path}")
            src = os.path.join(outroot, name)
            os.makedirs(dst, exist_ok=True)
            for f in ["cv_val_folds.csv", "summary.txt"]:
                shutil.copy2(os.path.join(src, f), dst)
            for vb in glob.glob(os.path.join(src, "fold_*", "val_best.json")):
                fold = os.path.basename(os.path.dirname(vb))
                shutil.copy2(vb, os.path.join(dst, f"{fold}_val_best.json"))
            print(f"done in {(time.time() - t0) / 60:.1f} min -> {dst}", flush=True)

    with open(os.path.join(args.save_dir, "recipe.json"), "w") as f:
        json.dump({"environment": env, "recipe": RECIPE, "commands": commands}, f, indent=2)


if __name__ == "__main__":
    main()
