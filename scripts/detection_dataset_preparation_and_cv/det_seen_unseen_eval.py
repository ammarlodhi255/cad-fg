#!/usr/bin/env python3
# Requires ultralytics==8.4.14 (the version used to train the detectors).
import os
import re
import json
import argparse
import yaml
from ultralytics import YOLO


# held-out test result of the final video-split detector, used as a reproduction check
GATE = {"precision": 0.8567, "recall": 0.5272, "mAP50": 0.6393}


def patients(img_dir):
    return {int(re.match(r"p(\d+)_v", f).group(1)) for f in os.listdir(img_dir) if re.match(r"p(\d+)_v", f)}


def run_val(model, img_dir, name, work_dir):
    data_yaml = os.path.join(work_dir, f"{name}.yaml")
    with open(data_yaml, "w") as f:
        yaml.dump({"train": img_dir, "val": img_dir, "names": {0: "lesion"}, "nc": 1}, f)
    m = model.val(data=data_yaml, project=os.path.join(work_dir, "runs"), name=name, exist_ok=True,
                  save=False, plots=False, verbose=False)
    p, r = float(m.box.p[0]), float(m.box.r[0])
    return {"images": len(os.listdir(img_dir)), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0,
            "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/final_det_models/video/runs/train/weights/last.pt")
    ap.add_argument("--seen_dir", default="data/det_video_stratified_folds/test",
                    help="video-split test set: new videos of patients seen in training")
    ap.add_argument("--unseen_dir", default="data/det_patient_stratified_folds/test",
                    help="patient-split test set: patients never seen by the video-split detector")
    ap.add_argument("--pool_dir", default="data/det_video_stratified_folds/fold_1",
                    help="any fold of the video split, to check the training patients")
    ap.add_argument("--out_dir", default="outputs/analysis/seen_unseen")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pool = set()
    for part in ("train", "inner_val", "outer_val"):
        pool |= patients(os.path.join(args.pool_dir, part, "images"))
    seen, unseen = patients(os.path.join(args.seen_dir, "images")), patients(os.path.join(args.unseen_dir, "images"))
    assert seen <= pool, "seen-patient test frames must come from patients of the training pool"
    assert not (unseen & pool), "unseen-patient test frames must come from patients outside the training pool"

    model = YOLO(args.weights)
    out = {"A_seen_patients": run_val(model, os.path.join(args.seen_dir, "images"), "seen", args.out_dir),
           "B_unseen_patients": run_val(model, os.path.join(args.unseen_dir, "images"), "unseen", args.out_dir)}
    a = out["A_seen_patients"]
    out["gate_ok"] = all(abs(a[k] - v) < 0.002 for k, v in GATE.items())
    out["patients"] = {"A_seen_patients": sorted(seen), "B_unseen_patients": sorted(unseen)}
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.out_dir, "seen_unseen_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    if not out["gate_ok"]:
        print("WARNING: the seen-patient set does not reproduce the reported result; check the ultralytics version.")


if __name__ == "__main__":
    main()
