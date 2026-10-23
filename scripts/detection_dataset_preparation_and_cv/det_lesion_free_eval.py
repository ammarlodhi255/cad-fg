#!/usr/bin/env python3
# Requires ultralytics==8.4.14 (the version used to train the detectors).
import os
import json
import argparse
import yaml
from ultralytics import YOLO


NEG_CONF_THRESHOLDS = (0.25, 0.5)


def run_val(model, work_dir, folder, device):
    img_dir = os.path.join(work_dir, folder, "images")
    data_yaml = os.path.join(work_dir, f"{folder}.yaml")
    with open(data_yaml, "w") as f:
        yaml.dump({"train": img_dir, "val": img_dir, "names": {0: "lesion"}, "nc": 1}, f)
    m = model.val(data=data_yaml, project=os.path.join(work_dir, "runs"), name=folder,
                  exist_ok=True, save=False, plots=False, device=device, verbose=False)
    p, r = float(m.box.p[0]), float(m.box.r[0])
    return {"images": len(os.listdir(img_dir)), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r) if p + r > 0 else 0.0, 4),
            "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}


def count_false_positives(model, neg_dir, conf, device):
    # every box predicted on a lesion-free frame is a false positive
    per_patient, frames_with_fp, fp_boxes = {}, 0, 0
    files = sorted(os.listdir(neg_dir))
    for fn in files:
        n = len(model.predict(os.path.join(neg_dir, fn), conf=conf, device=device, verbose=False)[0].boxes)
        d = per_patient.setdefault(fn.split("_")[0], {"frames": 0, "frames_with_fp": 0, "fp_boxes": 0})
        d["frames"] += 1
        d["frames_with_fp"] += int(n > 0)
        d["fp_boxes"] += n
        frames_with_fp += int(n > 0)
        fp_boxes += n
    return {"frames": len(files), "frames_with_fp": frames_with_fp,
            "pct_frames_with_fp": round(100 * frames_with_fp / len(files), 2),
            "fp_boxes_per_frame": round(fp_boxes / len(files), 3), "per_patient": per_patient}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/final_det_models/patient/runs/train/weights/last.pt")
    ap.add_argument("--work_dir", required=True,
                    help="folder with pos/{images,labels} (annotated test frames), neg/images (lesion-free "
                         "keyframes of the test patients) and all/{images,labels} (both; labels of pos only)")
    ap.add_argument("--stage", choices=["repro", "all"], default="all")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = {"positive_only": run_val(model, args.work_dir, "pos", args.device)}
    if args.stage == "all":
        out["with_negatives"] = run_val(model, args.work_dir, "all", args.device)
        neg_dir = os.path.join(args.work_dir, "neg", "images")
        for conf in NEG_CONF_THRESHOLDS:
            out[f"negatives_conf{conf}"] = count_false_positives(model, neg_dir, conf, args.device)
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.work_dir, f"lesion_free_eval_{args.stage}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
