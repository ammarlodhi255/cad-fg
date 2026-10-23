#!/usr/bin/env python3
# Requires ultralytics==8.4.14 (the version used to train the detectors).
import os
import re
import json
import argparse
import numpy as np
from ultralytics import YOLO


def frames(dirs):
    paths, pat, vid = [], [], []
    for d in dirs:
        img_dir = os.path.join(d, "images")
        for f in sorted(os.listdir(img_dir)):
            m = re.match(r"p(\d+)_v(\d+)_", f)
            if m:
                paths.append(os.path.join(img_dir, f))
                pat.append(int(m.group(1)))
                vid.append((int(m.group(1)), int(m.group(2))))
    return paths, np.array(pat), vid


def embed(weights, paths, chunk=32):
    # global-pooled features of the frozen network, L2-normalized
    model = YOLO(weights)
    out = []
    for i in range(0, len(paths), chunk):
        out += [e.cpu().numpy().ravel() for e in model.embed(paths[i:i + chunk], imgsz=640, verbose=False)]
    x = np.stack(out)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def retrieval(x, pat, vid):
    # nearest neighbour among frames of OTHER videos; hit if it belongs to the query's patient
    vid = np.array([hash(v) for v in vid])
    sim = x @ x.T
    hits, chance = [], []
    for i in range(len(x)):
        other = vid != vid[i]
        same_pat_other = other & (pat == pat[i])
        if not same_pat_other.any():
            continue
        j = np.where(other)[0][np.argmax(sim[i, other])]
        hits.append(pat[j] == pat[i])
        chance.append(same_pat_other.sum() / other.sum())
    return {"queries": len(hits), "top1_same_patient": round(100 * float(np.mean(hits)), 2),
            "chance": round(100 * float(np.mean(chance)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["data/det_video_stratified_folds/test", "data/det_patient_stratified_folds/test"])
    ap.add_argument("--weights", default="outputs/final_det_models/video/runs/train/weights/last.pt")
    ap.add_argument("--coco_weights", default="yolo26x.pt")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths, pat, vid = frames(args.sets)
    out = {"frames": len(paths), "patients": int(len(set(pat))), "videos": len(set(vid))}
    for name, w in {"video-split YOLO26x (CADICA)": args.weights, "COCO-pretrained YOLO26x": args.coco_weights}.items():
        out[name] = retrieval(embed(w, paths), pat, vid)
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.out_dir, "patient_identity_retrieval.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
