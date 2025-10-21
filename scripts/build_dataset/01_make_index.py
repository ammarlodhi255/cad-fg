import os, pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cadica.parser import build_index
from src.cadica.splits import split_by_patient

ROOT = "data/CADICA"
os.makedirs("outputs/index", exist_ok=True)

df = build_index(ROOT)

# drop exact duplicates just in case
df = df.drop_duplicates(subset=["patient","video","frame","img_path","label"]).reset_index(drop=True)

df.to_csv("outputs/index/all_keyframes.csv", index=False)
tr, va, te = split_by_patient(df, train=.8, val=.1, seed=1337)
tr.to_csv("outputs/index/train.csv", index=False)
va.to_csv("outputs/index/val.csv", index=False)
te.to_csv("outputs/index/test.csv", index=False)
print("[splits] train/val/test:", len(tr), len(va), len(te))