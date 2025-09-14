import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cadica.parser import build_index

ROOT = "data/CADICA"  # change if needed
df = build_index(ROOT)
print(df.head())
print("Columns:", df.columns.tolist())
print("Label counts:", df["label"].value_counts().to_dict())
print("Unique patients/videos:", df['patient'].nunique(), df['video'].nunique())