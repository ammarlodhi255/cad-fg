#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dataio.dataset import (CAT_VALS, load_metadata_any, _prepare_meta_base,
                                compute_meta_stats, build_meta_vectors_from_stats)


# metadata features used by the models (configs/default.yaml)
USED_NUM = ["age", "height", "weight", "bmi", "evol_dm_years", "lvef_pct"]
USED_CAT = ["sex", "diabetes", "dyslipidemia", "smoker", "hypertension",
            "kidney_failure", "heart_failure", "atrial_fib"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/CADICA/metadata.xlsx")
    ap.add_argument("--out_dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_excel = pd.read_excel(args.metadata)
    n = len(raw_excel)
    results = {"metadata_file": os.path.abspath(args.metadata), "n_patients": n}

    print(f"Patients: {n}\n\nMissing values per variable (raw file)")
    raw_missing = {c: int(raw_excel[c].isna().sum()) for c in raw_excel.columns}
    for c, m in raw_missing.items():
        print(f"   {c:52s} {m:3d}  ({100 * m / n:5.1f}%)")
    results["missing_raw_excel"] = raw_missing

    meta = load_metadata_any(args.metadata)
    prepared = _prepare_meta_base(meta)
    print("\nNumeric model features after preprocessing")
    model_missing = {}
    for k in USED_NUM:
        model_missing[k] = int(prepared[k].isna().sum())
        print(f"   {k:15s} {model_missing[k]:3d}  ({100 * model_missing[k] / n:5.1f}%)")
    results["missing_after_preprocessing_numeric"] = model_missing

    zeros = {k: int((prepared[k] == 0).sum()) for k in USED_NUM}
    n_non_diabetic = int((meta["diabetes"] == "no").sum())
    print("\nRecorded zeros among the numeric features:", zeros, f"(non-diabetic patients: {n_non_diabetic})")
    results["recorded_zeros_numeric"] = zeros
    results["n_non_diabetic"] = n_non_diabetic

    # a missing continuous value is set to 0 after robust standardization, i.e. to the training median
    stats = compute_meta_stats(meta, selected_features={"numeric": USED_NUM, "categorical": USED_CAT})
    stats["use_num"], stats["use_cat"] = USED_NUM, USED_CAT
    vectors = build_meta_vectors_from_stats(meta, stats=stats)
    i_height = USED_NUM.index("height")
    pid_missing = int(prepared.loc[prepared["height"].isna(), "patient"].iloc[0])
    med, scale = stats["num_stats"]["height"]["median"], stats["num_stats"]["height"]["scale"]
    v_std = float(vectors[pid_missing][i_height])
    print(f"\nPatient {pid_missing}: height missing -> model input {v_std:.1f} = median {med:.3f} m")
    results["example_missing_height"] = {
        "patient": pid_missing, "model_input_standardized": v_std,
        "equivalent_original_units": med + v_std * scale, "median": med, "scale": scale}

    lvef_raw = raw_excel["Left ventricular ejection fraction"]
    i_lvef = USED_NUM.index("lvef_pct")
    lvef_in_model = sorted({float(v[i_lvef]) for v in vectors.values()})
    print(f"\nLVEF raw values: {lvef_raw.astype(str).str.strip().value_counts().to_dict()}")
    print(f"LVEF NaN after parsing: {int(meta['lvef_pct'].isna().sum())} of {n}; "
          f"distinct values in the model input: {lvef_in_model}")
    results["lvef"] = {
        "raw_value_counts": lvef_raw.astype(str).str.strip().value_counts().to_dict(),
        "nan_after_to_float": int(meta["lvef_pct"].isna().sum()),
        "nan_after_prepare": int(prepared["lvef_pct"].isna().sum()),
        "distinct_values_in_model_input": lvef_in_model,
    }

    print("\nBinary features: values outside the expected categories (encoded as all-zero)")
    cat_missing = {}
    for k in USED_CAT:
        cat_missing[k] = int((~meta[k].isin(CAT_VALS[k])).sum())
        print(f"   {k:15s} {cat_missing[k]:3d}")
    results["missing_binary"] = cat_missing

    with open(os.path.join(args.out_dir, "metadata_missingness.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
