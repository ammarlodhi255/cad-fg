import os, cv2, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset

# Numeric & categorical spec (aligned to your CSV)
CLIN_NUM = ["age","height","weight","bmi","evol_dm_years","lvef_pct","n_vessels","max_involvement_pct"]
CLIN_CAT = ["sex","diabetes","dyslipidemia","smoker","hypertension","kidney_failure","heart_failure","atrial_fib"]

CAT_VALS = {
    "sex": ["female","male"],
    "diabetes": ["no","yes"],
    "dyslipidemia": ["no","yes"],
    "smoker": ["no","yes"],
    "hypertension": ["no","yes"],
    "kidney_failure": ["no","yes"],
    "heart_failure": ["no","yes"],
    "atrial_fib": ["no","yes"],
}

def _read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return np.stack([img,img,img], axis=-1)

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def load_metadata_any(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # exact mapping from your header -> canonical keys
    col_map = {
        "Patient ID": "patient",
        "Age (years)": "age",
        "Sex": "sex",
        "Height (m)": "height",
        "Weight (kg)": "weight",
        "BMI": "bmi",
        "Diabetes mellitus": "diabetes",
        "Evolution diabetes (years)": "evol_dm_years",
        "Dyslipidemia": "dyslipidemia",
        "Smoker": "smoker",
        "High blood pressure": "hypertension",
        "Another comorbidity": "another_comorb",
        "Kidney failure": "kidney_failure",
        "Heart failure": "heart_failure",
        "Atrial fibrillation": "atrial_fib",
        "Left ventricular ejection fraction": "lvef_pct",
        "Clinical indication for angiogrphy": "indication",
        "Number of vessels affected": "n_vessels",
        "Maximum degree of the coronary artery involvement": "max_involvement_pct",
        "Later medical events": "later_events",
    }
    df = df.rename(columns=col_map)

    # normalize
    if "patient" in df.columns: df["patient"] = df["patient"].astype(int)
    for k in df.columns:
        if df[k].dtype == object:
            df[k] = df[k].astype(str).str.strip().str.lower()

    # numeric coercion
    for k in ["age","height","weight","bmi","evol_dm_years","lvef_pct","n_vessels","max_involvement_pct"]:
        if k in df.columns:
            df[k] = df[k].apply(_to_float)

    # yes/no normalization
    def norm_bin(s):
        s = str(s).strip().lower()
        if s in ("1","y","yes","true","t","si"): return "yes"
        if s in ("0","n","no","false","f","non"): return "no"
        return s
    for k in ["diabetes","dyslipidemia","smoker","hypertension","kidney_failure","heart_failure","atrial_fib"]:
        if k in df.columns:
            df[k] = df[k].apply(norm_bin)

    return df

def _safe_num_col(df, key):
    v = pd.to_numeric(df.get(key, pd.Series([np.nan]*len(df))), errors="coerce")
    return v

# def build_meta_vectors(meta_df):
#     df = meta_df.copy()

#     # ---- Derive/fix numeric fields ----
#     # Coerce to numeric
#     for k in ["age","height","weight","bmi","evol_dm_years","lvef_pct","n_vessels","max_involvement_pct"]:
#         if k in df.columns:
#             df[k] = _safe_num_col(df, k)

#     # If BMI missing but height+weight present, compute BMI = kg / m^2
#     if {"bmi","height","weight"}.issubset(df.columns):
#         mask = df["bmi"].isna() & df["height"].gt(0) & df["weight"].gt(0)
#         df.loc[mask, "bmi"] = df.loc[mask, "weight"] / (df.loc[mask, "height"] ** 2)

#     # If LVEF given as text categories in your sheet (sometimes happens), map to % midpoints
#     if "lvef_pct" in df.columns:
#         # normalize stringy entries
#         txt = meta_df.get("lvef_pct").astype(str).str.strip().str.lower()
#         map_txt = {
#             "normal": 60.0, "mild": 50.0, "moderate": 40.0, "severe": 25.0,
#             "mild dysfunction": 50.0, "moderate dysfunction": 40.0,
#             "severe dysfunction": 25.0
#         }
#         # only fill where lvef_pct is NaN and text looks like category
#         fill_idx = df["lvef_pct"].isna() & txt.isin(map_txt)
#         df.loc[fill_idx, "lvef_pct"] = txt[fill_idx].map(map_txt)

#     # Reasonable ranges clamp to kill outliers or unit mistakes
#     clamps = {
#         "age": (18, 100),
#         "height": (1.3, 2.1),
#         "weight": (35, 200),
#         "bmi": (12, 60),
#         "evol_dm_years": (0, 60),
#         "lvef_pct": (10, 80),
#         "n_vessels": (0, 4),
#         "max_involvement_pct": (0, 100),
#     }
#     for k,(lo,hi) in clamps.items():
#         if k in df.columns:
#             df[k] = df[k].clip(lower=lo, upper=hi)

#     # ---- Z-score numeric (robust) and fill remaining NaNs with 0 ----
#     for k in CLIN_NUM:
#         if k in df.columns:
#             v = pd.to_numeric(df[k], errors="coerce")
#             m = v.median(skipna=True)
#             mad = (v - m).abs().median(skipna=True)
#             s = float(mad * 1.4826) if pd.notna(mad) and mad > 1e-6 else float(v.std(skipna=True) or 1.0)
#             df[k] = (v - (m if pd.notna(m) else 0.0)) / (s if s else 1.0)
#         else:
#             df[k] = 0.0  # missing column -> zeros

#     # ---- Categorical one-hots (guaranteed finite) ----
#     out = {}
#     for pid, rows in df.groupby("patient"):
#         row = rows.iloc[0]

#         num = np.array([row.get(k, 0.0) for k in CLIN_NUM], dtype=np.float32)
#         num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)

#         cat = []
#         for k in CLIN_CAT:
#             val = str(row.get(k, "")).strip().lower()
#             base = CAT_VALS.get(k, ["no","yes"])
#             onehot = [1.0 if val == b else 0.0 for b in base]
#             cat.extend(onehot)

#         vec = np.concatenate([num, np.array(cat, dtype=np.float32)], axis=0)
#         # final safety: finite only
#         vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
#         vec = np.clip(vec, -10.0, 10.0)
#         out[int(pid)] = vec

#     return out

def build_meta_vectors(meta_df, selected_features=None):
    """
    Build per-patient metadata vectors with robust cleaning and optional feature selection.

    selected_features (optional dict):
      {
        "numeric": ["age","bmi","lvef_pct", ...],        # subset of CLIN_NUM
        "categorical": ["sex","diabetes","smoker", ...]  # subset of CLIN_CAT
      }
    If selected_features is None or keys are empty/missing, defaults to ALL in CLIN_NUM/CLIN_CAT.
    """
    df = meta_df.copy()

    # -------------------- Derive/fix numeric fields --------------------
    # Coerce to numeric for all possible numeric columns we know about.
    for k in ["age","height","weight","bmi","evol_dm_years","lvef_pct","n_vessels","max_involvement_pct"]:
        if k in df.columns:
            df[k] = _safe_num_col(df, k)

    # Compute BMI when missing and height/weight are present and valid (kg / m^2).
    if {"bmi","height","weight"}.issubset(df.columns):
        mask = df["bmi"].isna() & df["height"].gt(0) & df["weight"].gt(0)
        df.loc[mask, "bmi"] = df.loc[mask, "weight"] / (df.loc[mask, "height"] ** 2)

    # Map textual LVEF categories (if present) to % midpoints, only where lvef_pct is NaN.
    if "lvef_pct" in df.columns:
        txt = meta_df.get("lvef_pct").astype(str).str.strip().str.lower()
        map_txt = {
            "normal": 60.0, "mild": 50.0, "moderate": 40.0, "severe": 25.0,
            "mild dysfunction": 50.0, "moderate dysfunction": 40.0, "severe dysfunction": 25.0
        }
        fill_idx = df["lvef_pct"].isna() & txt.isin(map_txt)
        df.loc[fill_idx, "lvef_pct"] = txt[fill_idx].map(map_txt)

    # Clamp to reasonable clinical ranges to kill outliers/unit mistakes.
    clamps = {
        "age": (18, 100),
        "height": (1.3, 2.1),
        "weight": (35, 200),
        "bmi": (12, 60),
        "evol_dm_years": (0, 60),
        "lvef_pct": (10, 80),
        "n_vessels": (0, 4),
        "max_involvement_pct": (0, 100),
    }
    for k, (lo, hi) in clamps.items():
        if k in df.columns:
            df[k] = df[k].clip(lower=lo, upper=hi)

    # -------------------- Decide which features to use --------------------
    # Defaults: use all declared features.
    use_num = list(CLIN_NUM)
    use_cat = list(CLIN_CAT)
    if selected_features is not None:
        if isinstance(selected_features, dict):
            if selected_features.get("numeric"):
                use_num = [k for k in selected_features["numeric"] if k in CLIN_NUM]
            if selected_features.get("categorical"):
                use_cat = [k for k in selected_features["categorical"] if k in CLIN_CAT]

    # -------------------- Robust z-scoring for numeric --------------------
    for k in CLIN_NUM:
        if k in df.columns:
            v = pd.to_numeric(df[k], errors="coerce")
            # if column empty → zeros
            if v.notna().sum() == 0:
                df[k] = 0.0
                continue
            m = v.median(skipna=True)
            mad = (v - m).abs().median(skipna=True)
            # robust sigma; fall back to std if MAD too small
            s = float(mad * 1.4826) if pd.notna(mad) and mad > 1e-6 else float(v.std(skipna=True) or 1.0)
            if not np.isfinite(s) or s <= 1e-12:
                s = 1.0
            df[k] = (v - (m if pd.notna(m) else 0.0)) / s
        else:
            df[k] = 0.0  # column missing entirely → zeros

    # -------------------- Build vectors per patient --------------------
    out = {}
    for pid, rows in df.groupby("patient"):
        row = rows.iloc[0]

        # numeric (selected subset)
        num = np.array([row.get(k, 0.0) for k in use_num], dtype=np.float32)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)

        # categorical (selected subset)
        cat = []
        for k in use_cat:
            val = str(row.get(k, "")).strip().lower()
            base = CAT_VALS.get(k, ["no", "yes"])
            onehot = [1.0 if val == b else 0.0 for b in base]
            cat.extend(onehot)

        vec = np.concatenate([num, np.array(cat, dtype=np.float32)], axis=0)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.clip(vec, -10.0, 10.0)
        out[int(pid)] = vec

    return out

class CadicaDataset(Dataset):
    # def __init__(self, csv_path, metadata_path, tfm):
    #     self.df = pd.read_csv(csv_path)
    #     self.meta_df = load_metadata_any(metadata_path)
    #     self.meta = build_meta_vectors(self.meta_df)
    #     self.tfm = tfm
    #     any_pid = int(self.df.iloc[0]["patient"])
    #     self.tab_dim = len(self.meta[int(any_pid)])

    # def __init__(self, csv_path, metadata_path, tfm, selected_features=None):
    #     self.df = pd.read_csv(csv_path)
    #     self.meta_df = load_metadata_any(metadata_path)
    #     self.selected_features = selected_features
    #     self.meta = build_meta_vectors(self.meta_df, selected_features=self.selected_features)
    #     self.tfm = tfm
    #     any_pid = int(self.df.iloc[0]["patient"])
    #     self.tab_dim = len(self.meta[int(any_pid)])

    def __init__(self, csv_path, metadata_path, tfm, selected_features=None):
        self.df = pd.read_csv(csv_path)
        self.meta_df = load_metadata_any(metadata_path)
        self.selected_features = selected_features

        # build per-patient vectors with (optional) subset
        self.meta = build_meta_vectors(self.meta_df, selected_features=self.selected_features)
        self.tfm = tfm

        # ---- record which features (and order) are used, to make auditing easy ----
        # default to full sets; then honor selection if provided
        self.used_num = list(CLIN_NUM)
        self.used_cat = list(CLIN_CAT)
        if isinstance(selected_features, dict):
            if selected_features.get("numeric"):
                self.used_num = [k for k in selected_features["numeric"] if k in CLIN_NUM]
            if selected_features.get("categorical"):
                self.used_cat = [k for k in selected_features["categorical"] if k in CLIN_CAT]

        cat_expanded = []
        for k in self.used_cat:
            base = CAT_VALS.get(k, ["no", "yes"])
            cat_expanded += [f"{k}={b}" for b in base]
        self.feature_names = self.used_num + cat_expanded

        # infer tab_dim from names (matches vectors we built)
        self.tab_dim = len(self.feature_names)

        # simple sanity: ensure at least one patient exists
        if len(self.df) == 0:
            raise RuntimeError("Empty dataframe in CadicaDataset.")

        
    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = _read_img(r.img_path)
        data = self.tfm(image=img)
        x = data["image"].float() / 255.0
        y = torch.tensor(int(r.label), dtype=torch.long)
        clin = torch.tensor(self.meta[int(r.patient)], dtype=torch.float32)
        return {"x": x, "y": y, "clin": clin,
                "patient": int(r.patient), "video": int(r.video), "frame": int(r.frame),
                "img_path": r.img_path}

    def get_feature_names(self):
        """Expanded feature names in vector order (numeric + one-hots)."""
        return list(self.feature_names)

    def get_feature_groups(self):
        """Original (pre-one-hot) feature groups actually used."""
        return {"numeric": list(self.used_num), "categorical": list(self.used_cat)}