import os, cv2, torch, pandas as pd, numpy as np, re, json
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from src.text.encoders import load_sentence_encoder, encode_texts

# ------------------ Clinical feature specs ------------------
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

TEXT_COLS_DEFAULT = [
    "Another comorbidity",
    "Clinical indication for angiogrphy",
    "Later medical events",
]

# ------------------ Utils ------------------
def _norm_text(x) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip()
    return re.sub(r"\s+", " ", s).lower()

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

# ------------------ Metadata loaders ------------------
def load_metadata_any(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

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

    if "patient" in df.columns:
        df["patient"] = df["patient"].astype(int)

    for k in df.columns:
        if df[k].dtype == object:
            df[k] = df[k].astype(str).str.strip().str.lower()

    for k in CLIN_NUM:
        if k in df.columns:
            df[k] = df[k].apply(_to_float)

    def norm_bin(s):
        s = str(s).strip().lower()
        if s in ("1","y","yes","true","t","si"): return "yes"
        if s in ("0","n","no","false","f","non"): return "no"
        return s
    for k in CLIN_CAT:
        if k in df.columns:
            df[k] = df[k].apply(norm_bin)

    return df

def _safe_num_col(df, key):
    v = pd.to_numeric(df.get(key, pd.Series([np.nan]*len(df))), errors="coerce")
    return v

# ------------------ Text embedding builder ------------------
def build_text_embed_map(
    meta_df: pd.DataFrame,
    patient_col: str = "patient",
    text_cols: Optional[List[str]] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = "outputs/text_embeds",
) -> Dict[int, np.ndarray]:
    os.makedirs(cache_dir, exist_ok=True)
    if text_cols is None:
        text_cols = TEXT_COLS_DEFAULT

    cols = [patient_col] + [c for c in text_cols if c in meta_df.columns]
    df = meta_df[cols].copy()

    df = df.groupby(patient_col).agg(lambda s: s.iloc[0] if len(s) else "").reset_index()
    df["__joined__"] = df.apply(
        lambda r: " [SEP] ".join([_norm_text(r[c]) for c in text_cols if c in r and _norm_text(r[c]) != ""]),
        axis=1
    )
    df.loc[df["__joined__"].str.len() == 0, "__joined__"] = "[none]"

    sig = {"model": model_name, "text_cols": [c for c in text_cols if c in meta_df.columns]}
    # stable cache key based on md5 of signature
    import hashlib
    sig_json = json.dumps(sig, sort_keys=True)
    sig_name = f"{os.path.basename(model_name).replace('/','_')}_{hashlib.md5(sig_json.encode('utf-8')).hexdigest()}"
    pt_path = os.path.join(cache_dir, f"{sig_name}.pt")

    if os.path.exists(pt_path):
        obj = torch.load(pt_path, map_location="cpu")
        if set(obj["pids"]) == set(df[patient_col].tolist()):
            emb = obj["emb"]
            return {int(pid): emb[i].numpy() for i, pid in enumerate(obj["pids"])}

    enc = load_sentence_encoder(model_name)
    emb = encode_texts(df["__joined__"].tolist(), enc)  # [N,D]
    pids = df[patient_col].tolist()
    torch.save({"pids": pids, "emb": emb}, pt_path)
    return {int(pid): emb[i].numpy() for i, pid in enumerate(pids)}

def _prepare_meta_base(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Common preparation of metadata DataFrame (type coercion, derived values, clamps)."""
    df = meta_df.copy()

    # numeric coercion
    for k in CLIN_NUM:
        if k in df.columns:
            df[k] = _safe_num_col(df, k)

    if {"bmi","height","weight"}.issubset(df.columns):
        mask = df["bmi"].isna() & df["height"].gt(0) & df["weight"].gt(0)
        df.loc[mask,"bmi"] = df.loc[mask,"weight"] / (df.loc[mask,"height"]**2)

    if "lvef_pct" in df.columns:
        txt = meta_df.get("lvef_pct").astype(str).str.strip().str.lower()
        map_txt = {"normal":60.0,"mild":50.0,"moderate":40.0,"severe":25.0}
        fill_idx = df["lvef_pct"].isna() & txt.isin(map_txt)
        df.loc[fill_idx,"lvef_pct"] = txt[fill_idx].map(map_txt)

    clamps = {"age":(18,100),"height":(1.3,2.1),"weight":(35,200),"bmi":(12,60),
              "evol_dm_years":(0,60),"lvef_pct":(10,80),"n_vessels":(0,4),"max_involvement_pct":(0,100)}
    for k,(lo,hi) in clamps.items():
        if k in df.columns:
            df[k] = df[k].clip(lo,hi)

    return df

def compute_meta_stats(meta_df: pd.DataFrame,
                       selected_features=None,
                       fit_patients: Optional[List[int]] = None) -> Dict:
    """
    Compute per-numeric feature robust center/scale on a training patient subset.
    Returns a dict with keys: 'use_num', 'use_cat', 'num_stats' mapping feature -> {median, scale}.
    """
    df = _prepare_meta_base(meta_df)

    use_num = list(CLIN_NUM)
    use_cat = list(CLIN_CAT)
    if isinstance(selected_features, dict):
        if selected_features.get("numeric"):
            use_num = [k for k in selected_features["numeric"] if k in CLIN_NUM]
        if selected_features.get("categorical"):
            use_cat = [k for k in selected_features["categorical"] if k in CLIN_CAT]

    if fit_patients is not None:
        df_fit = df[df["patient"].isin(list(fit_patients))].copy()
    else:
        df_fit = df

    num_stats = {}
    for k in CLIN_NUM:
        if k in df_fit.columns:
            v = pd.to_numeric(df_fit[k], errors="coerce")
            if v.notna().sum() == 0:
                m = 0.0; s = 1.0
            else:
                m = v.median(skipna=True)
                mad = (v - m).abs().median(skipna=True)
                s = float(mad * 1.4826) if pd.notna(mad) and mad > 1e-6 else float(v.std(skipna=True) or 1.0)
                if not np.isfinite(s) or s <= 1e-12:
                    s = 1.0
            num_stats[k] = {"median": float(m if pd.notna(m) else 0.0), "scale": float(s)}
        else:
            num_stats[k] = {"median": 0.0, "scale": 1.0}

    return {"use_num": use_num, "use_cat": use_cat, "num_stats": num_stats}

def build_meta_vectors_from_stats(meta_df: pd.DataFrame,
                                  selected_features=None,
                                  text_embed_map: Optional[Dict[int, np.ndarray]] = None,
                                  stats: Optional[Dict] = None) -> Dict[int, np.ndarray]:
    """
    Build per-patient vectors using provided numeric stats. If stats is None, computes from all patients
    (backward compatible, but not recommended for CV to avoid leakage).
    """
    df = _prepare_meta_base(meta_df)

    # features to include
    if stats is not None:
        use_num = list(stats.get("use_num", CLIN_NUM))
        use_cat = list(stats.get("use_cat", CLIN_CAT))
        num_stats = stats.get("num_stats", {})
    else:
        use_num = list(CLIN_NUM)
        use_cat = list(CLIN_CAT)
        if isinstance(selected_features, dict):
            if selected_features.get("numeric"):
                use_num = [k for k in selected_features["numeric"] if k in CLIN_NUM]
            if selected_features.get("categorical"):
                use_cat = [k for k in selected_features["categorical"] if k in CLIN_CAT]
        # compute fallback stats on whole df
        num_stats = compute_meta_stats(df, selected_features=selected_features).get("num_stats", {})

    for k in CLIN_NUM:
        if k in df.columns:
            v = pd.to_numeric(df[k], errors="coerce")
            st = num_stats.get(k, {"median": 0.0, "scale": 1.0})
            m = st.get("median", 0.0)
            s = st.get("scale", 1.0) or 1.0
            df[k] = (v - m) / s
        else:
            df[k] = 0.0

    out={}
    for pid, rows in df.groupby("patient"):
        row = rows.iloc[0]
        num = np.array([row.get(k,0.0) for k in use_num],dtype=np.float32)
        num = np.nan_to_num(num)

        cat=[]
        for k in use_cat:
            val=str(row.get(k,"")).strip().lower()
            base=CAT_VALS.get(k,["no","yes"])
            onehot=[1.0 if val==b else 0.0 for b in base]
            cat.extend(onehot)

        vecs=[num,np.array(cat,dtype=np.float32)]
        if text_embed_map is not None:
            if int(pid) in text_embed_map:
                t=text_embed_map[int(pid)].astype(np.float32)
                t=np.nan_to_num(t); t=np.clip(t,-10.0,10.0)
                vecs.append(t)

        vec=np.concatenate(vecs,axis=0)
        out[int(pid)] = vec
    return out

# Backward-compatible wrapper (computes stats on full df; may leak in CV, prefer from_stats)
def build_meta_vectors(meta_df, selected_features=None, text_embed_map: Optional[Dict[int,np.ndarray]]=None):
    return build_meta_vectors_from_stats(meta_df,
                                         selected_features=selected_features,
                                         text_embed_map=text_embed_map,
                                         stats=None)

# ------------------ Dataset ------------------
class CadicaDataset(Dataset):
    def __init__(self, csv_path, metadata_path, tfm,
                 selected_features=None,
                 text_features=None,
                 text_model="sentence-transformers/all-MiniLM-L6-v2",
                 text_cache_dir="outputs/text_embeds",
                 fit_patients: Optional[List[int]] = None,
                 stats: Optional[Dict] = None):

        self.df = pd.read_csv(csv_path)
        self.meta_df = load_metadata_any(metadata_path)
        self.selected_features = selected_features

        # optional text embedding map
        text_map=None
        if text_features:
            text_map=build_text_embed_map(self.meta_df, patient_col="patient",
                                          text_cols=text_features,
                                          model_name=text_model,
                                          cache_dir=text_cache_dir)

        # compute or use provided stats (fit on training patients to avoid leakage)
        if stats is None:
            # if fit_patients provided, compute on that subset; else fallback to all patients
            self.stats = compute_meta_stats(self.meta_df,
                                           selected_features=self.selected_features,
                                           fit_patients=fit_patients)
        else:
            self.stats = stats

        self.meta = build_meta_vectors_from_stats(self.meta_df,
                                                  selected_features=self.selected_features,
                                                  text_embed_map=text_map,
                                                  stats=self.stats)
        self.tfm=tfm

        # record which features are used
        self.used_num = list(self.stats.get("use_num", CLIN_NUM))
        self.used_cat = list(self.stats.get("use_cat", CLIN_CAT))

        cat_expanded=[]
        for k in self.used_cat:
            base=CAT_VALS.get(k,["no","yes"])
            cat_expanded += [f"{k}={b}" for b in base]

        self.feature_names=self.used_num+cat_expanded
        if text_features:
            self.feature_names += [f"text[{i}]" for i in range(len(next(iter(text_map.values()))))]

        self.tab_dim=len(self.feature_names)

        if len(self.df)==0:
            raise RuntimeError("Empty dataframe in CadicaDataset.")

    def __len__(self): return len(self.df)

    def __getitem__(self,i):
        r=self.df.iloc[i]
        img=_read_img(r.img_path)
        data=self.tfm(image=img)
        x=data["image"].float()
        y=torch.tensor(int(r.label),dtype=torch.long)
        clin=torch.tensor(self.meta[int(r.patient)],dtype=torch.float32)
        return {"x":x,"y":y,"clin":clin,
                "patient":int(r.patient),"video":int(r.video),"frame":int(r.frame),
                "img_path":r.img_path}

    def get_feature_names(self): return list(self.feature_names)

    def get_feature_groups(self):
        return {"numeric":list(self.used_num),"categorical":list(self.used_cat)}
