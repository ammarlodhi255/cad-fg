import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split_by_patient(df, train=.8, val=.1, seed=123):
    if "patient" not in df.columns:
        raise KeyError("Expected 'patient' column in frame index.")
    gss = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
    tr_idx, rest_idx = next(gss.split(df, groups=df["patient"]))
    df_tr = df.iloc[tr_idx]; df_rest = df.iloc[rest_idx]
    val_frac = val / (1 - train)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac, random_state=seed+1)
    va_idx, te_idx = next(gss2.split(df_rest, groups=df_rest["patient"]))
    return (df_tr.reset_index(drop=True),
            df_rest.iloc[va_idx].reset_index(drop=True),
            df_rest.iloc[te_idx].reset_index(drop=True))