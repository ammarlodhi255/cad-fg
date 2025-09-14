# import numpy as np
# from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# def metrics_from_probs(y_true, y_prob, thr=0.5):
#     y_pred = (y_prob >= thr).astype(int)
#     out = {
#         "bal_acc": balanced_accuracy_score(y_true, y_pred),
#         "f1": f1_score(y_true, y_pred),
#     }
#     try:
#         out["auc"] = roc_auc_score(y_true, y_prob)
#     except Exception:
#         out["auc"] = float("nan")
#     return out

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    accuracy_score, recall_score, precision_score,
    cohen_kappa_score, confusion_matrix
)

def metrics_from_probs(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)

    out = {}
    out["bal_acc"]   = balanced_accuracy_score(y_true, y_pred)
    out["acc"]       = accuracy_score(y_true, y_pred)
    out["recall"]    = recall_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred)
    out["f1"]        = f1_score(y_true, y_pred)
    try:
        out["auc"]   = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"]   = float("nan")

    # confusion-matrix based rates
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out["fpr"] = fp / (fp + tn + 1e-8)
    out["fnr"] = fn / (fn + tp + 1e-8)

    # inter-rater agreement
    out["kappa"] = cohen_kappa_score(y_true, y_pred)

    return out