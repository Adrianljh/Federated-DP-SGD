import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}