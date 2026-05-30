from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse_val = float(mean_squared_error(y_true, y_pred))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(mean_absolute_error(y_true, y_pred))
    r2_val = float(r2_score(y_true, y_pred))
    pearson = _safe_corr(y_true, y_pred, method="pearson")
    spearman = _safe_corr(y_true, y_pred, method="spearman")
    return {
        "mse": mse_val,
        "rmse": rmse_val,
        "mae": mae_val,
        "r2": r2_val,
        "spearman": spearman,
        "pearson": pearson,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    try:
        if method == "pearson":
            corr, _ = pearsonr(a, b)
        else:
            corr, _ = spearmanr(a, b)
        return float(corr)
    except Exception:
        return float("nan")
