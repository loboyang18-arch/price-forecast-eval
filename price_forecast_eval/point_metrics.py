"""点级 MAE / RMSE。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_point_metrics(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict:
    yt = df[y_true_col].to_numpy(dtype=float)
    yp = df[y_pred_col].to_numpy(dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not m.any():
        return {"mae": float("nan"), "rmse": float("nan"), "valid_point_count": 0}
    e = yt[m] - yp[m]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "valid_point_count": int(m.sum()),
    }
