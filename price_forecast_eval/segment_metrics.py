"""分场景评估（附录 §5）。"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .point_metrics import compute_point_metrics
from .shape_metrics import compute_shape_metrics


def compute_metrics_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    segment_values: Optional[List[str]] = None,
    date_col: str = "date",
    hour_col: str = "hour",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    include_extended: bool = False,
) -> dict:
    out: Dict[str, dict] = {}
    out["overall"] = {
        "point_metrics": compute_point_metrics(df, y_true_col, y_pred_col),
        "shape_metrics": compute_shape_metrics(
            df, date_col, hour_col, y_true_col, y_pred_col, include_extended
        ),
    }

    if segment_col not in df.columns:
        return out

    vals = segment_values
    if vals is None:
        vals = sorted(pd.unique(df[segment_col].dropna()), key=lambda x: str(x))

    for val in vals:
        sub = df[df[segment_col] == val]
        if sub.empty:
            continue
        out[str(val)] = {
            "point_metrics": compute_point_metrics(sub, y_true_col, y_pred_col),
            "shape_metrics": compute_shape_metrics(
                sub, date_col, hour_col, y_true_col, y_pred_col, include_extended
            ),
        }
    return out
