"""标准总入口 evaluate_model_predictions。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .composite import compute_composite_score
from .point_metrics import compute_point_metrics
from .segment_metrics import compute_metrics_by_segment
from .shape_metrics import compute_shape_metrics
from .validation import validate_eval_frame


def evaluate_model_predictions(
    df: pd.DataFrame,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    task_type: str = "da",
    date_col: str = "date",
    hour_col: str = "hour",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    include_extended: bool = True,
    segment_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    validation = validate_eval_frame(df, date_col, hour_col, y_true_col, y_pred_col)
    point_metrics = compute_point_metrics(df, y_true_col, y_pred_col)
    shape_metrics = compute_shape_metrics(
        df, date_col, hour_col, y_true_col, y_pred_col, include_extended
    )

    merged_for_composite = {**point_metrics, **shape_metrics}
    segments: Dict[str, Any] = {}
    if segment_cols:
        for sc in segment_cols:
            if sc in df.columns:
                segments[sc] = compute_metrics_by_segment(
                    df,
                    sc,
                    None,
                    date_col,
                    hour_col,
                    y_true_col,
                    y_pred_col,
                    include_extended,
                )

    composite = None
    if baseline_metrics is not None:
        composite = compute_composite_score(merged_for_composite, baseline_metrics, task_type)

    return {
        "validation": validation,
        "point_metrics": point_metrics,
        "shape_metrics": shape_metrics,
        "segments": segments,
        "composite": composite,
    }
