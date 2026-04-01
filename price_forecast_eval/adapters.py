"""标准评估表适配：DatetimeIndex / 结果 CSV → date, hour, y_true, y_pred。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .scenario_tags import ScenarioTagConfig, attach_scenario_tags


def to_eval_frame(
    index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attach_tags: bool = False,
    tag_config: Optional[ScenarioTagConfig] = None,
) -> pd.DataFrame:
    if len(index) != len(y_true) or len(index) != len(y_pred):
        raise ValueError("index, y_true, y_pred 长度须一致")
    ts = pd.to_datetime(pd.Series(index))
    df = pd.DataFrame(
        {
            "date": ts.dt.strftime("%Y-%m-%d"),
            "hour": ts.dt.hour,
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    if attach_tags:
        df = attach_scenario_tags(df, config=tag_config)
    return df


def from_result_columns(
    df: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "pred",
    ts_index: bool = True,
    attach_tags: bool = False,
    tag_config: Optional[ScenarioTagConfig] = None,
) -> pd.DataFrame:
    if ts_index:
        idx = pd.to_datetime(df.index)
    else:
        idx = pd.to_datetime(df["ts"])
    return to_eval_frame(
        idx,
        df[actual_col].values,
        df[pred_col].values,
        attach_tags=attach_tags,
        tag_config=tag_config,
    )
