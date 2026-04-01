"""输入表校验与有效日统计（附录 §3）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

_VAR_EPS = 1e-9


def validate_eval_frame(
    df: pd.DataFrame,
    date_col: str = "date",
    hour_col: str = "hour",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict:
    errors: list[str] = []
    if df.empty:
        return {
            "ok": False,
            "errors": ["empty dataframe"],
            "n_rows": 0,
            "n_point_valid": 0,
            "valid_point_count": 0,
            "n_shape_valid_days": 0,
            "valid_shape_days": 0,
            "n_shape_invalid_days": 0,
            "invalid_shape_days": 0,
            "invalid_shape_day_ratio": float("nan"),
            "total_calendar_days": 0,
        }

    req = {date_col, hour_col, y_true_col, y_pred_col}
    miss = req - set(df.columns)
    if miss:
        errors.append(f"missing columns: {miss}")

    n_rows = len(df)
    if not errors:
        try:
            pd.to_datetime(df[date_col])
        except Exception as e:
            errors.append(f"date parse: {e}")
        if not pd.api.types.is_numeric_dtype(df[hour_col]):
            errors.append("hour must be numeric")

    n_point_valid = 0
    n_shape_valid = 0
    n_shape_invalid = 0

    if not errors:
        yt = pd.to_numeric(df[y_true_col], errors="coerce")
        yp = pd.to_numeric(df[y_pred_col], errors="coerce")
        m_pt = yt.notna() & yp.notna()
        n_point_valid = int(m_pt.sum())

        dfp = df.loc[m_pt].copy()
        dfp["_yt"] = yt[m_pt].values
        dfp["_yp"] = yp[m_pt].values
        dfp["_date"] = pd.to_datetime(dfp[date_col]).dt.normalize()
        dfp["_hour"] = dfp[hour_col].astype(int)

        dup = dfp.groupby(["_date", "_hour"]).size()
        if (dup > 1).any():
            errors.append("duplicate (date, hour) rows present")

        dates_all = dfp["_date"].dt.date.unique()
        total_calendar_days = int(len(dates_all))

        if not errors:
            for d in sorted(dfp["_date"].unique()):
                sub = dfp[dfp["_date"] == d].sort_values("_hour")
                hrs = set(sub["_hour"].tolist())
                if len(sub) != 24 or hrs != set(range(24)):
                    n_shape_invalid += 1
                    continue
                y = sub["_yt"].to_numpy(dtype=float)
                p = sub["_yp"].to_numpy(dtype=float)
                if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
                    n_shape_invalid += 1
                    continue
                if np.var(y) <= _VAR_EPS:
                    n_shape_invalid += 1
                    continue
                n_shape_valid += 1

            inv_ratio = (
                float(n_shape_invalid / total_calendar_days)
                if total_calendar_days > 0
                else float("nan")
            )
            return {
                "ok": True,
                "errors": [],
                "n_rows": n_rows,
                "n_point_valid": n_point_valid,
                "valid_point_count": n_point_valid,
                "n_shape_valid_days": n_shape_valid,
                "valid_shape_days": n_shape_valid,
                "n_shape_invalid_days": n_shape_invalid,
                "invalid_shape_days": n_shape_invalid,
                "invalid_shape_day_ratio": round(inv_ratio, 6),
                "total_calendar_days": total_calendar_days,
            }

    return {
        "ok": False,
        "errors": errors,
        "n_rows": n_rows,
        "n_point_valid": n_point_valid,
        "valid_point_count": n_point_valid,
        "n_shape_valid_days": n_shape_valid,
        "valid_shape_days": n_shape_valid,
        "n_shape_invalid_days": n_shape_invalid,
        "invalid_shape_days": n_shape_invalid,
        "invalid_shape_day_ratio": float("nan"),
        "total_calendar_days": int(pd.to_datetime(df[date_col]).dt.normalize().dt.date.nunique())
        if date_col in df.columns
        else 0,
    }
