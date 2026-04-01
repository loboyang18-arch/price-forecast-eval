"""
附录 §5 分场景：日级标签广播到逐点行。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import AbstractSet, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .validation import _VAR_EPS


@dataclass
class ScenarioTagConfig:
    quantile_ref_dates: Optional[AbstractSet[date]] = None
    holiday_dates: Optional[AbstractSet[date]] = None
    weekend_weekday_min: int = 5


def _day_key(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _daily_amplitude_valid(
    df: pd.DataFrame,
    date_col: str,
    hour_col: str,
    y_true_col: str,
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    dfp = df.copy()
    dfp["_d"] = pd.to_datetime(dfp[date_col]).dt.normalize()
    dfp["_dk"] = dfp["_d"].map(_day_key)
    dfp["_h"] = dfp[hour_col].astype(int)
    dfp["_yt"] = pd.to_numeric(dfp[y_true_col], errors="coerce")

    amp: Dict[str, float] = {}
    ok: Dict[str, bool] = {}
    for dk in sorted(dfp["_dk"].unique()):
        sub = dfp[dfp["_dk"] == dk].sort_values("_h")
        if len(sub) != 24 or set(sub["_h"].tolist()) != set(range(24)):
            ok[dk] = False
            continue
        y = sub["_yt"].to_numpy(dtype=float)
        if not np.all(np.isfinite(y)) or np.var(y) <= _VAR_EPS:
            ok[dk] = False
            continue
        ok[dk] = True
        amp[dk] = float(np.max(y) - np.min(y))
    return amp, ok


def attach_scenario_tags(
    df: pd.DataFrame,
    date_col: str = "date",
    hour_col: str = "hour",
    y_true_col: str = "y_true",
    config: Optional[ScenarioTagConfig] = None,
) -> pd.DataFrame:
    cfg = config or ScenarioTagConfig()
    out = df.copy()
    dts = pd.to_datetime(out[date_col])

    out["tag_weekend"] = (dts.dt.weekday >= cfg.weekend_weekday_min).astype(int)

    if cfg.holiday_dates:
        d_only = dts.dt.date
        out["tag_holiday"] = d_only.map(lambda x: 1 if x in cfg.holiday_dates else 0).astype(int)
    else:
        out["tag_holiday"] = 0

    amp, day_ok = _daily_amplitude_valid(out, date_col, hour_col, y_true_col)

    ref_vals: list[float] = []
    for dk, a in amp.items():
        if not day_ok.get(dk, False):
            continue
        if cfg.quantile_ref_dates is not None and pd.Timestamp(dk).date() not in cfg.quantile_ref_dates:
            continue
        ref_vals.append(a)

    if len(ref_vals) < 3:
        q33 = q66 = p90 = float("nan")
    else:
        arr = np.asarray(ref_vals, dtype=float)
        q33, q66 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
        p90 = float(np.quantile(arr, 0.90))

    vol_class_by_day: Dict[str, object] = {}
    extreme_by_day: Dict[str, int] = {}

    for dk in day_ok:
        if not day_ok[dk]:
            vol_class_by_day[dk] = np.nan
            extreme_by_day[dk] = 0
            continue
        a = amp.get(dk, float("nan"))
        if not np.isfinite(a):
            vol_class_by_day[dk] = np.nan
            extreme_by_day[dk] = 0
            continue
        if not np.isfinite(q33) or not np.isfinite(q66):
            vol_class_by_day[dk] = "mid"
        elif a <= q33:
            vol_class_by_day[dk] = "typical"
        elif a <= q66:
            vol_class_by_day[dk] = "mid"
        else:
            vol_class_by_day[dk] = "high_vol"
        extreme_by_day[dk] = 1 if (np.isfinite(p90) and a >= p90) else 0

    row_dk = dts.dt.strftime("%Y-%m-%d")
    out["tag_vol_class"] = row_dk.map(lambda k: vol_class_by_day.get(k, np.nan))
    out["tag_extreme"] = row_dk.map(lambda k: extreme_by_day.get(k, 0)).astype(int)
    return out
