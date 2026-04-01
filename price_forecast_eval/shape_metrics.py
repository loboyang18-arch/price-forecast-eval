"""日级 shape 指标 — 附录 §4 口径（含预测平线 corr=0 仍计入 D_valid）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .validation import _VAR_EPS

APPENDIX_BLOCK_SLICES = {
    "overnight": slice(0, 6),
    "morning_peak": slice(6, 12),
    "midday": slice(12, 17),
    "evening_peak": slice(17, 22),
    "night": slice(22, 24),
}
_TOL_TURN = 1


def _iter_valid_days_appendix(
    df: pd.DataFrame, date_col: str, hour_col: str, y_true_col: str, y_pred_col: str
) -> List[Tuple[object, np.ndarray, np.ndarray]]:
    dfp = df.copy()
    dfp["_d"] = pd.to_datetime(dfp[date_col]).dt.normalize()
    dfp["_h"] = dfp[hour_col].astype(int)
    dfp["_yt"] = pd.to_numeric(dfp[y_true_col], errors="coerce")
    dfp["_yp"] = pd.to_numeric(dfp[y_pred_col], errors="coerce")
    out: List[Tuple[object, np.ndarray, np.ndarray]] = []
    for d in sorted(dfp["_d"].unique()):
        sub = dfp[dfp["_d"] == d].sort_values("_h")
        if len(sub) != 24 or set(sub["_h"].tolist()) != set(range(24)):
            continue
        y = sub["_yt"].to_numpy(dtype=float)
        p = sub["_yp"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
            continue
        if np.var(y) <= _VAR_EPS:
            continue
        out.append((d, y, p))
    return out


def _pearson_corr_d(y: np.ndarray, p: np.ndarray) -> float:
    if np.var(p) <= _VAR_EPS:
        return 0.0
    c = np.corrcoef(y, p)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _direction_acc_day(y: np.ndarray, p: np.ndarray) -> float:
    da = np.diff(y.astype(float))
    dp = np.diff(p.astype(float))
    hits = 0
    for t in range(len(da)):
        sa, sp = np.sign(da[t]), np.sign(dp[t])
        if (sa == 0 and sp == 0) or (sa == sp):
            hits += 1
    return hits / len(da) if len(da) else float("nan")


def _norm_profile_mae_day(y: np.ndarray, p: np.ndarray) -> float:
    sy, sp = np.std(y), np.std(p)
    if sy <= _VAR_EPS or sp <= _VAR_EPS:
        return float("nan")
    zy = (y - np.mean(y)) / sy
    zp = (p - np.mean(p)) / sp
    return float(np.mean(np.abs(zp - zy)))


def _turning_points(x: np.ndarray, min_abs_diff: float = 1e-6) -> List[int]:
    d = np.diff(x.astype(float))
    pts: List[int] = []
    for i in range(1, len(d)):
        if abs(d[i - 1]) < min_abs_diff or abs(d[i]) < min_abs_diff:
            continue
        if d[i - 1] * d[i] < 0:
            pts.append(i)
    return pts


def _turning_point_stats(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    ta = _turning_points(y)
    tp = _turning_points(p)
    if not ta:
        return float("nan"), float("nan")
    matched, offsets = 0, []
    for t in ta:
        ok = any(abs(t - u) <= _TOL_TURN for u in tp)
        if ok:
            matched += 1
            offsets.append(float(min(abs(t - u) for u in tp)))
    return matched / len(ta), (float(np.mean(offsets)) if offsets else float("nan"))


def _block_rank_acc_day(y: np.ndarray, p: np.ndarray) -> float:
    names = list(APPENDIX_BLOCK_SLICES.keys())
    ma = [float(np.mean(y[APPENDIX_BLOCK_SLICES[n]])) for n in names]
    mp = [float(np.mean(p[APPENDIX_BLOCK_SLICES[n]])) for n in names]
    return float(np.mean(np.argsort(np.argsort(ma)) == np.argsort(np.argsort(mp))))


def _block_mae_day(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {}
    for name, sl in APPENDIX_BLOCK_SLICES.items():
        aa, pp = y[sl], p[sl]
        if len(aa):
            out[f"block_mae_{name}"] = float(np.mean(np.abs(aa - pp)))
    return out


def _block_amp_err_day(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {}
    for name, sl in APPENDIX_BLOCK_SLICES.items():
        aa, pp = y[sl], p[sl]
        if len(aa) >= 2:
            out[f"block_amp_err_{name}"] = abs((np.max(pp) - np.min(pp)) - (np.max(aa) - np.min(aa)))
    return out


def compute_shape_metrics(
    df: pd.DataFrame,
    date_col: str = "date",
    hour_col: str = "hour",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    include_extended: bool = False,
) -> dict:
    days = _iter_valid_days_appendix(df, date_col, hour_col, y_true_col, y_pred_col)
    n_valid = len(days)
    total_calendar_days = int(pd.to_datetime(df[date_col]).dt.normalize().dt.date.nunique())
    all_dates = set(pd.to_datetime(df[date_col]).dt.date.unique())
    valid_dates = {pd.Timestamp(d).date() for d, _, _ in days}
    invalid_shape_days = int(len(all_dates - valid_dates))

    if n_valid == 0:
        base = {
            "profile_corr": float("nan"),
            "neg_corr_day_ratio": float("nan"),
            "neg_corr_day_count": 0,
            "amplitude_err": float("nan"),
            "direction_acc": float("nan"),
            "normalized_profile_mae": float("nan"),
            "peak_hour_error": float("nan"),
            "valley_hour_error": float("nan"),
            "valid_shape_days": 0,
            "total_calendar_days": total_calendar_days,
            "invalid_shape_days": invalid_shape_days,
            "invalid_shape_day_ratio": round(invalid_shape_days / total_calendar_days, 6)
            if total_calendar_days
            else float("nan"),
        }
        if include_extended:
            base.update(
                {
                    "turning_point_match_rate": float("nan"),
                    "turning_point_offset_mean": float("nan"),
                    "block_rank_acc": float("nan"),
                }
            )
            for name in APPENDIX_BLOCK_SLICES:
                base[f"block_mae_{name}"] = float("nan")
                base[f"block_amp_err_{name}"] = float("nan")
        return base

    corrs, amp_errs, dir_accs, norm_maes, peak_errs, valley_errs = [], [], [], [], [], []
    neg_n, tp_rates, tp_offs, branks = 0, [], [], []
    block_mae_acc: Dict[str, List[float]] = {f"block_mae_{k}": [] for k in APPENDIX_BLOCK_SLICES}
    block_amp_acc: Dict[str, List[float]] = {f"block_amp_err_{k}": [] for k in APPENDIX_BLOCK_SLICES}

    for _d, y, p in days:
        cd = _pearson_corr_d(y, p)
        corrs.append(cd)
        if cd < 0:
            neg_n += 1
        amp_errs.append(abs((np.max(p) - np.min(p)) - (np.max(y) - np.min(y))))
        dir_accs.append(_direction_acc_day(y, p))
        nm = _norm_profile_mae_day(y, p)
        if np.isfinite(nm):
            norm_maes.append(nm)
        peak_errs.append(abs(int(np.argmax(p)) - int(np.argmax(y))))
        valley_errs.append(abs(int(np.argmin(p)) - int(np.argmin(y))))

        if include_extended:
            tr, to = _turning_point_stats(y, p)
            if np.isfinite(tr):
                tp_rates.append(tr)
            if np.isfinite(to):
                tp_offs.append(to)
            branks.append(_block_rank_acc_day(y, p))
            for k, v in _block_mae_day(y, p).items():
                block_mae_acc[k].append(v)
            for k, v in _block_amp_err_day(y, p).items():
                block_amp_acc[k].append(v)

    base = {
        "profile_corr": round(float(np.mean(corrs)), 4),
        "neg_corr_day_ratio": round(float(neg_n / n_valid), 6),
        "neg_corr_day_count": int(neg_n),
        "amplitude_err": round(float(np.mean(amp_errs)), 4),
        "direction_acc": round(float(np.mean(dir_accs)), 4),
        "normalized_profile_mae": round(float(np.mean(norm_maes)), 4) if norm_maes else float("nan"),
        "peak_hour_error": round(float(np.mean(peak_errs)), 4),
        "valley_hour_error": round(float(np.mean(valley_errs)), 4),
        "valid_shape_days": n_valid,
        "total_calendar_days": total_calendar_days,
        "invalid_shape_days": invalid_shape_days,
        "invalid_shape_day_ratio": round(invalid_shape_days / total_calendar_days, 6)
        if total_calendar_days
        else float("nan"),
    }

    if include_extended:
        base["turning_point_match_rate"] = round(float(np.mean(tp_rates)), 4) if tp_rates else float("nan")
        base["turning_point_offset_mean"] = round(float(np.mean(tp_offs)), 4) if tp_offs else float("nan")
        base["block_rank_acc"] = round(float(np.mean(branks)), 4) if branks else float("nan")
        for k, vals in block_mae_acc.items():
            base[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
        for k, vals in block_amp_acc.items():
            base[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return base
