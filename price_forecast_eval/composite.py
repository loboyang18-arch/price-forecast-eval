"""Composite Score — 附录 §6；基线比值归一化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

_EPS = 1e-9


def _safe_ratio(model: float, baseline: float) -> float:
    if baseline is None or not (baseline == baseline):
        return 1.0
    if abs(float(baseline)) < _EPS:
        return 10.0 if abs(float(model)) > _EPS else 1.0
    return float(model) / float(baseline)


def _safe_loss_ratio(model_loss: float, baseline_loss: float) -> float:
    if baseline_loss is None or not (baseline_loss == baseline_loss):
        return 1.0
    bl = float(baseline_loss)
    if abs(bl) < _EPS:
        return 10.0 if abs(float(model_loss)) > _EPS else 1.0
    return float(model_loss) / bl


def compute_composite_score(
    metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    task_type: str,
) -> Dict[str, Any]:
    tt = task_type.lower().strip()
    if tt not in ("da", "rt"):
        raise ValueError("task_type must be 'da' or 'rt'")

    mae = float(metrics.get("mae", float("nan")))
    pc = float(metrics.get("profile_corr", float("nan")))
    neg = float(metrics.get("neg_corr_day_ratio", float("nan")))
    amp = float(metrics.get("amplitude_err", float("nan")))
    dacc = float(metrics.get("direction_acc", float("nan")))

    b_mae = float(baseline_metrics.get("mae", float("nan")))
    b_pc = float(baseline_metrics.get("profile_corr", float("nan")))
    b_neg = float(baseline_metrics.get("neg_corr_day_ratio", float("nan")))
    b_amp = float(baseline_metrics.get("amplitude_err", float("nan")))
    b_dacc = float(baseline_metrics.get("direction_acc", float("nan")))

    mae_n = _safe_ratio(mae, b_mae)
    corr_n = _safe_loss_ratio(1.0 - pc, 1.0 - b_pc)
    neg_n = _safe_ratio(neg, b_neg)
    amp_n = _safe_ratio(amp, b_amp)
    dir_n = _safe_loss_ratio(1.0 - dacc, 1.0 - b_dacc)

    if tt == "da":
        score = 0.25 * mae_n + 0.30 * corr_n + 0.20 * neg_n + 0.15 * amp_n + 0.10 * dir_n
    else:
        score = 0.30 * mae_n + 0.20 * corr_n + 0.20 * neg_n + 0.20 * amp_n + 0.10 * dir_n

    return {
        "composite_score": round(float(score), 6),
        "mae_norm": round(mae_n, 6),
        "corr_loss_norm": round(corr_n, 6),
        "neg_corr_norm": round(neg_n, 6),
        "amp_err_norm": round(amp_n, 6),
        "dir_loss_norm": round(dir_n, 6),
    }


def load_baseline_from_naive_summary_csv(
    path: Path | str,
    task: str,
    variant: str,
) -> Dict[str, float]:
    p = Path(path)
    df = pd.read_csv(p)
    row = df[(df["task"] == task) & (df["variant"] == variant)]
    if row.empty:
        raise ValueError(f"no row for task={task!r} variant={variant!r} in {p}")
    r = row.iloc[0]

    def pick(*names: str) -> float:
        for n in names:
            if n in row.columns and pd.notna(r.get(n)):
                return float(r[n])
        return float("nan")

    return {
        "mae": pick("mae"),
        "rmse": pick("rmse"),
        "profile_corr": pick("profile_corr"),
        "neg_corr_day_ratio": pick("neg_corr_day_ratio"),
        "amplitude_err": pick("amplitude_err"),
        "direction_acc": pick("direction_acc"),
        "peak_hour_error": pick("peak_hour_error", "peak_hour_err"),
        "valley_hour_error": pick("valley_hour_error", "valley_hour_err"),
    }
