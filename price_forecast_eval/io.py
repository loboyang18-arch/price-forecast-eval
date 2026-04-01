"""训练产物 CSV → 标准评估 JSON。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .adapters import from_result_columns, to_eval_frame
from .composite import load_baseline_from_naive_summary_csv
from .evaluate import evaluate_model_predictions
from .scenario_tags import ScenarioTagConfig
from .shape_metrics import compute_shape_metrics


def _compute_auto_lag24h_baseline_metrics(
    df: pd.DataFrame,
    *,
    actual_col: str,
    task_type: str,
    include_extended: bool,
) -> Dict[str, Any]:
    if actual_col not in df.columns:
        raise ValueError(f"auto baseline 需要列: {actual_col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("auto baseline 需要 DatetimeIndex（ts）")

    work = df.sort_index().copy()
    y_true = pd.to_numeric(work[actual_col], errors="coerce")
    # 以时间对齐构造 t-24h 基线，避免依赖行位置连续性
    y_pred_baseline = y_true.reindex(work.index - pd.Timedelta(hours=24)).to_numpy()

    ef_base = to_eval_frame(work.index, y_true.to_numpy(), y_pred_baseline)
    ev_base = evaluate_model_predictions(
        ef_base,
        baseline_metrics=None,
        task_type=task_type,
        include_extended=include_extended,
        segment_cols=None,
    )
    return {**(ev_base.get("point_metrics") or {}), **(ev_base.get("shape_metrics") or {})}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            return str(obj)
    return obj


def evaluate_predictions_csv(
    csv_path: Path,
    *,
    actual_col: str = "actual",
    pred_col: str = "predicted",
    task_type: str = "da",
    include_extended: bool = True,
    baseline_path: Optional[Path] = None,
    baseline_task: str = "da",
    baseline_variant: str = "lag24h",
    auto_baseline: Optional[str] = None,
    with_scenario_tags: bool = False,
    segment_cols: Optional[List[str]] = None,
    tag_config: Optional[ScenarioTagConfig] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, parse_dates=["ts"]).set_index("ts")
    ef = from_result_columns(
        df,
        actual_col=actual_col,
        pred_col=pred_col,
        ts_index=True,
        attach_tags=with_scenario_tags,
        tag_config=tag_config,
    )
    baseline = None
    if baseline_path is not None and baseline_path.is_file():
        baseline = load_baseline_from_naive_summary_csv(
            baseline_path, baseline_task, baseline_variant
        )
    elif auto_baseline:
        ab = auto_baseline.strip().lower()
        if ab != "lag24h":
            raise ValueError(f"不支持的 auto_baseline={auto_baseline!r}，当前仅支持 lag24h")
        baseline = _compute_auto_lag24h_baseline_metrics(
            df,
            actual_col=actual_col,
            task_type=task_type,
            include_extended=include_extended,
        )
    return evaluate_model_predictions(
        ef,
        baseline_metrics=baseline,
        task_type=task_type,
        include_extended=include_extended,
        segment_cols=segment_cols,
    )


def write_metrics_json(ev: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(ev), f, ensure_ascii=False, indent=2)


def quick_shape_report(actual, pred, index, include_extended: bool = True) -> Dict[str, Any]:
    """兼容旧 compute_shape_report(actual, pred, index) 的快捷接口。"""
    ef = to_eval_frame(index, actual, pred)
    return compute_shape_metrics(ef, include_extended=include_extended)
