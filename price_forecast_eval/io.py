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
