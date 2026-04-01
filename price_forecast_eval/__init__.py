"""Price forecast evaluation toolkit public API."""

from .adapters import from_result_columns, to_eval_frame
from .composite import compute_composite_score, load_baseline_from_naive_summary_csv
from .evaluate import evaluate_model_predictions
from .io import evaluate_predictions_csv, quick_shape_report, write_metrics_json
from .point_metrics import compute_point_metrics
from .scenario_tags import ScenarioTagConfig, attach_scenario_tags
from .segment_metrics import compute_metrics_by_segment
from .shape_metrics import compute_shape_metrics
from .validation import validate_eval_frame

__all__ = [
    "to_eval_frame",
    "from_result_columns",
    "validate_eval_frame",
    "compute_point_metrics",
    "compute_shape_metrics",
    "compute_metrics_by_segment",
    "compute_composite_score",
    "load_baseline_from_naive_summary_csv",
    "evaluate_model_predictions",
    "evaluate_predictions_csv",
    "write_metrics_json",
    "quick_shape_report",
    "ScenarioTagConfig",
    "attach_scenario_tags",
]
