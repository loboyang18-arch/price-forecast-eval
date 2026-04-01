"""附录标准 metrics 边界用例。"""

import numpy as np
import pandas as pd

from price_forecast_eval.adapters import to_eval_frame
from price_forecast_eval.evaluate import evaluate_model_predictions
from price_forecast_eval.io import evaluate_predictions_csv
from price_forecast_eval.scenario_tags import attach_scenario_tags
from price_forecast_eval.shape_metrics import compute_shape_metrics
from price_forecast_eval.validation import validate_eval_frame


def _three_flat_pred_days(n_days: int = 3) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        for h in range(24):
            rows.append(
                {
                    "date": f"2026-01-{d+1:02d}",
                    "hour": h,
                    "y_true": 300.0 + float(h) + d,
                    "y_pred": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_pred_flat_line_still_counts_valid_days():
    df = _three_flat_pred_days(3)
    sm = compute_shape_metrics(df, include_extended=False)
    assert sm["valid_shape_days"] == 3
    assert sm["profile_corr"] == 0.0
    assert sm["neg_corr_day_ratio"] == 0.0


def test_negative_corr_day():
    rows = []
    for h in range(24):
        rows.append(
            {
                "date": "2026-06-01",
                "hour": h,
                "y_true": float(h),
                "y_pred": float(23 - h),
            }
        )
    df = pd.DataFrame(rows)
    sm = compute_shape_metrics(df, include_extended=False)
    assert sm["valid_shape_days"] == 1
    assert sm["neg_corr_day_ratio"] == 1.0
    assert sm["neg_corr_day_count"] == 1


def test_incomplete_day_excluded():
    rows = []
    for h in range(23):
        rows.append(
            {
                "date": "2026-06-02",
                "hour": h,
                "y_true": 1.0,
                "y_pred": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    sm = compute_shape_metrics(df, include_extended=False)
    assert sm["valid_shape_days"] == 0


def test_evaluate_merge_and_validation():
    idx = pd.date_range("2026-02-09", periods=48, freq="h")
    y = np.linspace(300, 350, 48)
    p = y + np.random.default_rng(0).normal(0, 5, 48)
    ef = to_eval_frame(idx, y, p)
    v = validate_eval_frame(ef)
    assert v["ok"] is True
    ev = evaluate_model_predictions(ef, baseline_metrics=None, task_type="da")
    assert "point_metrics" in ev
    assert "shape_metrics" in ev
    assert ev["composite"] is None


def _two_days_curve(low_amp: bool, day_offset: int):
    rows = []
    d = f"2026-03-{10 + day_offset:02d}"
    base = 300.0 if low_amp else 200.0
    span = 5.0 if low_amp else 80.0
    for h in range(24):
        rows.append(
            {
                "date": d,
                "hour": h,
                "y_true": base + (span * h / 23.0),
                "y_pred": base + (span * h / 23.0) + 1.0,
            }
        )
    return rows


def test_scenario_tags_weekend_and_vol():
    assert pd.Timestamp("2026-03-14").weekday() == 5
    rows = []
    rows.extend(_two_days_curve(low_amp=True, day_offset=0))
    rows.extend(_two_days_curve(low_amp=False, day_offset=1))
    rows.extend(_two_days_curve(low_amp=True, day_offset=2))
    rows.extend(_two_days_curve(low_amp=False, day_offset=4))
    df = pd.DataFrame(rows)
    tagged = attach_scenario_tags(df)
    sat = tagged[tagged["date"] == "2026-03-14"]
    assert sat["tag_weekend"].iloc[0] == 1
    tue = tagged[tagged["date"] == "2026-03-10"]
    assert tue["tag_weekend"].iloc[0] == 0
    assert tagged["tag_vol_class"].notna().all()


def test_evaluate_with_scenario_segments():
    rows = []
    for di in range(3):
        rows.extend(_two_days_curve(low_amp=(di % 2 == 0), day_offset=di))
    df = pd.DataFrame(rows)
    tagged = attach_scenario_tags(df)
    ev = evaluate_model_predictions(
        tagged,
        baseline_metrics=None,
        task_type="da",
        segment_cols=["tag_weekend", "tag_vol_class"],
    )
    assert "segments" in ev
    assert "tag_weekend" in ev["segments"]
    assert "overall" in ev["segments"]["tag_weekend"]


def test_to_eval_frame_attach_tags():
    idx = pd.date_range("2026-03-15", periods=24, freq="h")
    y = np.linspace(100, 200, 24)
    p = y + 1.0
    ef = to_eval_frame(idx, y, p, attach_tags=True)
    assert "tag_weekend" in ef.columns
    assert "tag_vol_class" in ef.columns


def test_auto_baseline_lag24h_composite_not_null(tmp_path):
    ts = pd.date_range("2026-01-01", periods=24 * 4, freq="h")
    y = 300 + 20 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24)
    p = y + 2.0
    csv_path = tmp_path / "pred.csv"
    pd.DataFrame({"ts": ts, "actual": y, "predicted": p}).to_csv(csv_path, index=False)

    ev = evaluate_predictions_csv(
        csv_path,
        actual_col="actual",
        pred_col="predicted",
        task_type="da",
        auto_baseline="lag24h",
        with_scenario_tags=False,
    )
    assert ev["composite"] is not None
    assert "composite_score" in ev["composite"]
