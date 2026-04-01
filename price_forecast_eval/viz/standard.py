from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from ..scenario_tags import ScenarioTagConfig, attach_scenario_tags
from .plotting import (
    APPENDIX_SCENARIO_PLOTS,
    ScenarioName,
    load_prediction_csv,
    plot_day_overlay,
    plot_full_test_timeline,
    plot_weekly_timeline,
    select_days_appendix,
    select_scenario_days,
)

logger = logging.getLogger(__name__)


def _tagged_by_day(work: pd.DataFrame, tag_config: Optional[ScenarioTagConfig] = None) -> pd.DataFrame:
    eval_df = pd.DataFrame(
        {
            "date": work.index.strftime("%Y-%m-%d"),
            "hour": work.index.hour.astype(int),
            "y_true": work["actual"].values,
        }
    )
    tagged = attach_scenario_tags(
        eval_df, date_col="date", hour_col="hour", y_true_col="y_true", config=tag_config
    )
    return tagged.groupby("date", sort=True).first()


def run_standard_visualization(
    result_csv: Path,
    *,
    out_dir: Optional[Path] = None,
    label: str = "Model",
    actual_col: str = "actual",
    pred_col: Optional[str] = None,
    mode: str = "appendix",
    scenarios: Iterable[ScenarioName] = ("typical", "high", "low"),
    appendix_scenarios: Optional[Iterable[str]] = None,
    scenario_tag_config: Optional[ScenarioTagConfig] = None,
    weekly: bool = True,
    n_days_per_scenario: int = 6,
    ylim: Optional[Tuple[float, float]] = (250, 500),
) -> None:
    result_csv = Path(result_csv).resolve()
    if out_dir is None:
        out_dir = result_csv.parent / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, pcol = load_prediction_csv(result_csv, actual_col=actual_col, pred_col=pred_col)
    work = pd.DataFrame({"actual": raw[actual_col], "pred": raw[pcol]}, index=raw.index)

    mode_l = (mode or "appendix").strip().lower()
    if mode_l == "appendix":
        plots = list(APPENDIX_SCENARIO_PLOTS)
        if appendix_scenarios is not None:
            want = {str(x).strip() for x in appendix_scenarios}
            plots = [p for p in plots if p[0] in want]
        tagged_by_day = _tagged_by_day(work, scenario_tag_config)
        for key, col, val, title_suffix, fname in plots:
            if key == "holiday":
                if "tag_holiday" not in tagged_by_day.columns or tagged_by_day["tag_holiday"].max() < 1:
                    logger.info("无节假日样本，跳过 %s", fname)
                    continue
            days = select_days_appendix(work, tagged_by_day, col, val, n_days_per_scenario)
            if not days:
                logger.warning("附录场景 %s 无完整日样本，跳过", key)
                continue
            plot_day_overlay(
                work,
                days,
                f"{label} — {title_suffix}（附录分场景）",
                fname,
                "pred",
                label,
                out_dir,
                ylim=ylim,
            )
    elif mode_l == "legacy":
        title_map = {
            "typical": "典型日曲线叠图（legacy）",
            "high": "极端高价日曲线叠图（legacy）",
            "low": "极端低价日曲线叠图（legacy）",
        }
        file_map = {
            "typical": "da_typical_days.png",
            "high": "da_high_days.png",
            "low": "da_low_days.png",
        }
        for cat in scenarios:
            if cat not in title_map:
                logger.warning("未知 legacy 场景 %s，跳过", cat)
                continue
            days = select_scenario_days(work, "pred", cat, n_days_per_scenario)
            if not days:
                logger.warning("场景 %s 无可用日", cat)
                continue
            plot_day_overlay(
                work,
                days,
                f"{label} — {title_map[cat]}",
                file_map[cat],
                "pred",
                label,
                out_dir,
                ylim=ylim,
            )
    else:
        raise ValueError(f"未知 mode={mode!r}，请用 appendix 或 legacy")

    if weekly:
        n = plot_weekly_timeline(work, "pred", label, out_dir, ylim=ylim)
        if n == 0:
            logger.warning("未生成分周图（数据不足或为空）")
    plot_full_test_timeline(work, "pred", label, out_dir, filename="da_full_test.png", ylim=ylim)
