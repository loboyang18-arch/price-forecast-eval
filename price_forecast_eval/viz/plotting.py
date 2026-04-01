from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ScenarioName = str

APPENDIX_SCENARIO_PLOTS: List[Tuple[str, str, Any, str, str]] = [
    ("vol_typical", "tag_vol_class", "typical", "振幅分档 typical", "da_vol_typical_days.png"),
    ("vol_mid", "tag_vol_class", "mid", "振幅分档 mid", "da_vol_mid_days.png"),
    ("vol_high_vol", "tag_vol_class", "high_vol", "振幅分档 high_vol", "da_vol_high_vol_days.png"),
    ("weekend", "tag_weekend", 1, "周末", "da_weekend_days.png"),
    ("weekday", "tag_weekend", 0, "工作日", "da_weekday_days.png"),
    ("extreme", "tag_extreme", 1, "极端振幅(P90+)", "da_extreme_days.png"),
    ("holiday", "tag_holiday", 1, "节假日", "da_holiday_days.png"),
]


def setup_cn_font() -> None:
    for p in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["mathtext.fontset"] = "cm"


def load_prediction_csv(
    path: Path, actual_col: str = "actual", pred_col: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, parse_dates=["ts"])
    if "ts" in df.columns:
        df = df.set_index("ts")
    df = df.sort_index()
    if actual_col not in df.columns:
        raise ValueError(f"缺少列 {actual_col}: {path}")
    if pred_col is None:
        candidates = [c for c in df.columns if c != actual_col]
        if not candidates:
            raise ValueError(f"未找到预测列: {path}")
        pred_col = candidates[0]
    if pred_col not in df.columns:
        raise KeyError(pred_col)
    return df, pred_col


def select_scenario_days(
    results: pd.DataFrame, pred_col: str, category: ScenarioName, n: int = 6
) -> List:
    daily = results.groupby(results.index.date).agg(actual_mean=("actual", "mean"))
    daily["mae"] = results.groupby(results.index.date).apply(
        lambda g: np.mean(np.abs(g["actual"] - g[pred_col]))
    )
    if category == "typical":
        med = daily["mae"].median()
        daily["dist"] = (daily["mae"] - med).abs()
        return list(daily.nsmallest(n, "dist").index)
    if category == "high":
        return list(daily.nlargest(n, "actual_mean").index)
    if category == "low":
        return list(daily.nsmallest(n, "actual_mean").index)
    return []


def select_days_appendix(
    work: pd.DataFrame, tagged_by_day: pd.DataFrame, col: str, value: Any, n: int
) -> List:
    if col not in tagged_by_day.columns:
        return []
    sub = tagged_by_day[tagged_by_day[col] == value]
    if sub.empty:
        return []
    out: List = []
    for d in sorted(sub.index):
        ddt = pd.Timestamp(d).date()
        if len(work[work.index.date == ddt]) == 24:
            out.append(ddt)
        if len(out) >= n:
            break
    return out


def plot_day_overlay(
    results: pd.DataFrame,
    days: Sequence,
    title: str,
    filename: str,
    pred_col: str,
    pred_label: str,
    out_dir: Path,
    ylim: Optional[Tuple[float, float]] = (250, 500),
) -> None:
    setup_cn_font()
    n, cols = len(days), 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    last_i = 0
    for i, d in enumerate(sorted(days)):
        ax = axes[i // cols][i % cols]
        day_data = results.loc[str(d)]
        if len(day_data) != 24:
            continue
        last_i = i
        ax.plot(day_data["actual"].values, "k-", linewidth=2.0, label="实际", zorder=3)
        ax.plot(day_data[pred_col].values, "#E91E63", linewidth=1.5, alpha=0.9, label=pred_label)
        corr = np.corrcoef(day_data["actual"].values, day_data[pred_col].values)[0, 1]
        mae = np.mean(np.abs(day_data["actual"].values - day_data[pred_col].values))
        ax.set_title(f"{d}  r={corr:.2f}  MAE={mae:.1f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)

    for j in range(last_i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / filename)


def plot_weekly_timeline(
    results: pd.DataFrame,
    pred_col: str,
    pred_label: str,
    out_dir: Path,
    file_prefix: str = "da_week",
    ylim: Optional[Tuple[float, float]] = (250, 500),
) -> int:
    setup_cn_font()
    results = results.sort_index()
    all_dates = sorted(set(results.index.date))
    if not all_dates:
        return 0

    weeks: List[List] = []
    week = [all_dates[0]]
    for d in all_dates[1:]:
        if (d - week[0]).days >= 7:
            weeks.append(week)
            week = [d]
        else:
            week.append(d)
    if week:
        weeks.append(week)

    n_saved = 0
    for wi, week_dates in enumerate(weeks):
        start_d, end_d = week_dates[0], week_dates[-1]
        chunk = results.loc[(results.index.date >= start_d) & (results.index.date <= end_d)]
        if len(chunk) < 24:
            continue
        fig, ax = plt.subplots(figsize=(18, 5))
        x = range(len(chunk))
        ax.plot(x, chunk["actual"].values, "k-", linewidth=1.8, label="实际", zorder=3)
        ax.plot(x, chunk[pred_col].values, "#E91E63", linewidth=1.3, alpha=0.85, label=pred_label)

        day_boundaries, tick_positions, tick_labels = [], [], []
        for d in week_dates:
            idxs = np.where(chunk.index.date == d)[0]
            if len(idxs) > 0:
                day_boundaries.append(idxs[0])
                tick_positions.append(idxs[0] + 12)
                a = chunk["actual"].values[chunk.index.date == d]
                p = chunk[pred_col].values[chunk.index.date == d]
                if len(a) == 24 and np.std(a) > 1e-6 and np.std(p) > 1e-6:
                    tick_labels.append(f"{d}\nr={np.corrcoef(a, p)[0, 1]:.2f}")
                else:
                    tick_labels.append(str(d))
        for bd in day_boundaries[1:]:
            ax.axvline(bd, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{pred_label} — 第{wi+1}周 ({start_d} ~ {end_d})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fname = f"{file_prefix}{wi+1}.png"
        plt.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
        plt.close()
        n_saved += 1
        logger.info("Saved: %s", out_dir / fname)
    return n_saved


def plot_full_test_timeline(
    results: pd.DataFrame,
    pred_col: str,
    pred_label: str,
    out_dir: Path,
    filename: str = "da_full_test.png",
    ylim: Optional[Tuple[float, float]] = (250, 500),
) -> None:
    setup_cn_font()
    results = results.sort_index()
    if results.empty:
        logger.warning("全测试集为空，跳过 %s", filename)
        return
    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(results["actual"].values, "k-", linewidth=1.5, label="实际", zorder=3)
    ax.plot(results[pred_col].values, "#E91E63", linewidth=1.0, alpha=0.85, label=pred_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{pred_label} — 全测试集", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / filename)
