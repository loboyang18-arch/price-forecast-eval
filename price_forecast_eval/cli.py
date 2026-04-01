from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .io import evaluate_predictions_csv, write_metrics_json
from .viz import run_standard_visualization

DEFAULT_SEGMENT_COLS = ["tag_weekend", "tag_vol_class", "tag_extreme", "tag_holiday"]


def _parse_ylim(text: str):
    if text.strip().lower() == "auto":
        return None
    parts = text.split(",")
    return (float(parts[0].strip()), float(parts[1].strip()))


def _parse_list(text: str | None):
    if not text:
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Price forecast evaluation toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    eval_p = sub.add_parser("eval", help="计算评估指标并输出 metrics.json")
    eval_p.add_argument("csv_path", type=Path, help="含 ts 列与 actual/pred 列的 CSV")
    eval_p.add_argument("--actual", default="actual")
    eval_p.add_argument("--pred", default="predicted")
    eval_p.add_argument("--task", default="da", choices=("da", "rt"))
    eval_p.add_argument("--out", type=Path, default=None, help="输出 JSON 路径")
    eval_p.add_argument("--baseline", type=Path, default=None, help="naive summary.csv")
    eval_p.add_argument("--baseline-task", default="da")
    eval_p.add_argument("--baseline-variant", default="lag24h")
    eval_p.add_argument("--no-extended", action="store_true")
    eval_p.add_argument("--with-scenario-tags", action="store_true")
    eval_p.add_argument(
        "--segment-cols",
        default=None,
        help="逗号分隔；与 --with-scenario-tags 联用。默认 " + ",".join(DEFAULT_SEGMENT_COLS),
    )

    viz_p = sub.add_parser("viz", help="生成分场景与分周图")
    viz_p.add_argument("result_csv", type=Path, help="da_result.csv 或 pred_test.csv（含 ts）")
    viz_p.add_argument("--label", default="Model")
    viz_p.add_argument("--out-dir", type=Path, default=None)
    viz_p.add_argument("--actual", default="actual")
    viz_p.add_argument("--pred", default=None)
    viz_p.add_argument("--mode", choices=("appendix", "legacy"), default="appendix")
    viz_p.add_argument("--scenarios", default="typical,high,low")
    viz_p.add_argument("--appendix-scenarios", default=None)
    viz_p.add_argument("--no-weekly", action="store_true")
    viz_p.add_argument("--n-days", type=int, default=6)
    viz_p.add_argument("--ylim", default="250,500")

    run_p = sub.add_parser("run", help="先评估再绘图")
    run_p.add_argument("csv_path", type=Path)
    run_p.add_argument("--actual", default="actual")
    run_p.add_argument("--pred", default="predicted")
    run_p.add_argument("--task", default="da", choices=("da", "rt"))
    run_p.add_argument("--out", type=Path, default=None)
    run_p.add_argument("--baseline", type=Path, default=None)
    run_p.add_argument("--baseline-task", default="da")
    run_p.add_argument("--baseline-variant", default="lag24h")
    run_p.add_argument("--no-extended", action="store_true")
    run_p.add_argument("--with-scenario-tags", action="store_true")
    run_p.add_argument("--segment-cols", default=None)
    run_p.add_argument("--label", default="Model")
    run_p.add_argument("--out-dir", type=Path, default=None)
    run_p.add_argument("--mode", choices=("appendix", "legacy"), default="appendix")
    run_p.add_argument("--scenarios", default="typical,high,low")
    run_p.add_argument("--appendix-scenarios", default=None)
    run_p.add_argument("--no-weekly", action="store_true")
    run_p.add_argument("--n-days", type=int, default=6)
    run_p.add_argument("--ylim", default="250,500")
    return p


def _run_eval(args) -> Path:
    seg_cols = None
    if args.with_scenario_tags:
        seg_cols = _parse_list(args.segment_cols) or list(DEFAULT_SEGMENT_COLS)
    ev = evaluate_predictions_csv(
        args.csv_path,
        actual_col=args.actual,
        pred_col=args.pred,
        task_type=args.task,
        include_extended=not args.no_extended,
        baseline_path=args.baseline,
        baseline_task=args.baseline_task,
        baseline_variant=args.baseline_variant,
        with_scenario_tags=bool(args.with_scenario_tags),
        segment_cols=seg_cols,
    )
    out_path = args.out or (args.csv_path.parent / "metrics.json")
    write_metrics_json(ev, out_path)
    print(out_path)
    return out_path


def _run_viz(args) -> None:
    run_standard_visualization(
        args.result_csv if hasattr(args, "result_csv") else args.csv_path,
        out_dir=args.out_dir,
        label=args.label,
        actual_col=args.actual,
        pred_col=None if args.pred == "predicted" and hasattr(args, "csv_path") else args.pred,
        mode=args.mode,
        scenarios=_parse_list(args.scenarios) or ["typical", "high", "low"],
        appendix_scenarios=_parse_list(args.appendix_scenarios),
        weekly=not args.no_weekly,
        n_days_per_scenario=args.n_days,
        ylim=_parse_ylim(args.ylim),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    try:
        if args.cmd == "eval":
            _run_eval(args)
        elif args.cmd == "viz":
            _run_viz(args)
        elif args.cmd == "run":
            _run_eval(args)
            _run_viz(args)
        else:
            raise ValueError(f"unknown cmd: {args.cmd}")
    except Exception:
        logging.exception("command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
