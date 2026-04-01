# price-forecast-eval

电价预测模型量化评估与可视化工具包。

## 安装

```bash
pip install git+https://github.com/loboyang18-arch/price-forecast-eval.git
```

开发模式：

```bash
pip install -e .
```

## 功能概览

- 统一指标计算：`MAE/RMSE`、shape 指标、分场景指标、composite score
- 统一 CSV 评估入口：`da_result.csv` 直接生成 `metrics.json`
- 统一可视化入口：附录分场景图 + 分周图 + 全测试集图
- 统一 CLI：`eval` / `viz` / `run`

## Python API 快速示例

```python
from price_forecast_eval import evaluate_model_predictions
from price_forecast_eval import evaluate_predictions_csv, write_metrics_json
from price_forecast_eval.viz import run_standard_visualization
```

### 1) 从 DataFrame 直接评估

输入最少列要求：`date`, `hour`, `y_true`, `y_pred`

```python
import pandas as pd
from price_forecast_eval import evaluate_model_predictions

df = pd.DataFrame(
    {
        "date": ["2026-01-01"] * 24,
        "hour": list(range(24)),
        "y_true": [300 + i for i in range(24)],
        "y_pred": [302 + i for i in range(24)],
    }
)
ev = evaluate_model_predictions(df, task_type="da", include_extended=True)
print(ev["point_metrics"])
```

### 2) 从预测 CSV 评估并写 JSON

输入 CSV 需要含 `ts` 和预测列（默认 `actual/predicted`）。

```python
from pathlib import Path
from price_forecast_eval import evaluate_predictions_csv, write_metrics_json

ev = evaluate_predictions_csv(
    Path("output/experiments/v16d_default/da_result.csv"),
    task_type="da",
    with_scenario_tags=True,
)
write_metrics_json(ev, Path("output/experiments/v16d_default/metrics.json"))
```

### 3) 绘图

```python
from price_forecast_eval.viz import run_standard_visualization

run_standard_visualization(
    "output/experiments/v16d_default/da_result.csv",
    out_dir="output/experiments/v16d_default/plots",
    label="V16d",
    mode="appendix",
    weekly=True,
)
```

## CLI

```bash
# 评估
price-forecast-eval eval output/experiments/v16d_default/da_result.csv \
  --task da --with-scenario-tags --out output/experiments/v16d_default/metrics.json

# 评估 + 自动基线（从 actual 计算 lag24h baseline，并输出 composite）
price-forecast-eval eval output/experiments/v16d_default/da_result.csv \
  --task da --with-scenario-tags --auto-baseline lag24h \
  --out output/experiments/v16d_default/metrics_auto_baseline.json

# 可视化
price-forecast-eval viz output/experiments/v16d_default/da_result.csv \
  --label V16d --out-dir output/experiments/v16d_default/plots

# 一次完成评估 + 可视化
price-forecast-eval run output/experiments/v16d_default/da_result.csv \
  --task da --label V16d --out-dir output/experiments/v16d_default/plots
```

## CLI 参数说明（常用）

### `eval`

- `csv_path`：预测文件路径（必须含 `ts`）
- `--actual`：真实值列名，默认 `actual`
- `--pred`：预测列名，默认 `predicted`
- `--task`：`da` 或 `rt`
- `--out`：输出 JSON 路径（默认同目录 `metrics.json`）
- `--with-scenario-tags`：自动打附录场景标签
- `--segment-cols`：分场景列，逗号分隔
- `--baseline` / `--baseline-task` / `--baseline-variant`：外部基线输入（用于 composite）
- `--auto-baseline lag24h`：从当前 CSV 的 `actual` 自动构造 `t-24h` 基线并计算 composite

说明：
- 若同时提供 `--baseline` 与 `--auto-baseline`，优先使用 `--baseline`。
- 未提供任一基线输入时，`composite` 会是 `null`。

### `viz`

- `result_csv`：预测文件路径（必须含 `ts`）
- `--label`：图标题模型名
- `--out-dir`：输出目录，默认 `<csv目录>/plots`
- `--mode`：`appendix`（默认）或 `legacy`
- `--appendix-scenarios`：附录模式下指定场景子集
- `--no-weekly`：不画分周图
- `--n-days`：每个场景抽样天数
- `--ylim`：y 轴范围，例如 `250,500` 或 `auto`

### `run`

等价于先执行 `eval` 再执行 `viz`，参数是两者并集。
