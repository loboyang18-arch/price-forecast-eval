# price-forecast-eval

电价预测模型量化评估与可视化工具包。

## 安装

```bash
pip install git+https://github.com/<org>/price-forecast-eval.git
```

开发模式：

```bash
pip install -e .
```

## Python API

```python
from price_forecast_eval import evaluate_model_predictions
from price_forecast_eval import evaluate_predictions_csv, write_metrics_json
from price_forecast_eval.viz import run_standard_visualization
```

## CLI

```bash
# 评估
price-forecast-eval eval output/experiments/v16d_default/da_result.csv \
  --task da --with-scenario-tags --out output/experiments/v16d_default/metrics.json

# 可视化
price-forecast-eval viz output/experiments/v16d_default/da_result.csv \
  --label V16d --out-dir output/experiments/v16d_default/plots

# 一次完成评估 + 可视化
price-forecast-eval run output/experiments/v16d_default/da_result.csv \
  --task da --label V16d --out-dir output/experiments/v16d_default/plots
```
