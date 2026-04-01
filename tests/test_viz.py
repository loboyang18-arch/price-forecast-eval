from pathlib import Path

import numpy as np
import pandas as pd

from price_forecast_eval.viz import run_standard_visualization


def test_viz_smoke(tmp_path: Path):
    ts = pd.date_range("2026-01-01", periods=24 * 10, freq="h")
    y = 300 + 20 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24)
    p = y + 2
    df = pd.DataFrame({"ts": ts, "actual": y, "predicted": p})
    csv_path = tmp_path / "da_result.csv"
    out_dir = tmp_path / "plots"
    df.to_csv(csv_path, index=False)
    run_standard_visualization(csv_path, out_dir=out_dir, label="T")
    assert (out_dir / "da_full_test.png").exists()
