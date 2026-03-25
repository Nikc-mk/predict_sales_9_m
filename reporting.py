from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from metrics import coverage, mae, rmse, wmape


def build_report(
    holdout_df: pd.DataFrame,
    group_col: str,
    actual_col: str,
    pred_col: str,
    lower_col: str,
    upper_col: str,
    output_path: Path,
    notes: Dict[str, str],
) -> None:
    total_metrics = {
        "wmape": wmape(holdout_df[actual_col], holdout_df[pred_col]),
        "mae": mae(holdout_df[actual_col], holdout_df[pred_col]),
        "rmse": rmse(holdout_df[actual_col], holdout_df[pred_col]),
        "coverage": coverage(
            holdout_df[actual_col],
            holdout_df[lower_col],
            holdout_df[upper_col],
        ),
    }

    rows = []
    for group, g in holdout_df.groupby(group_col):
        rows.append(
            {
                group_col: group,
                "wmape": wmape(g[actual_col], g[pred_col]),
                "mae": mae(g[actual_col], g[pred_col]),
                "rmse": rmse(g[actual_col], g[pred_col]),
                "coverage": coverage(g[actual_col], g[lower_col], g[upper_col]),
            }
        )
    store_metrics = pd.DataFrame(rows).sort_values("wmape")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Forecast Report\n\n")
        f.write("## Total Metrics (Holdout)\n\n")
        for key, value in total_metrics.items():
            f.write(f"- {key}: {value:.4f}\n")
        f.write("\n## Store Metrics (Holdout)\n\n")
        f.write(store_metrics.to_markdown(index=False))
        f.write("\n\n## Notes\n\n")
        for key, value in notes.items():
            f.write(f"- {key}: {value}\n")
