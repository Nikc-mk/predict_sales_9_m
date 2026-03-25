from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class IntervalCalibration:
    target_coverage: float
    global_scale: float
    global_fallback_half_width: float
    store_scale: Dict[str, float]
    store_fallback_half_width: Dict[str, float]
    min_samples: int


def _safe_quantile(values: pd.Series, q: float, default: float) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return default
    return float(values.quantile(q))


def fit_interval_calibration(
    holdout_df: pd.DataFrame,
    store_col: str,
    actual_col: str,
    pred_col: str,
    lower_col: str,
    upper_col: str,
    target_coverage: float,
    min_samples: int,
) -> IntervalCalibration:
    df = holdout_df.copy()
    df["half_width"] = (df[upper_col] - df[lower_col]) / 2.0
    df["abs_err"] = (df[actual_col] - df[pred_col]).abs()

    global_fallback = _safe_quantile(df["abs_err"], 0.5, 1.0)
    global_ratio = df.loc[df["half_width"] > 0, "abs_err"] / df.loc[
        df["half_width"] > 0, "half_width"
    ]
    global_scale = max(1.0, _safe_quantile(global_ratio, target_coverage, 1.0))

    store_scale: Dict[str, float] = {}
    store_fallback: Dict[str, float] = {}
    for store, g in df.groupby(store_col):
        store_fallback[str(store)] = _safe_quantile(g["abs_err"], 0.5, global_fallback)
        ratios = g.loc[g["half_width"] > 0, "abs_err"] / g.loc[g["half_width"] > 0, "half_width"]
        if len(ratios) >= min_samples:
            store_scale[str(store)] = max(1.0, _safe_quantile(ratios, target_coverage, global_scale))
        else:
            store_scale[str(store)] = global_scale

    return IntervalCalibration(
        target_coverage=target_coverage,
        global_scale=global_scale,
        global_fallback_half_width=global_fallback,
        store_scale=store_scale,
        store_fallback_half_width=store_fallback,
        min_samples=min_samples,
    )


def apply_interval_calibration(
    df: pd.DataFrame,
    store_col: str,
    pred_col: str,
    lower_col: str,
    upper_col: str,
    calibration: IntervalCalibration,
) -> pd.DataFrame:
    result = df.copy()
    half_width = (result[upper_col] - result[lower_col]) / 2.0
    scales = result[store_col].astype(str).map(calibration.store_scale).fillna(
        calibration.global_scale
    )
    fallbacks = result[store_col].astype(str).map(calibration.store_fallback_half_width).fillna(
        calibration.global_fallback_half_width
    )
    effective_half_width = half_width.where(half_width > 0, fallbacks)
    adjusted = scales * effective_half_width

    result[lower_col] = result[pred_col] - adjusted
    result[upper_col] = result[pred_col] + adjusted
    return result
