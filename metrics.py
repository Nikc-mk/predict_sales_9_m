from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - forecast)) / denom * 100.0)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def group_metrics(df: pd.DataFrame, actual_col: str, pred_col: str, group_col: str) -> pd.DataFrame:
    results = []
    for group, g in df.groupby(group_col):
        results.append(
            {
                group_col: group,
                "wmape": wmape(g[actual_col], g[pred_col]),
                "mae": mae(g[actual_col], g[pred_col]),
                "rmse": rmse(g[actual_col], g[pred_col]),
            }
        )
    return pd.DataFrame(results)
