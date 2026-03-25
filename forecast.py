from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ForecastConfig
from features import add_holiday_features, add_store_age_features, add_time_features, HolidayFrame


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    clip_bounds: Dict[str, Tuple[float, float]]


def _static_lag_roll(
    history: pd.Series,
    lags: List[int],
    windows: List[int],
) -> Dict[str, float]:
    values = {}
    for lag in lags:
        values[f"lag_sales_amount_{lag}"] = history.iloc[-lag] if len(history) >= lag else np.nan
    for window in windows:
        window_vals = history.iloc[-window:] if len(history) >= window else history
        values[f"roll_mean_sales_amount_{window}"] = window_vals.mean() if len(window_vals) else np.nan
        values[f"roll_std_sales_amount_{window}"] = window_vals.std(ddof=0) if len(window_vals) else np.nan
    return values


def _static_lag_roll_qty(
    history: pd.Series,
    lags: List[int],
    windows: List[int],
) -> Dict[str, float]:
    values = {}
    for lag in lags:
        values[f"lag_sales_qty_{lag}"] = history.iloc[-lag] if len(history) >= lag else np.nan
    for window in windows:
        window_vals = history.iloc[-window:] if len(history) >= window else history
        values[f"roll_mean_sales_qty_{window}"] = window_vals.mean() if len(window_vals) else np.nan
        values[f"roll_std_sales_qty_{window}"] = window_vals.std(ddof=0) if len(window_vals) else np.nan
    return values


def _build_feature_row(
    base_row: Dict,
    date_value: pd.Timestamp,
    holiday_frame: HolidayFrame,
    open_date: pd.Timestamp,
) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                **base_row,
                "posting_date": date_value,
                "open_date": open_date,
                "is_spike": 0,
            }
        ]
    )
    df = add_time_features(df, "posting_date")
    df = add_holiday_features(df, "posting_date", holiday_frame)
    df = add_store_age_features(df, "posting_date", "open_date")
    return df


def forecast_recursive(
    history_df: pd.DataFrame,
    config: ForecastConfig,
    holiday_frame: HolidayFrame,
    feature_cols: List[str],
    quantile_models: Dict[float, object],
    qty_model: object,
    forecast_start: date,
    forecast_end: date,
) -> ForecastResult:
    history_df = history_df.sort_values([config.store_col, config.date_col])
    clip_bounds: Dict[str, Tuple[float, float]] = {}
    forecasts = []

    for store, group in history_df.groupby(config.store_col):
        group = group.sort_values(config.date_col)
        last_date = group[config.date_col].max()
        base_dates = pd.date_range(forecast_start, forecast_end, freq="D")

        sales_history = group.set_index(config.date_col)[config.target_col].astype(float)
        qty_history = group.set_index(config.date_col)[config.qty_col].astype(float)

        min_hist = sales_history.min()
        max_hist = sales_history.max()
        lower_bound = config.clip_lower_mult * min_hist
        upper_bound = config.clip_upper_mult * max_hist
        clip_bounds[str(store)] = (lower_bound, upper_bound)

        static_sales = _static_lag_roll(sales_history, config.lags, config.rolling_windows)
        static_qty = _static_lag_roll_qty(qty_history, config.lags, config.rolling_windows)

        recursive_series = sales_history.copy()
        recursive_qty = qty_history.copy()

        for step, d in enumerate(base_dates, start=1):
            base_row = {config.store_col: store}
            use_recursive = step <= config.recursive_horizon_days

            if use_recursive:
                sales_source = recursive_series
                qty_source = recursive_qty
                lag_vals = _static_lag_roll(sales_source, config.lags, config.rolling_windows)
                qty_vals = _static_lag_roll_qty(qty_source, config.lags, config.rolling_windows)
            else:
                lag_vals = static_sales
                qty_vals = static_qty

            row = _build_feature_row(base_row, d, holiday_frame, group["open_date"].iloc[0])
            for key, value in lag_vals.items():
                row[key] = value
            for key, value in qty_vals.items():
                row[key] = value

            row[config.store_col] = row[config.store_col].astype("category")
            X_row = row[feature_cols]

            preds = {alpha: model.predict(X_row)[0] for alpha, model in quantile_models.items()}
            qty_pred = qty_model.predict(X_row)[0]

            median = preds.get(0.5, np.median(list(preds.values())))
            clipped_median = float(np.clip(median, lower_bound, upper_bound))
            preds = {k: float(np.clip(v, lower_bound, upper_bound)) for k, v in preds.items()}

            forecasts.append(
                {
                    "posting_date": d,
                    config.store_col: store,
                    "sales_amount_pred": clipped_median,
                    "sales_amount_pred_lower": preds[min(preds.keys())],
                    "sales_amount_pred_upper": preds[max(preds.keys())],
                    "sales_qty_pred": float(max(0.0, qty_pred)),
                }
            )

            if use_recursive:
                recursive_series = pd.concat(
                    [recursive_series, pd.Series([clipped_median], index=[d])]
                )
                recursive_qty = pd.concat(
                    [recursive_qty, pd.Series([max(0.0, qty_pred)], index=[d])]
                )

    forecast_df = pd.DataFrame(forecasts)
    return ForecastResult(forecast=forecast_df, clip_bounds=clip_bounds)
