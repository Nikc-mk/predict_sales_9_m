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
    origin_date: date,
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
    df = add_time_features(df, "posting_date", origin_date=origin_date)
    df = add_holiday_features(df, "posting_date", holiday_frame)
    df = add_store_age_features(df, "posting_date", "open_date")
    return df


def forecast_recursive(
    history_df: pd.DataFrame,
    config: ForecastConfig,
    holiday_frame: HolidayFrame,
    feature_cols: List[str],
    quantile_models: Dict[float, object],
    long_feature_cols: List[str],
    long_quantile_models: Dict[float, object],
    qty_model: object,
    long_qty_model: object,
    forecast_start: date,
    forecast_end: date,
    origin_date: date,
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

        group = group.copy()
        group["year"] = group[config.date_col].dt.year
        group["month"] = group[config.date_col].dt.month
        sales_monthly_year = (
            group.groupby(["year", "month"])[config.target_col].median().to_dict()
        )
        qty_monthly_year = (
            group.groupby(["year", "month"])[config.qty_col].median().to_dict()
        )
        sales_monthly_median = group.groupby("month")[config.target_col].median().to_dict()
        qty_monthly_median = group.groupby("month")[config.qty_col].median().to_dict()
        sales_overall_median = float(sales_history.median())
        qty_overall_median = float(qty_history.median())
        max_hist_year = int(group["year"].max())

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

            row = _build_feature_row(
                base_row,
                d,
                holiday_frame,
                group["open_date"].iloc[0],
                origin_date,
            )
            for key, value in lag_vals.items():
                row[key] = value
            for key, value in qty_vals.items():
                row[key] = value

            row[config.store_col] = row[config.store_col].astype("category")
            if use_recursive:
                X_row = row[feature_cols]
                preds = {
                    alpha: model.predict(X_row)[0] for alpha, model in quantile_models.items()
                }
                qty_pred = qty_model.predict(X_row)[0]
            else:
                X_row = row[long_feature_cols]
                preds = {
                    alpha: model.predict(X_row)[0]
                    for alpha, model in long_quantile_models.items()
                }
                qty_pred = long_qty_model.predict(X_row)[0]

                month_key = d.month
                candidate_years = sorted({y for (y, m) in sales_monthly_year if m == month_key})
                candidate_years = [y for y in candidate_years if y <= d.year - 1] or candidate_years
                baseline_year = candidate_years[-1] if candidate_years else max_hist_year
                prev_year = baseline_year - 1

                sales_base = sales_monthly_year.get(
                    (baseline_year, month_key),
                    sales_monthly_median.get(month_key, sales_overall_median),
                )
                sales_prev = sales_monthly_year.get(
                    (prev_year, month_key),
                    sales_monthly_median.get(month_key, sales_overall_median),
                )
                qty_base = qty_monthly_year.get(
                    (baseline_year, month_key),
                    qty_monthly_median.get(month_key, qty_overall_median),
                )
                qty_prev = qty_monthly_year.get(
                    (prev_year, month_key),
                    qty_monthly_median.get(month_key, qty_overall_median),
                )

                sales_growth = sales_base / sales_prev if sales_prev not in (0, None) else 1.0
                qty_growth = qty_base / qty_prev if qty_prev not in (0, None) else 1.0
                sales_growth = float(np.clip(sales_growth, 0.5, 2.0))
                qty_growth = float(np.clip(qty_growth, 0.5, 2.0))

                sales_baseline = float(sales_base * sales_growth)
                qty_baseline = float(qty_base * qty_growth)
                blend_w = config.long_horizon_blend_weight
                preds = {k: blend_w * v + (1.0 - blend_w) * sales_baseline for k, v in preds.items()}
                qty_pred = blend_w * qty_pred + (1.0 - blend_w) * qty_baseline

            median = preds.get(0.5, np.median(list(preds.values())))
            month_floor = float(sales_baseline if not use_recursive else sales_monthly_median.get(d.month, sales_overall_median))
            clipped_median = float(np.clip(median, lower_bound, upper_bound))
            clipped_median = max(clipped_median, month_floor)
            preds = {k: float(np.clip(v, lower_bound, upper_bound)) for k, v in preds.items()}
            preds = {k: max(v, month_floor) for k, v in preds.items()}
            qty_floor = float(qty_baseline if not use_recursive else qty_monthly_median.get(d.month, qty_overall_median))
            qty_pred = max(float(qty_pred), qty_floor)

            forecasts.append(
                {
                    "posting_date": d,
                    config.store_col: store,
                    "sales_amount_pred": clipped_median,
                    "sales_amount_pred_lower": preds[min(preds.keys())],
                    "sales_amount_pred_upper": preds[max(preds.keys())],
                    "sales_qty_pred": float(qty_pred),
                }
            )

            if use_recursive:
                recursive_series = pd.concat(
                    [recursive_series, pd.Series([clipped_median], index=[d])]
                )
                recursive_qty = pd.concat(
                    [recursive_qty, pd.Series([qty_pred], index=[d])]
                )

    forecast_df = pd.DataFrame(forecasts)
    return ForecastResult(forecast=forecast_df, clip_bounds=clip_bounds)
