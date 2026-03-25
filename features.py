from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except ImportError:  # pragma: no cover
    holidays_lib = None


@dataclass
class HolidayFrame:
    holidays: pd.DataFrame
    source: str


def load_holidays(
    start_date: date,
    end_date: date,
    csv_path: Path = Path("data/ru_holidays_2024_2026.csv"),
) -> HolidayFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return HolidayFrame(holidays=df, source="csv")

    if holidays_lib is None:
        raise ImportError(
            "holidays package not installed and custom holiday CSV not found."
        )

    years = list(range(start_date.year, end_date.year + 1))
    ru_holidays = holidays_lib.country_holidays("RU", years=years)
    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["is_holiday"] = df["date"].dt.date.apply(lambda d: int(d in ru_holidays))
    df["is_pre_holiday"] = df["date"].dt.date.apply(
        lambda d: int((d + timedelta(days=1)) in ru_holidays)
    )
    df["is_post_holiday"] = df["date"].dt.date.apply(
        lambda d: int((d - timedelta(days=1)) in ru_holidays)
    )
    return HolidayFrame(holidays=df, source="holidays-lib")


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = df[date_col]
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["dayofyear"] = dt.dt.dayofyear
    df["dayofmonth"] = dt.dt.day
    return df


def add_holiday_features(df: pd.DataFrame, date_col: str, holiday_frame: HolidayFrame) -> pd.DataFrame:
    df = df.copy()
    hol = holiday_frame.holidays.rename(columns={"date": date_col})
    df = df.merge(hol, on=date_col, how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    if "is_pre_holiday" not in df:
        df["is_pre_holiday"] = 0
    if "is_post_holiday" not in df:
        df["is_post_holiday"] = 0
    df["is_pre_holiday"] = df["is_pre_holiday"].fillna(0).astype(int)
    df["is_post_holiday"] = df["is_post_holiday"].fillna(0).astype(int)
    return df


def add_store_age_features(df: pd.DataFrame, date_col: str, open_date_col: str) -> pd.DataFrame:
    df = df.copy()
    store_age_days = (df[date_col] - df[open_date_col]).dt.days
    df["store_age_days"] = store_age_days
    df["store_age_weeks"] = (store_age_days // 7).astype(int)
    df["store_age_months"] = (store_age_days // 30).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    store_col: str,
    target_cols: Iterable[str],
    lags: Iterable[int],
) -> pd.DataFrame:
    df = df.copy()
    for target in target_cols:
        for lag in lags:
            df[f"lag_{target}_{lag}"] = df.groupby(store_col)[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    store_col: str,
    target_cols: Iterable[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    df = df.copy()
    for target in target_cols:
        shifted = df.groupby(store_col)[target].shift(1)
        for window in windows:
            df[f"roll_mean_{target}_{window}"] = (
                shifted.groupby(df[store_col]).rolling(window).mean().reset_index(level=0, drop=True)
            )
            df[f"roll_std_{target}_{window}"] = (
                shifted.groupby(df[store_col]).rolling(window).std(ddof=0).reset_index(level=0, drop=True)
            )
    return df


def build_feature_frame(
    df: pd.DataFrame,
    date_col: str,
    store_col: str,
    open_date_col: str,
    target_cols: List[str],
    lags: List[int],
    rolling_windows: List[int],
    holiday_frame: HolidayFrame,
) -> pd.DataFrame:
    df = add_time_features(df, date_col)
    df = add_holiday_features(df, date_col, holiday_frame)
    df = add_store_age_features(df, date_col, open_date_col)
    df = add_lag_features(df, store_col, target_cols, lags)
    df = add_rolling_features(df, store_col, target_cols, rolling_windows)
    return df


def select_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    exclude = set(target_cols + ["posting_date", "open_date"])
    return [col for col in df.columns if col not in exclude]
