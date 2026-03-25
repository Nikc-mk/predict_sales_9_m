from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class CappingStats:
    capped_points: int
    thresholds: Dict[str, Tuple[float, float]]


def prepare_panel_data(
    df: pd.DataFrame,
    date_col: str,
    store_col: str,
    target_col: str,
    qty_col: str,
    end_date: date,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([store_col, date_col])

    panels = []
    for store, group in df.groupby(store_col, sort=False):
        group = group.sort_values(date_col)
        open_date = group.loc[group[target_col] > 0, date_col].min()
        if pd.isna(open_date):
            open_date = group[date_col].min()
        full_index = pd.date_range(open_date, pd.Timestamp(end_date), freq="D")
        filled = (
            group.set_index(date_col)
            .reindex(full_index)
            .assign(**{store_col: store})
        )
        filled[target_col] = filled[target_col].fillna(0.0)
        filled[qty_col] = filled[qty_col].fillna(0.0)
        filled["open_date"] = pd.Timestamp(open_date)
        filled = filled.reset_index().rename(columns={"index": date_col})
        panels.append(filled)

    panel_df = pd.concat(panels, ignore_index=True)
    return panel_df


def cap_spikes(
    df: pd.DataFrame,
    store_col: str,
    target_col: str,
    qty_col: str,
    sigma: float,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, CappingStats]:
    df = df.copy()
    thresholds: Dict[str, Tuple[float, float]] = {}
    capped_points = 0

    df["is_spike"] = 0
    for store, group_idx in df.groupby(store_col).groups.items():
        values = df.loc[group_idx, target_col].astype(float)
        mean = values.mean()
        std = values.std(ddof=0)
        upper = mean + sigma * std
        lower = max(0.0, mean - sigma * std)
        thresholds[str(store)] = (lower, upper)

        original = df.loc[group_idx, target_col].astype(float)
        clipped = original.clip(lower=lower, upper=upper)
        spikes = original.ne(clipped)
        capped_points += int(spikes.sum())
        df.loc[group_idx, target_col] = clipped
        df.loc[group_idx, "is_spike"] = spikes.astype(int)

    if logger:
        logger.info("Capping applied. sigma=%.2f capped_points=%d", sigma, capped_points)
        for store, (lower, upper) in thresholds.items():
            logger.info("Capping threshold %s: [%.3f, %.3f]", store, lower, upper)

    return df, CappingStats(capped_points=capped_points, thresholds=thresholds)


def validate_input(df: pd.DataFrame, date_col: str, target_col: str, store_col: str) -> None:
    if df[target_col].lt(0).any():
        raise ValueError("Negative sales detected. Please resolve before training.")
    if df.duplicated([date_col, store_col]).any():
        raise ValueError("Duplicate store-date rows detected.")
