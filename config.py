from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass(frozen=True)
class ForecastConfig:
    target_col: str = "sales_amount"
    qty_col: str = "sales_qty"
    date_col: str = "posting_date"
    store_col: str = "shop_account_name"
    open_date_col: str = "open_date"

    forecast_start_date: Optional[date] = None
    forecast_end_date: date = date(2026, 12, 31)

    holdout_days: int = 45
    backtest_iters: int = 3
    backtest_gap_days: int = 2
    backtest_test_days: int = 30

    lags: List[int] = (1, 3, 7, 14, 28, 56)
    rolling_windows: List[int] = (7, 14, 28, 56)

    recursive_horizon_days: int = 120
    clip_lower_mult: float = 0.5
    clip_upper_mult: float = 2.5

    enable_capping: bool = False
    capping_sigma: float = 4.0

    quantile_alphas: List[float] = (0.1, 0.5, 0.9)

    random_state: int = 42

    long_horizon_blend_weight: float = 0.9
    interval_target_coverage: float = 0.8
    interval_calibration_min_samples: int = 20
