from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_sales_data(
    start_date: date,
    end_date: date,
    n_stores: int,
    seed: int | None,
    missing_rate: float,
    zero_rate: float,
) -> pd.DataFrame:
    rng = _rng(seed)
    stores = [f"Store_{i+1:02d}" for i in range(n_stores)]
    all_rows: list[dict] = []

    total_days = (end_date - start_date).days + 1
    opening_offsets = rng.integers(0, max(1, total_days // 3), size=n_stores)

    for store, offset in zip(stores, opening_offsets):
        open_date = start_date + timedelta(days=int(offset))
        dates = pd.date_range(open_date, end_date, freq="D")
        n = len(dates)

        base = rng.uniform(20000, 80000)
        trend = rng.uniform(-2, 6)  # mild trend per day
        weekly = 1 + 0.15 * np.sin(2 * np.pi * (dates.dayofweek.values / 7))
        monthly = 1 + 0.10 * np.sin(2 * np.pi * (dates.dayofyear.values / 365))

        noise = rng.normal(0, 0.12, size=n)
        sales = base * weekly * monthly + trend * np.arange(n) + base * noise
        sales = np.maximum(0, sales)

        spikes = rng.random(n) < 0.02
        sales[spikes] *= rng.uniform(1.8, 3.5, size=spikes.sum())

        qty = np.maximum(0, sales / rng.uniform(800, 1500) + rng.normal(0, 3, size=n))

        mask_missing = rng.random(n) < missing_rate
        mask_zero = rng.random(n) < zero_rate
        sales[mask_zero] = 0
        qty[mask_zero] = 0

        for d, s, q, miss in zip(dates, sales, qty, mask_missing):
            if miss:
                continue
            all_rows.append(
                {
                    "posting_date": d.date(),
                    "shop_account_name": store,
                    "sales_amount": float(round(s, 2)),
                    "sales_qty": float(round(q, 2)),
                }
            )

    df = pd.DataFrame(all_rows)
    return df.sort_values(["shop_account_name", "posting_date"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sales data")
    parser.add_argument("--start-date", default="2024-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--n-stores", type=int, default=6, help="Number of stores (3-10 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--zero-rate", type=float, default=0.03)
    parser.add_argument("--output", default="data/data.csv")
    args = parser.parse_args()

    df = generate_sales_data(
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        n_stores=args.n_stores,
        seed=args.seed,
        missing_rate=args.missing_rate,
        zero_rate=args.zero_rate,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
