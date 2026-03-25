# Sales Forecasting 2026

This project implements a full pipeline for daily sales forecasting per store and total network through 2026-12-31.
It follows the technical specification provided on 2026-03-25.

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Place your input data at `data/data.csv` with columns:
`posting_date`, `shop_account_name`, `sales_amount`, `sales_qty`.

3. Run the pipeline.

```bash
python run_pipeline.py --input data/data.csv --output-dir artifacts
```

## Outputs

- `artifacts/forecast_2026.csv`
- `artifacts/models/` (LightGBM quantile models + sales_qty model)
- `artifacts/feature_config.json`
- `artifacts/report.md`
- `project_spec.toml` (Codex project specification)

## Notes

- The pipeline supports optional holiday calendar overrides via
  `data/ru_holidays_2024_2026.csv`.
- Default forecast start date is today (system date) unless overridden.
