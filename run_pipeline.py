from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ForecastConfig
from data_prep import cap_spikes, prepare_panel_data, validate_input
from features import build_feature_frame, load_holidays, select_feature_columns
from forecast import forecast_recursive
from modeling import (
    make_backtest_splits,
    make_holdout_split,
    predict_quantiles,
    train_qty_model,
    train_quantile_models,
)
from reporting import build_report


def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales forecasting pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV/Parquet")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--forecast-start", default=None, help="Forecast start date YYYY-MM-DD")
    parser.add_argument("--forecast-end", default="2026-12-31", help="Forecast end date YYYY-MM-DD")
    parser.add_argument("--enable-capping", action="store_true", help="Enable spike capping")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "preprocessing.log"

    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("forecasting")

    config = ForecastConfig(
        forecast_start_date=_parse_date(args.forecast_start),
        forecast_end_date=_parse_date(args.forecast_end) or date(2026, 12, 31),
        enable_capping=args.enable_capping,
    )

    df = _load_input(Path(args.input))
    validate_input(df, config.date_col, config.target_col, config.store_col)

    panel_df = prepare_panel_data(
        df,
        config.date_col,
        config.store_col,
        config.target_col,
        config.qty_col,
        config.forecast_end_date,
    )

    if config.enable_capping:
        panel_df, _ = cap_spikes(
            panel_df,
            config.store_col,
            config.target_col,
            config.qty_col,
            config.capping_sigma,
            logger,
        )
    else:
        panel_df["is_spike"] = 0

    start_date = panel_df[config.date_col].min().date()
    holiday_frame = load_holidays(start_date, config.forecast_end_date)

    panel_df[config.store_col] = panel_df[config.store_col].astype("category")

    features_df = build_feature_frame(
        panel_df,
        config.date_col,
        config.store_col,
        config.open_date_col,
        [config.target_col, config.qty_col],
        list(config.lags),
        list(config.rolling_windows),
        holiday_frame,
    )

    feature_cols = select_feature_columns(features_df, [config.target_col, config.qty_col])
    long_feature_cols = [c for c in feature_cols if not c.startswith("lag_") and not c.startswith("roll_")]

    train_df = features_df.dropna(subset=feature_cols + [config.target_col, config.qty_col])
    long_train_df = features_df.dropna(subset=long_feature_cols + [config.target_col, config.qty_col])
    X = train_df[feature_cols]
    y_sales = train_df[config.target_col]
    y_qty = train_df[config.qty_col]

    categorical_features = [config.store_col]

    quantile_models = train_quantile_models(
        X, y_sales, categorical_features, list(config.quantile_alphas), config.random_state
    )
    qty_model = train_qty_model(X, y_qty, categorical_features, config.random_state)

    long_quantile_models = train_quantile_models(
        long_train_df[long_feature_cols],
        long_train_df[config.target_col],
        categorical_features,
        list(config.quantile_alphas),
        config.random_state,
    )
    long_qty_model = train_qty_model(
        long_train_df[long_feature_cols],
        long_train_df[config.qty_col],
        categorical_features,
        config.random_state,
    )

    try:
        import shap

        median_model = quantile_models.get(0.5)
        if median_model is not None:
            sample = X.sample(min(2000, len(X)), random_state=config.random_state)
            explainer = shap.TreeExplainer(median_model)
            shap_values = explainer.shap_values(sample)
            shap_importance = (
                np.abs(shap_values).mean(axis=0)
                if isinstance(shap_values, np.ndarray)
                else np.abs(shap_values[0]).mean(axis=0)
            )
            shap_df = (
                pd.DataFrame({"feature": sample.columns, "mean_abs_shap": shap_importance})
                .sort_values("mean_abs_shap", ascending=False)
            )
            shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    except Exception:
        logger.info("SHAP computation skipped due to missing dependency or runtime error.")

    holdout_split = make_holdout_split(train_df, config.date_col, config.holdout_days)
    holdout_df = train_df.loc[holdout_split.test_idx].copy()

    quantile_preds = predict_quantiles(quantile_models, holdout_df[feature_cols])
    holdout_df["sales_amount_pred_lower"] = quantile_preds[min(config.quantile_alphas)].values
    holdout_df["sales_amount_pred"] = quantile_preds[0.5].values
    holdout_df["sales_amount_pred_upper"] = quantile_preds[max(config.quantile_alphas)].values

    notes = {
        "holiday_source": holiday_frame.source,
        "capping_enabled": str(config.enable_capping),
    }

    build_report(
        holdout_df,
        config.store_col,
        config.target_col,
        "sales_amount_pred",
        "sales_amount_pred_lower",
        "sales_amount_pred_upper",
        output_dir / "report.md",
        notes,
    )

    backtest_rows = []
    backtest_splits = make_backtest_splits(
        train_df, config.date_col, config.backtest_iters, config.backtest_test_days, config.backtest_gap_days
    )
    for i, split in enumerate(backtest_splits, start=1):
        split_df = train_df.loc[split.test_idx].copy()
        split_preds = predict_quantiles(quantile_models, split_df[feature_cols])
        split_df["pred"] = split_preds[0.5].values
        backtest_rows.append(
            {
                "split": i,
                "start_date": split_df[config.date_col].min().date(),
                "end_date": split_df[config.date_col].max().date(),
                "wmape": float(
                    np.sum(np.abs(split_df[config.target_col] - split_df["pred"]))
                    / np.sum(np.abs(split_df[config.target_col]))
                    * 100
                ),
            }
        )
    if backtest_rows:
        pd.DataFrame(backtest_rows).to_csv(output_dir / "backtest_metrics.csv", index=False)

    feature_config = {
        "features": feature_cols,
        "long_horizon_features": long_feature_cols,
        "categorical_features": categorical_features,
        "target": config.target_col,
        "secondary_target": config.qty_col,
        "recursive_horizon_days": config.recursive_horizon_days,
        "long_horizon_blend_weight": config.long_horizon_blend_weight,
    }
    (output_dir / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2), encoding="utf-8"
    )

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for alpha, model in quantile_models.items():
        joblib.dump(model, model_dir / f"lgbm_quantile_{alpha:.1f}.joblib")
    joblib.dump(qty_model, model_dir / "lgbm_qty.joblib")

    if config.forecast_start_date is None:
        config = ForecastConfig(
            forecast_start_date=date.today(),
            forecast_end_date=config.forecast_end_date,
            enable_capping=config.enable_capping,
        )

    forecast_result = forecast_recursive(
        history_df=panel_df,
        config=config,
        holiday_frame=holiday_frame,
        feature_cols=feature_cols,
        long_feature_cols=long_feature_cols,
        quantile_models=quantile_models,
        long_quantile_models=long_quantile_models,
        qty_model=qty_model,
        long_qty_model=long_qty_model,
        forecast_start=config.forecast_start_date,
        forecast_end=config.forecast_end_date,
    )

    forecast_df = forecast_result.forecast.copy()
    total_df = (
        forecast_df.groupby("posting_date")[["sales_amount_pred", "sales_amount_pred_lower", "sales_amount_pred_upper", "sales_qty_pred"]]
        .sum()
        .reset_index()
    )
    total_df[config.store_col] = "TOTAL"
    forecast_df = pd.concat([forecast_df, total_df], ignore_index=True)

    forecast_df.to_csv(output_dir / "forecast_2026.csv", index=False)

    monthly_df = forecast_df[forecast_df[config.store_col] != "TOTAL"].copy()
    monthly_df["month"] = pd.to_datetime(monthly_df["posting_date"]).dt.to_period("M").dt.to_timestamp()
    monthly_agg = (
        monthly_df.groupby("month")[["sales_amount_pred", "sales_amount_pred_lower", "sales_amount_pred_upper", "sales_qty_pred"]]
        .sum()
        .reset_index()
        .rename(columns={"month": "posting_month"})
    )
    monthly_agg.to_csv(output_dir / "forecast_2026_monthly.csv", index=False)


if __name__ == "__main__":
    main()
