from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_holdout_split(df: pd.DataFrame, date_col: str, holdout_days: int) -> Split:
    max_date = df[date_col].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    train_idx = df[df[date_col] <= cutoff].index.values
    test_idx = df[df[date_col] > cutoff].index.values
    return Split(train_idx=train_idx, test_idx=test_idx)


def make_backtest_splits(
    df: pd.DataFrame,
    date_col: str,
    iters: int,
    test_days: int,
    gap_days: int,
) -> List[Split]:
    max_date = df[date_col].max()
    splits: List[Split] = []
    for i in range(iters):
        test_end = max_date - pd.Timedelta(days=i * test_days)
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        train_end = test_start - pd.Timedelta(days=gap_days + 1)
        train_idx = df[df[date_col] <= train_end].index.values
        test_idx = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)].index.values
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append(Split(train_idx=train_idx, test_idx=test_idx))
    return list(reversed(splits))


def train_quantile_models(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    alphas: List[float],
    random_state: int,
) -> Dict[float, lgb.LGBMRegressor]:
    models: Dict[float, lgb.LGBMRegressor] = {}
    for alpha in alphas:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
        model.fit(X, y, categorical_feature=categorical_features)
        models[alpha] = model
    return models


def train_qty_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    random_state: int,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )
    model.fit(X, y, categorical_feature=categorical_features)
    return model


def predict_quantiles(models: Dict[float, lgb.LGBMRegressor], X: pd.DataFrame) -> pd.DataFrame:
    preds = {alpha: model.predict(X) for alpha, model in models.items()}
    return pd.DataFrame(preds)
