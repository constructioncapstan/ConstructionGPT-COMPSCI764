"""Compact structured baselines suitable for the very small dataset."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.preprocessing.structured_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_xgboost(random_state: int = 42) -> Pipeline:
    model = XGBRegressor(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
    )
    return Pipeline([("prep", _preprocessor(False)), ("model", model)])


def make_mlp(random_state: int = 42) -> Pipeline:
    model = MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="relu",
        alpha=1.0,
        learning_rate_init=0.005,
        max_iter=3000,
        early_stopping=False,
        random_state=random_state,
    )
    return Pipeline([("prep", _preprocessor(True)), ("model", model)])
