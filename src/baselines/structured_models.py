"""Structured baselines used in the final 11-project LOPO experiment."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.preprocessing.structured_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_xgboost(random_state: int = 764) -> Pipeline:
    """Compact boosted-tree baseline fixed before final score interpretation."""
    model = XGBRegressor(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=10.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
    )
    return Pipeline([("prep", _preprocessor(False)), ("model", model)])


def make_mlp(random_state: int = 764) -> Pipeline:
    """Small regularised MLP used as the neural structured baseline."""
    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=2.0,
        max_iter=3000,
        random_state=random_state,
    )
    return Pipeline([("prep", _preprocessor(True)), ("model", model)])
