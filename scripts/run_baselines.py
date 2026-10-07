"""Run project-level baselines on the private 11-project dataset.

Example:
    python scripts/run_baselines.py --data data/private/research_dataset.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.cost_per_area import predict_cost_per_area
from src.baselines.structured_models import make_mlp, make_xgboost
from src.evaluation.metrics import regression_report
from src.preprocessing.structured_features import ALL_FEATURES, TARGETS


def loocv_predictions(df: pd.DataFrame, target_col: str, model_name: str):
    labelled = df[df[target_col].notna()].copy().reset_index(drop=True)
    predictions = []

    for test_idx in range(len(labelled)):
        train = labelled.drop(index=test_idx)
        test = labelled.iloc[[test_idx]]

        y_train = train[target_col].astype(float).to_numpy()

        if model_name == "cost_per_area":
            pred = predict_cost_per_area(
                train["Floor_Area_m2"].to_numpy(),
                y_train,
                float(test.iloc[0]["Floor_Area_m2"]),
            )
        else:
            model = make_xgboost() if model_name == "xgboost" else make_mlp()
            model.fit(train[ALL_FEATURES], y_train)
            pred = float(model.predict(test[ALL_FEATURES])[0])

        predictions.append(
            {
                "Project_ID": test.iloc[0]["Project_ID"],
                "actual": float(test.iloc[0][target_col]),
                "predicted": pred,
                "absolute_error": abs(float(test.iloc[0][target_col]) - pred),
            }
        )

    y_true = [x["actual"] for x in predictions]
    y_pred = [x["predicted"] for x in predictions]
    return predictions, regression_report(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/baseline_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    results = {}
    for trade, target_col in TARGETS.items():
        results[trade] = {}
        for model_name in ("cost_per_area", "xgboost", "mlp"):
            preds, metrics = loocv_predictions(df, target_col, model_name)
            results[trade][model_name] = {
                "n": len(preds),
                "metrics": metrics,
                "predictions": preds,
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
