"""Run the final leakage-controlled ConstructionGPT experiment.

Private inputs are intentionally not committed. Expected inputs:
- an 11-project CSV with anonymised Project_ID, structured features and trade targets;
- a NumPy array containing one 32x32 grayscale retrieval view per selected page;
- JSON page metadata with project_id and image name;
- data/plan_manifest_public.csv.

Example:
    python scripts/run_full_experiment.py \
      --dataset data/private/research_dataset.csv \
      --images data/private/plan32.npy \
      --image-meta data/private/plan32_meta.json \
      --output results/local_run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.baselines.structured_models import make_mlp, make_xgboost
from src.preprocessing.structured_features import ALL_FEATURES
from src.vlm.contrastive_encoder import CVLConfig, fit_contrastive_encoder

SEED = 764
TARGETS = {
    "Foundation": "Foundation_Target_exGST",
    "Framing": "Framing_Package_Target_exGST",
    "Roofing": "Roofing_Target_exGST",
}

# Historical-price normalisation used in the reported experiment.
# 2022-2025 values are Stats NZ residential-building input PPI reference values.
# The 2026-Q2 bridge uses the reported 3.5% annual construction-cost growth.
INDEX_BY_YEAR = {2022: 1516.0, 2023: 1587.0, 2024: 1625.0, 2025: 1641.0}
TARGET_INDEX_2026_Q2 = 1641.0 * 1.035
COST_BASIS_YEAR = {
    "P01": 2023, "P02": 2024, "P03": 2023, "P04": 2024,
    "P05": 2022, "P06": 2022, "P07": 2022, "P08": 2022,
    "P09": 2025, "P10": 2024, "P11": 2025,
}


def metric_dict(actual, predicted):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "n": len(y),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MAPE": float(np.mean(np.abs((y - p) / y)) * 100.0),
    }


def page_description(row: pd.Series, page_type: str) -> str:
    return (
        f"{page_type.replace('_', ' ')} residential building consent drawing. "
        f"floor area {float(row.Floor_Area_m2):.0f} square metres. "
        f"{int(row.Units)} dwelling units. {int(row.Storeys)} storeys. "
        f"{int(row.Bedrooms_Total)} bedrooms. foundation {row.Foundation_Type}. "
        f"roof {row.Roof_Form}. cladding {row.Primary_Cladding}. "
        f"{row.Site_Complexity} site complexity."
    )


def fit_fold_embeddings(df, projects, heldout, images, image_meta, page_type_by_name):
    train_indices = [i for i, item in enumerate(image_meta) if item["project_id"] != heldout]
    text = [
        page_description(
            df.loc[image_meta[i]["project_id"]],
            page_type_by_name[image_meta[i]["name"]],
        )
        for i in train_indices
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=256)
    text_features = vectorizer.fit_transform(text).toarray().astype("float32")

    config = CVLConfig(seed=SEED)
    model = fit_contrastive_encoder(images[train_indices], text_features, config)
    with torch.no_grad():
        all_page_embeddings = model.encode_image(
            torch.as_tensor(images, dtype=torch.float32)
        ).cpu().numpy()

    project_embeddings = {}
    for project_id in projects:
        idx = [i for i, item in enumerate(image_meta) if item["project_id"] == project_id]
        vector = all_page_embeddings[idx].mean(axis=0)
        project_embeddings[project_id] = vector / max(np.linalg.norm(vector), 1e-12)
    return project_embeddings


def cost_per_area(df, train_projects, test_project, target):
    rates = [
        float(df.loc[p, target]) / float(df.loc[p, "Floor_Area_m2"])
        for p in train_projects
        if pd.notna(df.loc[p, target])
    ]
    return float(np.mean(rates)) * float(df.loc[test_project, "Floor_Area_m2"])


def structured_predict(df, train_projects, test_project, target, method):
    labelled = [p for p in train_projects if pd.notna(df.loc[p, target])]
    x_train = df.loc[labelled, ALL_FEATURES]
    y_train = df.loc[labelled, target].astype(float).values
    x_test = df.loc[[test_project], ALL_FEATURES]
    model = make_xgboost(SEED) if method == "XGBoost" else make_mlp(SEED)
    model.fit(x_train, y_train)
    return max(float(model.predict(x_test)[0]), 0.0)


def visual_predict(df, train_projects, test_project, target, embeddings):
    labelled = [p for p in train_projects if pd.notna(df.loc[p, target])]
    e = np.stack([embeddings[p] for p in labelled])
    y = df.loc[labelled, target].astype(float).values
    areas = np.array([df.loc[p, "Floor_Area_m2"] for p in labelled], dtype=float)

    components = max(1, min(3, len(labelled) - 1, e.shape[1]))
    pca = PCA(n_components=components, random_state=SEED)
    z_train = pca.fit_transform(e)
    z_test = pca.transform(embeddings[test_project][None, :])

    f_train = np.column_stack([z_train, np.log(areas)])
    f_test = np.column_stack([
        z_test,
        [math.log(float(df.loc[test_project, "Floor_Area_m2"]))],
    ])
    scaler = StandardScaler()
    f_train = scaler.fit_transform(f_train)
    f_test = scaler.transform(f_test)
    model = Ridge(alpha=5.0).fit(f_train, y)
    return max(float(model.predict(f_test)[0]), 0.0)


def retrieval_predict(df, train_projects, test_project, target, embeddings, k=3, temperature=0.10):
    candidates = [p for p in train_projects if pd.notna(df.loc[p, target])]
    similarity = np.array([
        float(np.dot(embeddings[test_project], embeddings[p])) for p in candidates
    ])
    order = np.argsort(-similarity)[: min(k, len(candidates))]
    selected = [candidates[i] for i in order]
    selected_similarity = similarity[order]
    weights = np.exp((selected_similarity - selected_similarity.max()) / temperature)
    weights /= weights.sum()
    rates = np.array([
        float(df.loc[p, target]) / float(df.loc[p, "Floor_Area_m2"])
        for p in selected
    ])
    prediction = float(np.dot(weights, rates)) * float(df.loc[test_project, "Floor_Area_m2"])
    return max(prediction, 0.0), selected, selected_similarity.tolist(), weights.tolist()


def tune_blend(df, train_projects, target, embeddings):
    labelled = [p for p in train_projects if pd.notna(df.loc[p, target])]
    if len(labelled) < 4:
        return 0.5
    inner = []
    for validation_project in labelled:
        subtrain = [p for p in labelled if p != validation_project]
        visual = visual_predict(df, subtrain, validation_project, target, embeddings)
        retrieved, *_ = retrieval_predict(df, subtrain, validation_project, target, embeddings)
        inner.append((float(df.loc[validation_project, target]), visual, retrieved))
    alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    errors = [
        np.mean([abs(y - (a * v + (1 - a) * r)) for y, v, r in inner])
        for a in alpha_grid
    ]
    return alpha_grid[int(np.argmin(errors))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-meta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/plan_manifest_public.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/local_run"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    df = pd.read_csv(args.dataset).set_index("Project_ID")
    projects = df.index.tolist()
    for project_id in projects:
        basis = COST_BASIS_YEAR[project_id]
        factor = TARGET_INDEX_2026_Q2 / INDEX_BY_YEAR[basis]
        for trade, raw_target in TARGETS.items():
            df.loc[project_id, f"{trade}_Target_2026Q2"] = df.loc[project_id, raw_target] * factor

    # Images are stored as white-background grayscale [0, 1]; invert so linework has larger values.
    images = 1.0 - np.load(args.images).reshape(-1, 1024).astype("float32")
    image_meta = json.loads(args.image_meta.read_text())
    manifest = pd.read_csv(args.manifest)
    page_type_by_name = {
        f"{r.project_id}_{r.page_type}_{r.source_role}_p{int(r.page_number):02d}_retrieval.jpg": r.page_type
        for _, r in manifest.iterrows()
    }

    predictions = []
    neighbours = []
    for heldout in projects:
        train_projects = [p for p in projects if p != heldout]
        embeddings = fit_fold_embeddings(
            df, projects, heldout, images, image_meta, page_type_by_name
        )
        for trade in TARGETS:
            target = f"{trade}_Target_2026Q2"
            retrieved, selected, sims, weights = retrieval_predict(
                df, train_projects, heldout, target, embeddings
            )
            for rank, (project_id, similarity, weight) in enumerate(
                zip(selected, sims, weights), 1
            ):
                neighbours.append({
                    "heldout": heldout,
                    "trade": trade,
                    "rank": rank,
                    "neighbor": project_id,
                    "cosine_similarity": similarity,
                    "weight": weight,
                })

            actual = df.loc[heldout, target]
            if pd.isna(actual):
                continue
            visual = visual_predict(df, train_projects, heldout, target, embeddings)
            alpha = tune_blend(df, train_projects, target, embeddings)
            predictions.append({
                "Project_ID": heldout,
                "Trade": trade,
                "Actual_2026Q2": float(actual),
                "Cost_per_m2": cost_per_area(df, train_projects, heldout, target),
                "XGBoost": structured_predict(df, train_projects, heldout, target, "XGBoost"),
                "MLP": structured_predict(df, train_projects, heldout, target, "MLP"),
                "CVL_Visual": visual,
                "Retrieval": retrieved,
                "CVL_plus_Retrieval": alpha * visual + (1 - alpha) * retrieved,
                "Blend_alpha_visual": alpha,
            })

    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(args.output / "lopo_predictions.csv", index=False)
    pd.DataFrame(neighbours).to_csv(args.output / "retrieval_neighbors.csv", index=False)

    methods = [
        "Cost_per_m2", "XGBoost", "MLP", "CVL_Visual", "Retrieval", "CVL_plus_Retrieval"
    ]
    metrics = []
    for trade in TARGETS:
        rows = prediction_df[prediction_df.Trade == trade]
        for method in methods:
            metrics.append({
                "Trade": trade,
                "Method": method,
                **metric_dict(rows.Actual_2026Q2, rows[method]),
            })
    for method in methods:
        metrics.append({
            "Trade": "Overall",
            "Method": method,
            **metric_dict(prediction_df.Actual_2026Q2, prediction_df[method]),
        })
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
