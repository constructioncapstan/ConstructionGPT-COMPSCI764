"""Training-fold cost-per-area baseline."""

from __future__ import annotations

import numpy as np


def predict_cost_per_area(train_area, train_cost, test_area) -> float:
    """Predict a held-out target using the training-fold aggregate $/m² rate."""
    train_area = np.asarray(train_area, dtype=float)
    train_cost = np.asarray(train_cost, dtype=float)

    if len(train_area) == 0:
        raise ValueError("At least one labelled training project is required.")
    if np.any(train_area <= 0):
        raise ValueError("Floor areas must be positive.")

    # Aggregate rate is less unstable than averaging project-level ratios at tiny n.
    rate = float(train_cost.sum() / train_area.sum())
    return rate * float(test_area)
