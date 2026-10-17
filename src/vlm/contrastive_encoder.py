"""Lightweight fold-specific contrastive image-text encoder used in the final experiment.

The model is intentionally compact because the research dataset contains only 11
independent projects. It is retrained inside every outer LOPO fold so the held-out
project cannot influence representation weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CVLConfig:
    image_dim: int = 1024
    image_hidden: int = 128
    text_hidden: int = 64
    embedding_dim: int = 32
    dropout: float = 0.10
    temperature: float = 0.07
    learning_rate: float = 0.002
    weight_decay: float = 0.001
    epochs: int = 200
    seed: int = 764


class ContrastivePlanEncoder(nn.Module):
    """CLIP-inspired encoder for downsampled plan pages and metadata text."""

    def __init__(self, text_dim: int, config: CVLConfig = CVLConfig()) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.image_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.image_hidden, config.embedding_dim),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, config.text_hidden),
            nn.ReLU(),
            nn.Linear(config.text_hidden, config.embedding_dim),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(text), dim=1)

    def forward(self, images: torch.Tensor, text: torch.Tensor):
        return self.encode_image(images), self.encode_text(text)


def fit_contrastive_encoder(
    images: np.ndarray,
    text_features: np.ndarray,
    config: CVLConfig = CVLConfig(),
) -> ContrastivePlanEncoder:
    """Fit one encoder using training-fold pages only."""
    torch.manual_seed(config.seed)
    model = ContrastivePlanEncoder(text_features.shape[1], config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    x = torch.as_tensor(images, dtype=torch.float32)
    t = torch.as_tensor(text_features, dtype=torch.float32)
    labels = torch.arange(len(x))

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        zi, zt = model(x, t)
        logits = (zi @ zt.T) / config.temperature
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )
        loss.backward()
        optimizer.step()

    return model.eval()


def aggregate_project_embeddings(
    page_embeddings: np.ndarray,
    project_ids: list[str],
) -> dict[str, np.ndarray]:
    """Average page embeddings per project and L2-normalise the result."""
    result: dict[str, np.ndarray] = {}
    for project_id in sorted(set(project_ids)):
        idx = [i for i, p in enumerate(project_ids) if p == project_id]
        vector = page_embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(vector)
        result[project_id] = vector / max(norm, 1e-12)
    return result
