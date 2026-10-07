"""CLIP-style project retrieval with cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievedProject:
    project_id: str
    similarity: float


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return x / norm


def top_k_projects(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str],
    k: int = 3,
) -> list[RetrievedProject]:
    """Return top-k cosine-similar training projects.

    The caller is responsible for ensuring validation/test projects are absent
    from candidate_embeddings.
    """
    if len(candidate_ids) != len(candidate_embeddings):
        raise ValueError("candidate_ids and candidate_embeddings must align.")

    q = l2_normalize(np.asarray(query_embedding, dtype=float).reshape(1, -1))[0]
    c = l2_normalize(np.asarray(candidate_embeddings, dtype=float))
    similarities = c @ q

    k = min(k, len(candidate_ids))
    order = np.argsort(-similarities)[:k]
    return [
        RetrievedProject(candidate_ids[i], float(similarities[i]))
        for i in order
    ]
