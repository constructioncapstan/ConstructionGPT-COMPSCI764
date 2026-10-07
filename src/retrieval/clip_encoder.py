"""Frozen CLIP image encoder for project-level retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPProjectEncoder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[str | Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("At least one page image is required.")

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        features = self.model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.detach().cpu().numpy()

    def encode_project(self, image_paths: list[str | Path]) -> np.ndarray:
        """Mean-pool page embeddings into one project embedding."""
        page_embeddings = self.encode_images(image_paths)
        project = page_embeddings.mean(axis=0)
        norm = np.linalg.norm(project)
        if norm > 0:
            project = project / norm
        return project
