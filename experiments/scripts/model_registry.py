#!/usr/bin/env python3
"""Simple registry for the sparse-view 3DGS experiment scaffold."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # feedforward or optimization
    requires_pose: bool
    supports_views: List[int]
    notes: str = ""


MODEL_REGISTRY = {
    "DepthSplat": ModelSpec(
        name="DepthSplat",
        family="feedforward",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Zero-shot feed-forward model.",
    ),
    "MVSplat": ModelSpec(
        name="MVSplat",
        family="feedforward",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Zero-shot feed-forward model.",
    ),
    "Vanilla3DGS": ModelSpec(
        name="Vanilla3DGS",
        family="optimization",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Standard 3DGS optimization baseline.",
    ),
    "SparseGS": ModelSpec(
        name="SparseGS",
        family="optimization",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Sparse-view specialized optimization baseline.",
    ),
}


DATASET_REGISTRY = {
    "RE10K": {
        "path": "/data/re10k",
        "description": "Main benchmark candidate for large-scale sparse-view experiments.",
        "recommended_scenes": 20,
        "notes": "Good for primary regime-map experiments and diverse indoor/outdoor scenes.",
    },
    "DL3DV": {
        "path": "/data/dl3dv",
        "description": "High-quality multi-view dataset with richer geometry variation.",
        "recommended_scenes": 20,
        "notes": "Useful as a secondary benchmark if available.",
    },
    "DTU": {
        "path": "/data/dtu",
        "description": "External validation set with GT geometry available.",
        "recommended_scenes": 8,
        "notes": "Best choice for depth/geometry and failure analysis.",
    },
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]


def get_dataset_spec(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name]
