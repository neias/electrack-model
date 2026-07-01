"""Konfig yönetimi + determinizm/seed yardımcıları (FR-013, SC-006)."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from electrack.config_constants import (
    DEFAULT_CLASS_THRESHOLD,
    DEFAULT_DET_THRESHOLD,
    DEFAULT_SEED,
    MODEL_INPUT_SIZE,
)


@dataclass
class TrainConfig:
    """Eğitim konfigürasyonu (mvp.yaml ile eşlenir)."""

    data_yaml: str = "datasets/yolo/data.yaml"
    model: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = MODEL_INPUT_SIZE
    batch: int = 16
    seed: int = DEFAULT_SEED
    project: str = "models/weights"
    name: str = "mvp"
    augment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        known = {k: doc[k] for k in doc if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ThresholdConfig:
    det_threshold: float = DEFAULT_DET_THRESHOLD
    class_threshold: float = DEFAULT_CLASS_THRESHOLD


def set_determinism(seed: int = DEFAULT_SEED) -> None:
    """Tüm rastgelelik kaynaklarını sabitle (tekrar üretilebilirlik).

    Ağır kütüphaneler (torch/numpy) varsa onları da tohumlar; yoksa sessizce geçer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p: str | None, default: str) -> Path:
    return Path(p) if p else Path(default)
