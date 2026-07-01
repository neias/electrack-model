"""YOLO eğitim giriş noktası — konfig-odaklı (FR-012, research.md R1).

Ultralytics `ultralytics` paketini LAZY import eder; sınıf listesi yalnızca
`data.yaml`'dan gelir (FR-009). Ağır bağımlılık yoksa açık, yönlendirici bir
hata verir.
"""

from __future__ import annotations

from pathlib import Path

from electrack.config import TrainConfig, set_determinism
from electrack.data.classes import ClassRegistry
from electrack.logging_setup import get_logger

log = get_logger("electrack.train")


def train(config_path: Path, epochs: int | None = None) -> Path:
    """Modeli eğit ve en iyi ağırlık yolunu (.pt) döndür."""
    cfg = TrainConfig.from_yaml(config_path)
    if epochs is not None:
        cfg.epochs = epochs
    set_determinism(cfg.seed)

    # Sınıf kümesi doğrulaması — data.yaml tek doğru kaynak.
    registry = ClassRegistry.from_data_yaml(Path(cfg.data_yaml))
    log.info("Sınıf sayısı=%d: %s", registry.num_classes, registry.names())

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics kurulu değil. Eğitim için: pip install '.[train]'") from e

    model = YOLO(cfg.model)
    results = model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        project=cfg.project,
        name=cfg.name,
        **cfg.augment,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Eğitim tamamlandı. En iyi ağırlık: %s", best)
    return best
