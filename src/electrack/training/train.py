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


def _select_device() -> str:
    """Apple Silicon'da MPS'i tercih et; yoksa CUDA; yoksa CPU (FR/SC-003 hedefi M4)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def train(
    config_path: Path,
    epochs: int | None = None,
    device: str | None = None,
    fraction: float = 1.0,
    model: str | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    resume: bool = False,
) -> Path:
    """Modeli eğit ve en iyi ağırlık yolunu (.pt) döndür.

    `model` verilirse ondan (ör. sıcak başlangıç: last.pt) başlar; `batch`/`imgsz`
    config'i geçersiz kılar (bellek/çözünürlük ayarı için).
    `resume=True` ise `model` (last.pt) yarıda kalan koşuyu AYNI LR programıyla
    sürdürür (MPS aralıklı çökmesine karşı — warm-start'ın aksine ağırlıkları bozmaz).
    """
    cfg = TrainConfig.from_yaml(config_path)
    if epochs is not None:
        cfg.epochs = epochs
    if model is not None:
        cfg.model = model
    if batch is not None:
        cfg.batch = batch
    if imgsz is not None:
        cfg.imgsz = imgsz
    set_determinism(cfg.seed)
    dev = device or _select_device()
    log.info(
        "Cihaz: %s | model: %s | batch: %d | imgsz: %d | fraction: %.2f",
        dev,
        cfg.model,
        cfg.batch,
        cfg.imgsz,
        fraction,
    )

    # Sınıf kümesi doğrulaması — data.yaml tek doğru kaynak.
    registry = ClassRegistry.from_data_yaml(Path(cfg.data_yaml))
    log.info("Sınıf sayısı=%d: %s", registry.num_classes, registry.names())

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics kurulu değil. Eğitim için: pip install '.[train]'") from e

    model = YOLO(cfg.model)
    if resume:
        # Yarıda kalan koşuyu kaldığı yerden sürdür (kaydedilmiş args/LR/optimizer state).
        log.info("RESUME: %s kontrol noktasından devam.", cfg.model)
        results = model.train(resume=True)
    else:
        results = model.train(
            data=cfg.data_yaml,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            seed=cfg.seed,
            project=cfg.project,
            name=cfg.name,
            device=dev,
            fraction=fraction,
            **cfg.augment,
        )
    best = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Eğitim tamamlandı. En iyi ağırlık: %s", best)
    return best
