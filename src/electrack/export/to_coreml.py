"""Core ML dışa aktarımı — gömülü NMS + eşik (FR-002a, FR-006, research.md R4).

`.pt` ağırlıklarını Neural Engine için `.mlpackage`'a aktarır. `nms=True` ile NMS ve
`conf=det_threshold` ile genel tespit eşiği modele gömülür; böylece tüketen uygulama
ham aday filtrelemesiyle uğraşmaz (sınıf-kesinlik `unknown` mantığı çıkarım
sarmalayıcısında, config'ten okunan class_threshold ile uygulanır).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from electrack.config_constants import DEFAULT_DET_THRESHOLD, MODEL_INPUT_SIZE
from electrack.logging_setup import get_logger

log = get_logger("electrack.export")


def export_coreml(
    weights: Path,
    det_threshold: float = DEFAULT_DET_THRESHOLD,
    imgsz: int = MODEL_INPUT_SIZE,
    out_dir: Path = Path("models/export"),
) -> Path:
    """`.pt` → `.mlpackage`. Dışa aktarılan paketin yolunu döndürür."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics/coremltools kurulu değil. Dışa aktarım için: "
            "pip install '.[train,export]'"
        ) from e

    model = YOLO(str(weights))
    log.info("Core ML'e aktarılıyor: nms=True, conf=%.2f, imgsz=%d", det_threshold, imgsz)
    exported = model.export(
        format="coreml",
        nms=True,
        conf=det_threshold,
        imgsz=imgsz,
    )
    exported_path = Path(exported)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "electrack.mlpackage"
    # Ultralytics çıktısını kanonik hedefe TAŞI (README/evaluate bu yolu bekler).
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(exported_path), str(target))
    log.info("Dışa aktarıldı: %s → %s", exported_path, target)
    return target
