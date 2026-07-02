"""Çıkarım sarmalayıcısı (US1: FR-001, FR-001a, FR-002, FR-002a, FR-004, FR-007).

- Girdi: herhangi çözünürlükte RGB kare; ön-işleme (letterbox/normalize) burada.
- Çıktı: `detection-output` sözleşmesine uygun tespit listesi
  (normalize [0-1] xyxy köşe kutu + class_name + class_id + confidence).
- İki-eşik mantığı: det_threshold altı → tespit yok (FP bastırma);
  class_threshold altı → "unknown" (research.md R5).

Ağır bağımlılık (ultralytics/coreml) LAZY yüklenir. Aşağıdaki `postprocess`,
`derive_label`, `normalize_xyxy`, `letterbox_params` fonksiyonları saf/pür Python'dur
ve model olmadan test edilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from electrack.config import ThresholdConfig
from electrack.config_constants import MODEL_INPUT_SIZE, UNKNOWN_LABEL
from electrack.data.classes import ClassRegistry

# --------------------------- Saf geometri --------------------------- #


@dataclass
class LetterboxParams:
    scale: float
    pad_x: float
    pad_y: float
    new_w: float
    new_h: float


def letterbox_params(orig_w: int, orig_h: int, size: int = MODEL_INPUT_SIZE) -> LetterboxParams:
    """En-boy koruyan letterbox parametreleri (kare `size`e sığdırma)."""
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Görüntü boyutları pozitif olmalı.")
    scale = min(size / orig_w, size / orig_h)
    new_w = orig_w * scale
    new_h = orig_h * scale
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    return LetterboxParams(scale, pad_x, pad_y, new_w, new_h)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def unletterbox_norm_point(px: float, py: float, p: LetterboxParams) -> tuple[float, float]:
    """Letterbox'lanmış piksel noktasını orijinale göre normalize [0-1]'e çevir."""
    nx = _clamp01((px - p.pad_x) / p.new_w)
    ny = _clamp01((py - p.pad_y) / p.new_h)
    return nx, ny


def normalize_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[float]:
    """Orijinal piksel köşe kutusunu normalize [0-1] xyxy'ye çevir + sırala + kırp."""
    nx1, nx2 = sorted((_clamp01(x1 / w), _clamp01(x2 / w)))
    ny1, ny2 = sorted((_clamp01(y1 / h), _clamp01(y2 / h)))
    return [nx1, ny1, nx2, ny2]


# --------------------------- İki-eşik etiketleme --------------------------- #


def derive_label(
    class_id: int,
    score: float,
    registry: ClassRegistry,
    thresholds: ThresholdConfig,
) -> tuple[str, int | None] | None:
    """(class_name, class_id) döndür; det eşiği altındaysa None (tespit yok)."""
    if score < thresholds.det_threshold:
        return None  # FP bastırma (FR-007).
    if score < thresholds.class_threshold:
        return (UNKNOWN_LABEL, None)  # nesne var, tür belirsiz (FR-004).
    return (registry.name(class_id), class_id)


def build_detection(
    bbox_norm: list[float], class_name: str, class_id: int | None, score: float
) -> dict:
    return {
        "bbox": [round(float(v), 6) for v in bbox_norm],
        "class_name": class_name,
        "class_id": class_id,
        "confidence": round(float(score), 6),
    }


def postprocess(
    raw: list[tuple[tuple[float, float, float, float], int, float]],
    orig_w: int,
    orig_h: int,
    registry: ClassRegistry,
    thresholds: ThresholdConfig,
) -> dict:
    """Ham model çıktısını (orijinal-piksel xyxy, class_id, score) sözleşme çıktısına çevir.

    `raw` elemanları modelin (NMS uygulanmış) çıktısıdır; koordinatlar orijinal
    görüntü piksel uzayındadır. Sonuç `detection-output.schema.json` yapısındadır.
    """
    detections: list[dict] = []
    for (x1, y1, x2, y2), cid, score in raw:
        label = derive_label(cid, score, registry, thresholds)
        if label is None:
            continue
        class_name, class_id = label
        bbox = normalize_xyxy(x1, y1, x2, y2, orig_w, orig_h)
        detections.append(build_detection(bbox, class_name, class_id, score))
    return {"image_width": orig_w, "image_height": orig_h, "detections": detections}


# --------------------------- Model sarmalayıcısı (lazy) --------------------------- #


class Detector:
    """Core ML `.mlpackage` referans çıkarım sarmalayıcısı."""

    def __init__(
        self,
        model_path: Path,
        registry: ClassRegistry | None = None,
        thresholds: ThresholdConfig | None = None,
        imgsz: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.registry = registry or ClassRegistry.from_data_yaml()
        self.thresholds = thresholds or ThresholdConfig()
        self.imgsz = imgsz  # None → model varsayılanı; eğitim çözünürlüğüyle eşleştir
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError(
                    "ultralytics kurulu değil. Çıkarım için: pip install '.[train,export]'"
                ) from e
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict_path(self, image_path: Path) -> dict:
        """Bir görüntü dosyasında çıkarım yapıp sözleşme çıktısını döndür."""
        model = self._load()
        # det_threshold'u modele geçir: model-içi eşik ile sözleşme eşiği hizalansın
        # (aksi halde ultralytics varsayılanı ~0.25 aday öncesi filtreler — FR-007).
        predict_kwargs = {"conf": self.thresholds.det_threshold, "verbose": False}
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz  # eğitim çözünürlüğüyle eşleştir
        results = model.predict(str(image_path), **predict_kwargs)
        res = results[0]
        orig_h, orig_w = res.orig_shape  # (h, w)
        raw: list[tuple[tuple[float, float, float, float], int, float]] = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cid = int(box.cls[0].item())
            score = float(box.conf[0].item())
            raw.append(((x1, y1, x2, y2), cid, score))
        return postprocess(raw, orig_w, orig_h, self.registry, self.thresholds)
