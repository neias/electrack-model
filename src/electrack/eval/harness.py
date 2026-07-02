"""Uçtan-uca değerlendirme koşucusu — model → tespit → metrik → kabul raporu.

Bir YOLO-biçimli değerlendirme kümesi (images/ + labels/) üzerinde modeli çalıştırır,
tahminleri ground-truth ile eşleştirir (electrack.eval.metrics) ve
`eval-report.schema.json`'a uyan bir kabul raporu üretir (electrack.eval.acceptance).

Saf yardımcılar (`load_yolo_gt`, `iter_images`) ağır bağımlılık gerektirmez ve
test edilebilir; asıl çıkarım `Detector` üzerinden LAZY yüklenir.
"""

from __future__ import annotations

from pathlib import Path

from electrack.config import ThresholdConfig
from electrack.data.classes import ClassRegistry

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def load_yolo_gt(label_path: Path) -> list[dict]:
    """YOLO etiket dosyasını ({class} cx cy w h) metrik GT biçimine çevir.

    Dönüş: [{'class_id': int, 'bbox': [x1,y1,x2,y2]}] (normalize xyxy).
    Dosya yoksa/boşsa (negatif görüntü) boş liste döner.
    """
    label_path = Path(label_path)
    gts: list[dict] = []
    if not label_path.is_file():
        return gts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        x1, y1 = _clamp01(cx - w / 2.0), _clamp01(cy - h / 2.0)
        x2, y2 = _clamp01(cx + w / 2.0), _clamp01(cy + h / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        gts.append({"class_id": cid, "bbox": [x1, y1, x2, y2]})
    return gts


def iter_images(images_dir: Path) -> list[Path]:
    """Bir dizindeki görüntü dosyalarını (deterministik sırada) döndür."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Görüntü dizini yok: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)


def resolve_split(dataset: Path, split: str | None) -> tuple[Path, Path]:
    """(images_dir, labels_dir) çöz. `split` verilmişse images/<split> + labels/<split>."""
    dataset = Path(dataset)
    if split:
        return dataset / "images" / split, dataset / "labels" / split
    return dataset / "images", dataset / "labels"


def run_evaluation(
    model_path: Path,
    dataset: Path,
    registry: ClassRegistry,
    thresholds: ThresholdConfig,
    split: str | None = None,
    measure_latency: bool = False,
    imgsz: int | None = None,
) -> dict:
    """Modeli değerlendir ve kabul raporu (dict) üret."""
    # Ağır/iç bağımlılıklar burada (lazy).
    from electrack.data.manifest import dataset_id as make_dataset_id
    from electrack.data.manifest import model_id as make_model_id
    from electrack.eval.acceptance import build_report
    from electrack.eval.metrics import ImageEval, evaluate
    from electrack.inference.detector import Detector

    images_dir, labels_dir = resolve_split(dataset, split)
    images = iter_images(images_dir)
    if not images:
        raise ValueError(f"Değerlendirme için görüntü bulunamadı: {images_dir}")

    detector = Detector(model_path, registry=registry, thresholds=thresholds, imgsz=imgsz)
    evals: list[ImageEval] = []
    for img in images:
        out = detector.predict_path(img)
        gts = load_yolo_gt(labels_dir / f"{img.stem}.txt")
        evals.append(ImageEval(gts=gts, preds=out["detections"]))

    agg = evaluate(evals)

    latency = None
    if measure_latency:
        from electrack.eval.latency import measure_latency as _measure

        latency = _measure(detector, images)

    return build_report(
        model_id=make_model_id(Path(model_path)),
        dataset_id=make_dataset_id(images_dir),
        agg=agg,
        id_to_name=registry.name,
        thresholds=thresholds,
        latency=latency,
    )
