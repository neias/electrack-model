"""T016 [US1] Çıkarım-sonrası davranış: eşikler, unknown, FP bastırma, normalize.

Model gerektiren uçtan-uca çıkarım 'heavy' işaretli ve deps yoksa atlanır; çekirdek
postprocess mantığı saf/pür olarak test edilir (sentetik ham çıktı ile).
"""

from pathlib import Path

import pytest

from electrack.config import ThresholdConfig
from electrack.data.classes import ClassRegistry
from electrack.inference.detector import (
    letterbox_params,
    normalize_xyxy,
    postprocess,
    unletterbox_norm_point,
)
from electrack.inference.validate import validate_detection_output

REG = ClassRegistry.from_data_yaml(Path("datasets/yolo/data.yaml"))


def test_high_conf_gets_class():
    raw = [((10, 10, 110, 210), 0, 0.92)]
    out = postprocess(raw, 1000, 1000, REG, ThresholdConfig(0.25, 0.50))
    d = out["detections"][0]
    assert d["class_name"] == "ic" and d["class_id"] == 0
    validate_detection_output(out)


def test_mid_conf_becomes_unknown():
    # det (0.25) üstü ama class (0.50) altı → unknown.
    raw = [((10, 10, 110, 210), 3, 0.40)]
    out = postprocess(raw, 1000, 1000, REG, ThresholdConfig(0.25, 0.50))
    d = out["detections"][0]
    assert d["class_name"] == "unknown" and d["class_id"] is None
    validate_detection_output(out)


def test_low_conf_suppressed():
    # det eşiği altı → hiç tespit (FP bastırma, FR-007).
    raw = [((10, 10, 110, 210), 0, 0.10)]
    out = postprocess(raw, 1000, 1000, REG, ThresholdConfig(0.25, 0.50))
    assert out["detections"] == []


def test_normalization_and_clamp():
    box = normalize_xyxy(-5, 10, 1005, 500, 1000, 1000)
    assert box == [0.0, 0.01, 1.0, 0.5]


def test_normalize_orders_corners():
    box = normalize_xyxy(300, 400, 100, 200, 1000, 1000)
    assert box[0] < box[2] and box[1] < box[3]


def test_letterbox_roundtrip_center():
    p = letterbox_params(1000, 500, 640)
    # Orijinal merkez (500,250) letterbox merkezine (320,320) gitmeli; geri normalize ~0.5.
    nx, ny = unletterbox_norm_point(320.0, 320.0, p)
    assert abs(nx - 0.5) < 1e-6 and abs(ny - 0.5) < 1e-6


@pytest.mark.heavy
def test_end_to_end_inference_if_model_present():
    ultralytics = pytest.importorskip("ultralytics")  # noqa: F841
    model = Path("models/export/electrack.mlpackage")
    if not model.exists():
        pytest.skip("Dışa aktarılmış model yok.")
    from electrack.inference.detector import Detector

    det = Detector(model)
    for name in ("board_01.jpg", "empty_surface.jpg", "ambiguous.jpg"):
        img = Path("datasets/raw/samples") / name
        if img.exists():
            out = det.predict_path(img)
            validate_detection_output(out)
