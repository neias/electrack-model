"""Çıktı sözleşmesi doğrulaması (T021).

`jsonschema` kuruluysa onu kullanır; değilse eşdeğer bir yapısal doğrulayıcıya
düşer — böylece sözleşme kontrolü ağır bağımlılık olmadan da çalışır.
"""

from __future__ import annotations

import json
from pathlib import Path

CONTRACT_PATH = Path("specs/001-pcb-component-detection/contracts/detection-output.schema.json")

VALID_LABELS = {
    "ic",
    "capacitor",
    "connector",
    "led",
    "resistor",
    "transistor",
    "unknown",
}


class ContractError(ValueError):
    pass


def validate_detection_output(doc: dict, extra_labels: list[str] = None) -> None:
    """Bir tespit-çıktısı dokümanını sözleşmeye göre doğrula. Hatalıysa ContractError."""
    allowed = set(VALID_LABELS)
    if extra_labels:
        allowed.update(extra_labels)

    # Önce jsonschema varsa onunla dene (kanonik doğrulama).
    try:
        import jsonschema

        schema = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(doc, schema)
        except jsonschema.ValidationError as e:
            raise ContractError(str(e)) from e
        # jsonschema enum'u sabit; genişletilmiş etiketler için yapısal kontrol devam eder.
    except ImportError:
        pass

    # Yapısal doğrulama (jsonschema yoksa da geçerli).
    if not isinstance(doc, dict) or "detections" not in doc:
        raise ContractError("'detections' alanı zorunlu.")
    if not isinstance(doc["detections"], list):
        raise ContractError("'detections' bir liste olmalı.")
    for i, d in enumerate(doc["detections"]):
        _validate_one(d, i, allowed)


def _validate_one(d: dict, i: int, allowed: set) -> None:
    for key in ("bbox", "class_name", "class_id", "confidence"):
        if key not in d:
            raise ContractError(f"detections[{i}]: '{key}' eksik.")
    bbox = d["bbox"]
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ContractError(f"detections[{i}].bbox 4 elemanlı olmalı.")
    for v in bbox:
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            raise ContractError(f"detections[{i}].bbox değeri [0,1] içinde olmalı: {v}")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ContractError(f"detections[{i}].bbox: x_min<x_max ve y_min<y_max olmalı.")
    if d["class_name"] not in allowed:
        raise ContractError(f"detections[{i}].class_name geçersiz: {d['class_name']}")
    cid, name = d["class_id"], d["class_name"]
    if name == "unknown":
        if cid is not None:
            raise ContractError(f"detections[{i}]: 'unknown' için class_id null olmalı.")
    else:
        if not isinstance(cid, int) or cid < 0:
            raise ContractError(f"detections[{i}].class_id negatif olmayan tamsayı olmalı.")
    conf = d["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise ContractError(f"detections[{i}].confidence [0,1] içinde olmalı.")


def validate_file(path: Path) -> None:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_detection_output(doc)
