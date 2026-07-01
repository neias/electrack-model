"""T015 [US1] Kontrat testi: çıktı detection-output.schema.json'a uyar."""

import json
from pathlib import Path

import pytest

from electrack.inference.validate import ContractError, validate_detection_output

SCHEMA = Path("specs/001-pcb-component-detection/contracts/detection-output.schema.json")


def test_schema_file_parses():
    doc = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert doc["title"] == "PCB Detection Output"


def test_schema_examples_are_valid():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    for ex in schema["examples"]:
        validate_detection_output(ex)  # hata fırlatmamalı


def test_valid_document_passes():
    doc = {
        "image_width": 1920,
        "image_height": 1080,
        "detections": [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "class_name": "ic", "class_id": 0, "confidence": 0.9},
            {
                "bbox": [0.5, 0.5, 0.6, 0.7],
                "class_name": "unknown",
                "class_id": None,
                "confidence": 0.4,
            },
        ],
    }
    validate_detection_output(doc)


def test_empty_detections_valid():
    validate_detection_output({"detections": []})


def test_unknown_must_have_null_id():
    doc = {
        "detections": [
            {
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "class_name": "unknown",
                "class_id": 2,
                "confidence": 0.4,
            }
        ]
    }
    with pytest.raises(ContractError):
        validate_detection_output(doc)


def test_bbox_out_of_range_rejected():
    doc = {
        "detections": [
            {"bbox": [0.1, 0.2, 1.3, 0.4], "class_name": "ic", "class_id": 0, "confidence": 0.9}
        ]
    }
    with pytest.raises(ContractError):
        validate_detection_output(doc)


def test_bbox_min_max_order_enforced():
    doc = {
        "detections": [
            {"bbox": [0.5, 0.2, 0.3, 0.4], "class_name": "ic", "class_id": 0, "confidence": 0.9}
        ]
    }
    with pytest.raises(ContractError):
        validate_detection_output(doc)


def test_invalid_class_name_rejected():
    doc = {
        "detections": [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "class_name": "diode", "class_id": 9, "confidence": 0.9}
        ]
    }
    with pytest.raises(ContractError):
        validate_detection_output(doc)
