"""T023 [US2] Kontrat testi: değerlendirme raporu eval-report.schema.json'a uyar."""

import json
from pathlib import Path

from electrack.config import ThresholdConfig
from electrack.data.classes import ClassRegistry
from electrack.eval.acceptance import build_report, compute_verdict
from electrack.eval.metrics import ImageEval, evaluate

SCHEMA = Path("specs/001-pcb-component-detection/contracts/eval-report.schema.json")
REG = ClassRegistry.from_data_yaml(Path("datasets/yolo/data.yaml"))


def _structural_validate(report: dict) -> None:
    required = {
        "model_id",
        "dataset_id",
        "iou_threshold",
        "overall_recall",
        "false_positive_rate",
        "per_class",
        "verdict",
    }
    missing = required - report.keys()
    assert not missing, f"eksik alanlar: {missing}"
    assert report["iou_threshold"] == 0.5
    assert 0.0 <= report["overall_recall"] <= 1.0
    assert 0.0 <= report["false_positive_rate"] <= 1.0
    assert report["verdict"] in ("PASS", "FAIL")
    for _name, m in report["per_class"].items():
        assert {"recall", "precision", "support"} <= m.keys()


def _sample_report() -> dict:
    gt = {"class_id": 0, "bbox": [0.1, 0.1, 0.3, 0.3]}
    pred = {"bbox": [0.11, 0.11, 0.31, 0.31], "class_name": "ic", "class_id": 0, "confidence": 0.9}
    agg = evaluate([ImageEval(gts=[gt], preds=[pred])])
    return build_report(
        model_id="m_test",
        dataset_id="ds_test",
        agg=agg,
        id_to_name=REG.name,
        thresholds=ThresholdConfig(),
        latency={
            "fps": 20.0,
            "ms_per_frame": 50.0,
            "objects_per_frame": 50,
            "hardware": "Apple M4 Mac Mini",
        },
        unknown={"high_conf_misclassifications": 0, "unknown_recall": 1.0},
    )


def test_report_structural():
    _structural_validate(_sample_report())


def test_report_jsonschema_if_available():
    try:
        import jsonschema
    except ImportError:
        return  # jsonschema yoksa yapısal test yeterli.
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(_sample_report(), schema)


def test_verdict_thresholds():
    assert compute_verdict(0.85, 0.05, 20.0) == "PASS"
    assert compute_verdict(0.79, 0.05, 20.0) == "FAIL"  # recall düşük
    assert compute_verdict(0.85, 0.11, 20.0) == "FAIL"  # FP yüksek
    assert compute_verdict(0.85, 0.05, 10.0) == "FAIL"  # FPS düşük
    assert compute_verdict(0.85, 0.05, None) == "PASS"  # FPS ölçülmemiş
