"""T033 [US3] Yeni sınıf ekleme: sınıf kümesi tamamen data.yaml'dan türetilir ve
mevcut sınıf metrikleri korunur (parametrik)."""

from pathlib import Path

from electrack.config import ThresholdConfig
from electrack.data.classes import ClassRegistry
from electrack.eval.acceptance import build_report
from electrack.eval.metrics import ImageEval, evaluate

BASE_YAML = """\
path: .
train: images/train
val: images/val
test: images/test
names:
  0: ic
  1: electrolytic_capacitor
  2: connector_header
  3: led
  4: through_hole_resistor
  5: to92_transistor
"""

EXTENDED_YAML = BASE_YAML + "  6: smd_resistor\n"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "data.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_registry_parametric_from_yaml(tmp_path):
    base = ClassRegistry.from_data_yaml(_write(tmp_path, BASE_YAML))
    assert base.num_classes == 6

    ext = ClassRegistry.from_data_yaml(_write(tmp_path, EXTENDED_YAML))
    assert ext.num_classes == 7
    assert ext.name(6) == "smd_resistor"
    assert "smd_resistor" in ext.output_labels()
    assert "unknown" in ext.output_labels()
    # Mevcut id'ler değişmemeli (geriye uyumluluk / FR-009).
    for i in range(6):
        assert base.name(i) == ext.name(i)


def test_existing_class_metrics_unaffected_by_new_class(tmp_path):
    ext = ClassRegistry.from_data_yaml(_write(tmp_path, EXTENDED_YAML))
    # Mevcut sınıf (ic) doğru tespit; yeni sınıf varlığı recall'ü etkilememeli.
    gts = [{"class_id": 0, "bbox": [0.1, 0.1, 0.3, 0.3]}]
    preds = [{"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "ic", "class_id": 0, "confidence": 0.95}]
    agg = evaluate([ImageEval(gts=gts, preds=preds)])
    report = build_report("m", "ds", agg, ext.name, ThresholdConfig())
    assert report["overall_recall"] == 1.0
    assert report["per_class"]["ic"]["recall"] == 1.0
