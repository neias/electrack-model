"""T024 [US2] Tekrar üretilebilirlik: split determinizmi + metrik tekrarlanabilirliği.

Uçtan-uca train/export 'heavy' işaretli ve deps yoksa atlanır; determinizm ve metrik
tekrarlanabilirliği saf mantıkla doğrulanır (SC-006).
"""

import pytest

from electrack.config import ThresholdConfig
from electrack.data.classes import ClassRegistry
from electrack.data.splits import assign_split, split_dataset
from electrack.eval.acceptance import build_report
from electrack.eval.metrics import ImageEval, evaluate


def test_split_is_deterministic():
    keys = [f"img_{i}.jpg" for i in range(500)]
    a = split_dataset(keys, seed=1337)
    b = split_dataset(keys, seed=1337)
    assert a == b


def test_split_stable_when_keys_added():
    base = [f"img_{i}.jpg" for i in range(100)]
    a = {k: assign_split(k, 1337) for k in base}
    # Küme büyüsün — mevcut anahtar atamaları değişmemeli.
    more = base + [f"img_{i}.jpg" for i in range(100, 200)]
    b = {k: assign_split(k, 1337) for k in more}
    for k in base:
        assert a[k] == b[k]


def test_split_fractions_reasonable():
    keys = [f"img_{i}.jpg" for i in range(2000)]
    s = split_dataset(keys, seed=1337, val_frac=0.15, test_frac=0.15)
    total = len(keys)
    assert 0.10 < len(s["val"]) / total < 0.20
    assert 0.10 < len(s["test"]) / total < 0.20
    assert 0.60 < len(s["train"]) / total < 0.80


def test_metrics_reproducible():
    reg = ClassRegistry.from_data_yaml()
    gts = [
        {"class_id": 0, "bbox": [0.1, 0.1, 0.3, 0.3]},
        {"class_id": 1, "bbox": [0.5, 0.5, 0.7, 0.7]},
    ]
    preds = [
        {"bbox": [0.11, 0.11, 0.31, 0.31], "class_name": "ic", "class_id": 0, "confidence": 0.9}
    ]
    imgs = [ImageEval(gts=gts, preds=preds)]
    r1 = build_report("m", "ds", evaluate(imgs), reg.name, ThresholdConfig())
    r2 = build_report("m", "ds", evaluate(imgs), reg.name, ThresholdConfig())
    assert r1 == r2


@pytest.mark.heavy
def test_full_pipeline_smoke():
    pytest.importorskip("ultralytics")
    pytest.skip("Uçtan-uca smoke: küçük küme + 1 epoch — veri hazır olduğunda çalıştırın.")
