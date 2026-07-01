"""T041 Birim testleri: metrics (IoU, recall, FP oranı, unknown davranışı)."""

from electrack.eval.metrics import (
    ImageEval,
    evaluate,
    iou_xyxy,
    unknown_behavior,
)


def test_iou_identical():
    assert iou_xyxy([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_iou_disjoint():
    assert iou_xyxy([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_iou_half_overlap():
    v = iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2])
    assert abs(v - (2.0 / 6.0)) < 1e-9


def test_recall_correct_class():
    gts = [{"class_id": 0, "bbox": [0.1, 0.1, 0.3, 0.3]}]
    preds = [{"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "ic", "class_id": 0, "confidence": 0.9}]
    agg = evaluate([ImageEval(gts=gts, preds=preds)])
    assert agg.overall_recall == 1.0
    assert agg.false_positive_rate == 0.0


def test_wrong_class_no_recall():
    gts = [{"class_id": 0, "bbox": [0.1, 0.1, 0.3, 0.3]}]
    preds = [{"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "led", "class_id": 3, "confidence": 0.9}]
    agg = evaluate([ImageEval(gts=gts, preds=preds)])
    assert agg.overall_recall == 0.0
    # Gerçek komponentle örtüşüyor → phantom DEĞİL.
    assert agg.false_positive_rate == 0.0


def test_phantom_on_empty_is_fp():
    # Negatif görüntü: GT yok, tahmin var → phantom FP.
    preds = [{"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "ic", "class_id": 0, "confidence": 0.9}]
    agg = evaluate([ImageEval(gts=[], preds=preds)])
    assert agg.false_positive_rate == 1.0


def test_unknown_over_real_component_not_fp():
    gts = [{"class_id": 0, "bbox": [0.1, 0.1, 0.3, 0.3]}]
    preds = [
        {"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "unknown", "class_id": None, "confidence": 0.4}
    ]
    agg = evaluate([ImageEval(gts=gts, preds=preds)])
    assert agg.false_positive_rate == 0.0  # gerçek nesne üzerinde
    assert agg.overall_recall == 0.0  # ama doğru tür sayılmaz


def test_unknown_behavior_flags_high_conf_mis():
    ambiguous = [
        ImageEval(
            gts=[{"class_id": -1, "bbox": [0.1, 0.1, 0.3, 0.3]}],
            preds=[
                {"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "ic", "class_id": 0, "confidence": 0.9}
            ],
        )
    ]
    res = unknown_behavior(ambiguous, class_threshold=0.5)
    assert res["high_conf_misclassifications"] == 1
    assert res["unknown_recall"] == 0.0


def test_unknown_behavior_catches_unknown():
    ambiguous = [
        ImageEval(
            gts=[{"class_id": -1, "bbox": [0.1, 0.1, 0.3, 0.3]}],
            preds=[
                {
                    "bbox": [0.1, 0.1, 0.3, 0.3],
                    "class_name": "unknown",
                    "class_id": None,
                    "confidence": 0.4,
                }
            ],
        )
    ]
    res = unknown_behavior(ambiguous, class_threshold=0.5)
    assert res["high_conf_misclassifications"] == 0
    assert res["unknown_recall"] == 1.0
