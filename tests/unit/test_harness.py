"""Değerlendirme koşucusunun saf yardımcıları (ağır bağımlılık gerektirmez)."""

from electrack.eval.harness import load_yolo_gt, resolve_split


def test_load_yolo_gt_converts_cxcywh_to_xyxy(tmp_path):
    p = tmp_path / "img.txt"
    # class cx cy w h → xyxy: (0.5,0.5,0.2,0.4) => [0.4,0.3,0.6,0.7]
    p.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    gts = load_yolo_gt(p)
    assert len(gts) == 1
    assert gts[0]["class_id"] == 2
    bbox = gts[0]["bbox"]
    assert bbox == [0.4, 0.3, 0.6, 0.7]


def test_load_yolo_gt_clamps_and_skips_degenerate(tmp_path):
    p = tmp_path / "img.txt"
    # 1. kutu kenardan taşar (clamp), 2. kutu sıfır alan (atlanır)
    p.write_text("0 0.05 0.05 0.2 0.2\n1 0.5 0.5 0.0 0.3\n", encoding="utf-8")
    gts = load_yolo_gt(p)
    assert len(gts) == 1
    assert gts[0]["bbox"][0] == 0.0 and gts[0]["bbox"][1] == 0.0


def test_load_yolo_gt_missing_file_is_negative(tmp_path):
    assert load_yolo_gt(tmp_path / "yok.txt") == []


def test_resolve_split():
    from pathlib import Path

    img, lbl = resolve_split(Path("datasets/yolo"), "test")
    assert img == Path("datasets/yolo/images/test")
    assert lbl == Path("datasets/yolo/labels/test")
    img2, lbl2 = resolve_split(Path("datasets/acceptance"), None)
    assert img2 == Path("datasets/acceptance/images")
