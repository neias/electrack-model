"""T041 Birim testleri: convert, splits, classes."""

from pathlib import Path

import pytest

from electrack.data.classes import ClassRegistry
from electrack.data.convert import Box, map_class, to_yolo_lines, xyxy_pixel_to_yolo
from electrack.data.splits import assign_split

# --- classes ---


def test_registry_loads_six_mvp_classes():
    reg = ClassRegistry.from_data_yaml(Path("datasets/yolo/data.yaml"))
    assert reg.num_classes == 6
    assert reg.name(0) == "ic"
    assert reg.id_of("transistor") == 5


def test_registry_rejects_noncontiguous_ids():
    with pytest.raises(ValueError):
        ClassRegistry({0: "a", 2: "b"})


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError):
        ClassRegistry({0: "a", 1: "a"})


# --- convert ---


def test_map_class_known_and_unknown():
    assert map_class("roboflow_pcb_components", "IC") == "ic"
    assert map_class("roboflow_pcb_components", "smd_capacitor") is None
    assert map_class("roboflow_pcb_components", "not_a_class") is None


def test_map_class_fpic_reference_designators():
    # Referans-tanımlayıcı harfleri kanonik türe eşlenir (büyük/küçük harf duyarsız).
    assert map_class("fpic", "R") == "resistor"
    assert map_class("fpic", "RN") == "resistor"
    assert map_class("fpic", "C") == "capacitor"
    assert map_class("fpic", "U") == "ic"
    assert map_class("fpic", "J") == "connector"
    assert map_class("fpic", "Q") == "transistor"
    assert map_class("fpic", "LED") == "led"
    # kapsam dışı: diyot/indüktör/jumper vb.
    assert map_class("fpic", "D") is None
    assert map_class("fpic", "L") is None
    assert map_class("fpic", "JP") is None


def _make_supervisely_bitmap(mask, origin):
    """numpy maskeyi Supervisely `bitmap` nesnesine kodla (base64+zlib PNG)."""
    import base64
    import zlib

    import cv2
    import numpy as np

    png = cv2.imencode(".png", (mask.astype(np.uint8) * 255))[1].tobytes()
    return {"data": base64.b64encode(zlib.compress(png)).decode(), "origin": list(origin)}


def test_supervisely_bitmap_to_xyxy_roundtrip():
    import numpy as np

    from electrack.data.convert import supervisely_bitmap_to_xyxy

    # 10x10 maske; içinde (2..5, 3..7) dolu bir dikdörtgen. origin (100, 50).
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:8, 2:6] = 1  # satır 3..7 (y), sütun 2..5 (x)
    bmp = _make_supervisely_bitmap(mask, origin=(100, 50))
    x1, y1, x2, y2 = supervisely_bitmap_to_xyxy(bmp)
    assert (x1, y1) == (100 + 2, 50 + 3)  # origin + min
    assert (x2, y2) == (100 + 5 + 1, 50 + 7 + 1)  # origin + max + 1 (yarı-açık)


def test_supervisely_bitmap_to_xyxy_empty_mask():
    import numpy as np

    from electrack.data.convert import supervisely_bitmap_to_xyxy

    bmp = _make_supervisely_bitmap(np.zeros((5, 5), dtype=np.uint8), origin=(0, 0))
    assert supervisely_bitmap_to_xyxy(bmp) is None


def test_xyxy_pixel_to_yolo():
    cx, cy, w, h = xyxy_pixel_to_yolo(0, 0, 100, 200, 1000, 1000)
    assert (cx, cy, w, h) == (0.05, 0.1, 0.1, 0.2)


def test_to_yolo_lines_excludes_unmapped():
    name_to_id = {"ic": 0, "led": 3}
    boxes = [Box("ic", 0.5, 0.5, 0.1, 0.1), Box("resistor_smd", 0.2, 0.2, 0.1, 0.1)]
    lines = to_yolo_lines(boxes, name_to_id)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")


# --- splits ---


def test_assign_split_values():
    assert assign_split("x.jpg", 1337) in ("train", "val", "test")


def test_assign_split_deterministic():
    assert assign_split("same.jpg", 42) == assign_split("same.jpg", 42)


def test_assign_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        assign_split("x", 1, val_frac=0.6, test_frac=0.6)
