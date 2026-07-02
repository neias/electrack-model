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
