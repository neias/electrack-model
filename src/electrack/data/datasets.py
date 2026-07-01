"""Veri kümesi keşfi + doğrulama (data-model.md §4/§5).

YOLO formatı veri kümesinin yapısal geçerliliğini kontrol eder:
- data.yaml tutarlılığı (sınıf kümesi)
- etiket dosyalarının biçimi (class_id cx cy w h, [0-1] normalize)
- negatif (boş) görüntülerin kabulü
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from electrack.data.classes import ClassRegistry


@dataclass
class LabelIssue:
    label_path: str
    line_no: int
    message: str


@dataclass
class DatasetStats:
    num_images: int = 0
    num_labels: int = 0
    num_negative_images: int = 0  # boş etiket dosyası
    num_boxes: int = 0
    class_counts: dict = field(default_factory=dict)
    issues: list[LabelIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _parse_label_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"5 alan bekleniyordu, {len(parts)} bulundu")
    cid = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:])
    return cid, cx, cy, w, h


def validate_label_file(path: Path, registry: ClassRegistry) -> tuple[int, list[LabelIssue]]:
    """Bir YOLO etiket dosyasını doğrular. (kutu_sayısı, sorunlar) döndürür."""
    issues: list[LabelIssue] = []
    n_boxes = 0
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, issues  # negatif (boş) görüntü — geçerli.
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cid, cx, cy, w, h = _parse_label_line(line)
        except ValueError as e:
            issues.append(LabelIssue(str(path), i, str(e)))
            continue
        if not registry.is_valid_id(cid):
            issues.append(LabelIssue(str(path), i, f"geçersiz class_id={cid}"))
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= val <= 1.0):
                issues.append(LabelIssue(str(path), i, f"{name}={val} [0,1] dışında"))
        if w <= 0 or h <= 0:
            issues.append(LabelIssue(str(path), i, "w/h pozitif olmalı"))
        n_boxes += 1
    return n_boxes, issues


def validate_split(images_dir: Path, labels_dir: Path, registry: ClassRegistry) -> DatasetStats:
    """Bir bölünmedeki (train/val/test) tüm görüntü+etiket çiftlerini doğrular."""
    stats = DatasetStats()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in image_exts)
    stats.num_images = len(images)
    for img in images:
        label = labels_dir / (img.stem + ".txt")
        if not label.exists():
            stats.issues.append(LabelIssue(str(label), 0, "etiket dosyası yok"))
            continue
        stats.num_labels += 1
        n_boxes, issues = validate_label_file(label, registry)
        stats.issues.extend(issues)
        stats.num_boxes += n_boxes
        if n_boxes == 0:
            stats.num_negative_images += 1
        else:
            for line in label.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        cid = int(line.split()[0])
                        stats.class_counts[cid] = stats.class_counts.get(cid, 0) + 1
                    except (ValueError, IndexError):
                        pass
    return stats
