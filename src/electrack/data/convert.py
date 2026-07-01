"""Açık veri kümesi → YOLO formatı dönüştürücüler + sınıf-eşleme (FR-010).

Her açık kaynağın kendi taksonomisi bizim 6-sınıf kümesine eşlenir. Eşlenemeyen
(gürültülü/kapsam dışı) sınıflar dışlanır. Dönüştürücüler, kaynak etiketlerini
YOLO `class_id cx cy w h` (normalize) biçimine çevirir.

NOT: Somut kaynak parser'ları (COCO json, Pascal VOC xml, Roboflow export) burada
iskeletlenmiştir; gerçek kaynak indirildiğinde eşleme tabloları doldurulur
(bkz. datasets/raw/<source>/DATASET_CARD.md — T039).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Kaynağa özgü sınıf-eşleme tabloları.
# Anahtar: kaynak etiket adı (küçük harf). Değer: bizim kanonik sınıf adı veya None (dışla).
# Gerçek kaynaklar değerlendirildikçe (research.md R3) doldurulur.
CLASS_MAPPINGS: dict[str, dict[str, str | None]] = {
    # Örnek şablon (Roboflow PCB component projeleri için tipik adlar):
    "roboflow_pcb_components": {
        "ic": "ic",
        "integrated_circuit": "ic",
        "capacitor": "electrolytic_capacitor",
        "electrolytic": "electrolytic_capacitor",
        "connector": "connector_header",
        "header": "connector_header",
        "led": "led",
        "resistor": "through_hole_resistor",
        "transistor": "to92_transistor",
        # kapsam dışı örnekler:
        "smd_capacitor": None,
        "text": None,
    },
}


@dataclass
class Box:
    class_name: str
    cx: float
    cy: float
    w: float
    h: float


def map_class(source: str, raw_name: str) -> str | None:
    """Kaynak sınıf adını kanonik ada eşle; eşlenemezse None (dışla)."""
    table = CLASS_MAPPINGS.get(source, {})
    return table.get(raw_name.strip().lower())


def to_yolo_lines(boxes: list[Box], name_to_id: dict[str, int]) -> list[str]:
    """Eşlenmiş kutuları YOLO etiket satırlarına çevir (dışlananlar atlanır)."""
    lines: list[str] = []
    for b in boxes:
        if b.class_name not in name_to_id:
            continue
        cid = name_to_id[b.class_name]
        lines.append(f"{cid} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}")
    return lines


def xyxy_pixel_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Piksel köşe kutusunu YOLO normalize (cx, cy, w, h) biçimine çevir."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def convert_source(source: str, raw_dir: Path, out_dir: Path) -> int:
    """Bir kaynağı YOLO formatına dönüştürür. İşlenen görüntü sayısını döndürür.

    Kaynak formatına göre parser seçimi burada yapılır (COCO/VOC/Roboflow).
    Gerçek parser'lar kaynak indirildiğinde eklenir; iskelet, sözleşmeyi sabitler.
    """
    raise NotImplementedError(
        f"'{source}' için parser henüz eklenmedi. Kaynak indirildiğinde "
        f"CLASS_MAPPINGS['{source}'] doldurulup formatına uygun parser eklenir "
        f"(bkz. {raw_dir}/DATASET_CARD.md)."
    )
