"""Açık veri kümesi → YOLO formatı dönüştürücüler + sınıf-eşleme (FR-010).

Her açık kaynağın kendi taksonomisi bizim 6-sınıf kümesine eşlenir. Eşlenemeyen
(gürültülü/kapsam dışı) sınıflar dışlanır. Dönüştürücüler, kaynak etiketlerini
YOLO `class_id cx cy w h` (normalize) biçimine çevirir.

NOT: Somut kaynak parser'ları (COCO json, Pascal VOC xml, Roboflow export) burada
iskeletlenmiştir; gerçek kaynak indirildiğinde eşleme tabloları doldurulur
(bkz. datasets/raw/<source>/DATASET_CARD.md — T039).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

# Kaynağa özgü sınıf-eşleme tabloları.
# Anahtar: kaynak etiket adı (küçük harf). Değer: bizim kanonik sınıf adı veya None (dışla).
# Gerçek kaynaklar değerlendirildikçe (research.md R3) doldurulur.
CLASS_MAPPINGS: dict[str, dict[str, str | None]] = {
    # Örnek şablon (Roboflow PCB component projeleri için tipik adlar):
    "roboflow_pcb_components": {
        "ic": "ic",
        "integrated_circuit": "ic",
        "capacitor": "capacitor",
        "electrolytic": "capacitor",
        "connector": "connector",
        "header": "connector",
        "led": "led",
        "resistor": "resistor",
        "transistor": "transistor",
        # kapsam dışı örnekler:
        "text": None,
    },
    # Roboflow Universe: test-ywvl9/pcb-components-mechu (v6, CC BY 4.0).
    # 16 tür sınıfı → 6 kanonik türümüz; kalanlar kapsam dışı (None).
    "pcb_components_mechu": {
        "ic": "ic",
        "capacitor": "capacitor",
        "connector": "connector",
        "led": "led",
        "resistor": "resistor",
        "transistor": "transistor",
        # kapsam dışı (MVP taksonomisinde yok):
        "battery": None,
        "buzzer": None,
        "clock": None,
        "diode": None,
        "display": None,
        "fuse": None,
        "inductor": None,
        "potentiometer": None,
        "relay": None,
        "switch": None,
    },
    # Roboflow Universe: carddata-3mujr/pcb-electronic-components (v1, CC BY 4.0).
    # 23 sınıf (isim büyük/küçük harf karışık; map_class küçük harfe indirger).
    "pcb_electronic_components": {
        "ic": "ic",  # "IC" ve "iC" yazımlarının ikisini de kapsar
        "capacitor": "capacitor",
        "electrolytic capacitor": "capacitor",
        "connector": "connector",
        "led": "led",
        "resistor": "resistor",
        "transistor": "transistor",
        # kapsam dışı (jumper/ağ/nokta/mekanik vb.):
        "button": None,
        "capacitor jumper": None,
        "clock": None,
        "diode": None,
        "em": None,
        "ferrite bead": None,
        "inductor": None,
        "jumper": None,
        "pads": None,
        "pins": None,
        "resistor jumper": None,
        "resistor network": None,
        "switch": None,
        "test point": None,
        "unknown unlabeled": None,
    },
    # Roboflow Universe: labsin/pcb-sae2u (v1, CC BY 4.0). Sınıflar isimsiz numaralar;
    # efsane kullanıcı tarafından doğrulandı. SMD ağırlıklı, ~8 kutu/görüntü, bbox.
    "pcb_sae2u": {
        "0": "capacitor",  # SMD kondansatör
        "1": "resistor",  # SMD direnç
        "2": "ic",
        "3": None,  # diyot — kapsam dışı
        "4": None,  # indüktör — kapsam dışı
        "5": "capacitor",  # kutuplu (elektrolitik) kondansatör
        "6": "led",
        "7": "capacitor",  # tantal kondansatör
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


# Roboflow YOLO export'u "valid" der; bizim düzen "val".
_SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def yolo_geom_to_bbox(coords: list[str]) -> list[float]:
    """Bir YOLO etiket satırının (id sonrası) geometrisini `cx cy w h`'ye indirger.

    - 4 değer → zaten `cx cy w h` (tespit formatı), olduğu gibi.
    - Çift sayıda >4 değer → poligon `x1 y1 x2 y2 ...` (segmentasyon); eksen-hizalı
      sınırlayıcı kutuya çevrilir.
    """
    vals = [float(v) for v in coords]
    if len(vals) == 4:
        return vals
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
    raise ValueError(f"Beklenmeyen YOLO geometri uzunluğu: {len(vals)}")


def load_yolo_names(data_yaml: Path) -> dict[int, str]:
    """YOLO/Roboflow `data.yaml`'dan kaynak id→ad eşlemesi (list veya dict biçimi)."""
    doc = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = doc.get("names")
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    raise ValueError(f"{data_yaml}: 'names' bir list veya dict olmalı.")


def is_roboflow_yolo_export(export_dir: Path) -> bool:
    """Dizin bir Roboflow/YOLO export'u mu? (data.yaml + split/images düzeni)."""
    export_dir = Path(export_dir)
    if not (export_dir / "data.yaml").is_file():
        return False
    return any((export_dir / s / "images").is_dir() for s in _SPLIT_MAP)


def convert_roboflow_yolo(
    source: str,
    export_dir: Path,
    name_to_id: dict[str, int],
    out_root: Path = Path("datasets/yolo"),
) -> dict[str, int]:
    """Roboflow/YOLO export'unu bizim `datasets/yolo` düzenimize dönüştürür.

    - Kaynak sınıf id'leri `CLASS_MAPPINGS[source]` üzerinden kanonik türe eşlenir;
      eşlenemeyen (None / kapsam dışı) kutular DÜŞÜRÜLÜR.
    - Görüntüler kopyalanır; etiket satırları yeni id ile yeniden yazılır.
    - Kutusu kalmayan görüntü NEGATİF örnek olarak korunur (boş etiket dosyası).
    - `valid` → `val` olarak yeniden adlandırılır.

    İşlenen görüntü/kutu istatistiklerini döndürür.
    """
    export_dir = Path(export_dir)
    src_id_to_name = load_yolo_names(export_dir / "data.yaml")
    stats = {"images": 0, "negatives": 0, "boxes_in": 0, "boxes_out": 0, "dropped": 0}

    for split_src, split_dst in _SPLIT_MAP.items():
        img_src = export_dir / split_src / "images"
        lbl_src = export_dir / split_src / "labels"
        if not img_src.is_dir():
            continue
        img_dst = out_root / "images" / split_dst
        lbl_dst = out_root / "labels" / split_dst
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img in sorted(p for p in img_src.iterdir() if p.is_file()):
            out_lines: list[str] = []
            lbl = lbl_src / f"{img.stem}.txt"
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    stats["boxes_in"] += 1
                    canonical = map_class(source, src_id_to_name.get(int(parts[0]), ""))
                    if canonical is None or canonical not in name_to_id:
                        stats["dropped"] += 1
                        continue
                    cx, cy, w, h = yolo_geom_to_bbox(parts[1:])
                    out_lines.append(f"{name_to_id[canonical]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats["boxes_out"] += 1

            shutil.copy2(img, img_dst / img.name)
            text = "\n".join(out_lines) + ("\n" if out_lines else "")
            (lbl_dst / f"{img.stem}.txt").write_text(text, encoding="utf-8")
            stats["images"] += 1
            if not out_lines:
                stats["negatives"] += 1

    return stats


def convert_source(source: str, raw_dir: Path, out_dir: Path = Path("datasets/yolo")) -> int:
    """Bir kaynağı YOLO formatına dönüştürür. İşlenen görüntü sayısını döndürür.

    Kaynak formatına göre parser seçimi burada yapılır. Roboflow/YOLO export'ları
    doğrudan desteklenir; diğer formatlar (COCO/VOC) kaynak indirildiğinde eklenir.
    """
    from electrack.data.classes import ClassRegistry

    raw_dir = Path(raw_dir)
    if is_roboflow_yolo_export(raw_dir):
        registry = ClassRegistry.from_data_yaml()
        name_to_id = {registry.name(i): i for i in range(registry.num_classes)}
        stats = convert_roboflow_yolo(source, raw_dir, name_to_id, out_dir)
        return stats["images"]
    raise NotImplementedError(
        f"'{source}' ({raw_dir}) Roboflow/YOLO düzeninde değil. Bu format için parser "
        f"henüz eklenmedi; CLASS_MAPPINGS['{source}'] doldurulup uygun parser eklenir."
    )
