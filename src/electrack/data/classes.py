"""Komponent sınıf kayıt defteri — TAMAMEN `data.yaml`'dan türetilir (FR-009, US3).

Hiçbir sınıf adı burada sabit-kodlanmaz; sınıf listesi tek doğru kaynak olan
`datasets/yolo/data.yaml` dosyasından okunur. Yeni sınıf eklemek yalnızca
`data.yaml`'ı düzenlemeyi gerektirir — bu modül veya hattın geri kalanı değişmez.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from electrack.config_constants import UNKNOWN_LABEL

# İnsan-okur adlar (opsiyonel; tespit sözleşmesini etkilemez). Kanonik ada göre.
DISPLAY_NAMES: dict[str, str] = {
    "ic": "Entegre / IC",
    "capacitor": "Kondansatör",
    "connector": "Konnektör",
    "led": "LED",
    "resistor": "Direnç",
    "transistor": "Transistör",
}

DEFAULT_DATA_YAML = Path("datasets/yolo/data.yaml")


class ClassRegistry:
    """`data.yaml`'daki `names` eşlemesinden türetilen sınıf kümesi."""

    def __init__(self, id_to_name: dict[int, str]):
        if not id_to_name:
            raise ValueError("Sınıf kümesi boş — data.yaml 'names' eksik.")
        # id'ler 0..N-1 aralığında ve sürekli olmalı.
        expected = list(range(len(id_to_name)))
        if sorted(id_to_name.keys()) != expected:
            raise ValueError(
                f"Sınıf id'leri 0..{len(id_to_name) - 1} aralığında sürekli olmalı; "
                f"bulunan: {sorted(id_to_name.keys())}"
            )
        self._id_to_name: dict[int, str] = dict(id_to_name)
        self._name_to_id: dict[str, int] = {v: k for k, v in id_to_name.items()}
        if len(self._name_to_id) != len(self._id_to_name):
            raise ValueError("Sınıf adları benzersiz olmalı.")

    @classmethod
    def from_data_yaml(cls, path: Path | None = None) -> ClassRegistry:
        path = Path(path) if path else DEFAULT_DATA_YAML
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        names = doc.get("names")
        if names is None:
            raise ValueError(f"{path} içinde 'names' yok.")
        # Ultralytics hem dict {id: name} hem list [name, ...] destekler.
        if isinstance(names, list):
            id_to_name = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            id_to_name = {int(k): v for k, v in names.items()}
        else:
            raise ValueError("'names' bir list veya dict olmalı.")
        return cls(id_to_name)

    @property
    def num_classes(self) -> int:
        return len(self._id_to_name)

    def names(self) -> list[str]:
        return [self._id_to_name[i] for i in range(self.num_classes)]

    def name(self, class_id: int) -> str:
        return self._id_to_name[class_id]

    def id_of(self, name: str) -> int:
        return self._name_to_id[name]

    def is_valid_id(self, class_id: int) -> bool:
        return class_id in self._id_to_name

    def display_name(self, class_id: int) -> str:
        canonical = self._id_to_name[class_id]
        return DISPLAY_NAMES.get(canonical, canonical)

    def output_labels(self) -> list[str]:
        """Çıktı sözleşmesindeki geçerli class_name kümesi: sınıflar + 'unknown'."""
        return self.names() + [UNKNOWN_LABEL]
