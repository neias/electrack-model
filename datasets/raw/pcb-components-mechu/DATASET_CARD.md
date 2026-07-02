# Veri Kartı — pcb-components-mechu (v6)

## Köken
- **Kaynak:** Roboflow Universe — `test-ywvl9/pcb-components-mechu`, sürüm 6
- **URL:** https://universe.roboflow.com/test-ywvl9/pcb-components-mechu/dataset/6
- **Lisans:** CC BY 4.0 (atıf gerektirir)
- **İndirilen format:** YOLOv11 (segmentasyon/poligon etiketleri)

## Kapsam
- **Görüntü:** 1400 (train 975 / valid 283 / test 142)
- **Kaynak sınıf sayısı:** 16 (tür bazlı, tanımlayıcı değil)
- **Etiket türü:** Instance segmentation (poligon). Dönüştürme sırasında
  eksen-hizalı sınırlayıcı kutuya (`cx cy w h`) indirgenir — bkz.
  `convert.py:yolo_geom_to_bbox`.

## Taksonomi eşlemesi → bizim 6 kanonik tür
`convert.py` → `CLASS_MAPPINGS["pcb_components_mechu"]` içinde tanımlıdır.

| Kaynak sınıf | Kanonik tür | Kutu (yaklaşık) |
|---|---|---|
| IC | `ic` | 1228 |
| capacitor | `capacitor` | 3744 |
| connector | `connector` | 1293 |
| LED | `led` | 1377 |
| resistor | `resistor` | 5704 |
| transistor | `transistor` | 1314 |
| battery, buzzer, clock, diode, display, fuse, inductor, potentiometer, relay, switch | — (kapsam dışı, düşürülür) | 3399 |

Dönüştürme sonrası: **14.660 kutu** korundu, 3399 kutu düşürüldü, 384 görüntü
negatif (hedef sınıf içermeyen) örnek olarak korundu.

## Yeniden üretim
```sh
# Roboflow export'unu bir dizine çıkar, sonra:
python -c "
from pathlib import Path
from electrack.data.classes import ClassRegistry
from electrack.data.convert import convert_roboflow_yolo
reg = ClassRegistry.from_data_yaml()
n2i = {reg.name(i): i for i in range(reg.num_classes)}
print(convert_roboflow_yolo('pcb_components_mechu', Path('<export-dizini>'), n2i))
"
./run.sh prepare-data   # doğrula → 0 sorun beklenir
```

## Notlar / kısıtlar
- Sınıflar **genel türlerdir**; alt-tür (elektrolitik/seramik, delikli/SMD) ayrımı yoktur.
- Poligon→kutu indirgemesi kutuları hafifçe büyütebilir (dönme/eğik parçalarda).
- Tek kaynak; genelleme için ek kaynaklarla birleştirme önerilir (T031 kabul kümesi ayrı tutulmalı).
