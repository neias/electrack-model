# Veri Kartı — pcb-electronic-components (v1)

## Köken
- **Kaynak:** Roboflow Universe — `carddata-3mujr/pcb-electronic-components`, sürüm 1
- **URL:** https://universe.roboflow.com/carddata-3mujr/pcb-electronic-components/dataset/1
- **Lisans:** CC BY 4.0 (atıf gerektirir)
- **İndirilen format:** YOLO (poligon/segmentasyon etiketleri)

## Kapsam
- **Görüntü:** 672 (train 470 / valid 135 / test 67)
- **Kaynak sınıf sayısı:** 23 (isim büyük/küçük harf karışık: `IC` ve `iC` ayrı)
- **Etiket türü:** Instance segmentation (poligon) → dönüştürmede sınırlayıcı kutuya indirgenir.
- **Yoğunluk:** ~200 kutu/görüntü (belirli bir kart ailesinin yoğun etiketlenmesi).

## Taksonomi eşlemesi → bizim 6 kanonik tür
`convert.py` → `CLASS_MAPPINGS["pcb_electronic_components"]` içinde tanımlıdır.

| Kaynak sınıf(lar) | Kanonik tür | Kutu |
|---|---|---|
| IC + iC | `ic` | 7.604 |
| Capacitor + Electrolytic Capacitor | `capacitor` | 19.890 |
| Connector | `connector` | 4.558 |
| Led | `led` | 784 |
| Resistor | `resistor` | 16.031 |
| Transistor | `transistor` | 4.273 |
| Resistor/Capacitor Jumper, Resistor Network, Test Point, Pads, Pins, Diode, Inductor, Button, Switch, Clock, EM, Ferrite Bead, Jumper, Unknown Unlabeled | — (kapsam dışı) | ~80.900 |

Dönüştürme sonrası: **53.140 kutu** korundu, 80.904 kutu düşürüldü (çoğu 0-ohm jumper),
20 görüntü negatif.

## Yeniden üretim
```sh
python -c "
from pathlib import Path
from electrack.data.classes import ClassRegistry
from electrack.data.convert import convert_roboflow_yolo
reg = ClassRegistry.from_data_yaml()
n2i = {reg.name(i): i for i in range(reg.num_classes)}
print(convert_roboflow_yolo('pcb_electronic_components', Path('<export-dizini>'), n2i))
"
./run.sh prepare-data
```

## Notlar / kısıtlar
- İkinci kaynak; `pcb-components-mechu` ile birleştirilir (farklı workspace → çeşitlilik katkısı).
- Poligon→kutu indirgemesi kutuları hafifçe büyütebilir.
- Çok yoğun/kalabalık kareler; `capacitor` ve `resistor` sınıfları baskın (sınıf dengesizliği).
- Jumper'lar (0-ohm köprü) bilinçli dışlandı — direnç/kondansatör görünümlü ama işlevsel bileşen değil.
