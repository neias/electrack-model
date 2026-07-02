# Veri Kartı — pcb-sae2u (v1)

## Köken
- **Kaynak:** Roboflow Universe — `labsin/pcb-sae2u`, sürüm 1
- **URL:** https://universe.roboflow.com/labsin/pcb-sae2u/dataset/1
- **Lisans:** CC BY 4.0 (atıf gerektirir)
- **İndirilen format:** YOLO (eksen-hizalı bbox — poligon değil)

## Kapsam
- **Görüntü:** 5100 (train 3570 / valid 1021 / test 509)
- **Kaynak sınıf sayısı:** 8 (isimsiz numaralar `0`–`7`)
- **Yoğunluk:** ~8 kutu/görüntü (sağlıklı; yoğun/minik-nesne değil).

## Taksonomi eşlemesi → bizim 6 kanonik tür
Sınıflar isimsiz numaralı; **efsane kullanıcı tarafından doğrulandı** (Roboflow sayfası).
`convert.py` → `CLASS_MAPPINGS["pcb_sae2u"]`.

| id | kaynak anlamı | kanonik tür | kutu |
|----|---------------|-------------|------|
| 0 | SMD kondansatör | `capacitor` | 17.892 |
| 1 | SMD direnç | `resistor` | 16.686 |
| 2 | IC | `ic` | 4.068 |
| 5 | kutuplu (elektrolitik) kondansatör | `capacitor` | 282 |
| 6 | LED | `led` | 324 |
| 7 | tantal kondansatör | `capacitor` | 372 |
| 3 | diyot | — (kapsam dışı) | 234 |
| 4 | indüktör | — (kapsam dışı) | 774 |

Dönüştürme sonrası: **39.624 kutu** korundu, 1008 düşürüldü (diyot + indüktör), 48 negatif.

## Notlar / kısıtlar
- SMD ağırlıklı — mecha ile birleştirilir (çeşitlilik + hacim). Bu kaynakta
  `connector`/`transistor` yok; onlar mecha'dan gelir.
- Birleşik eğitim seti (mecha + pcb_sae2u): 6500 görüntü, ~54.000 kutu.
  Sınıf dengesizliği: capacitor/resistor ~22k baskın; connector/transistor/led ~1.3-1.7k.
- `PCB Electronic components` (carddata-3mujr) kümesi bilinçli DIŞLANDI: ~200 kutu/görüntü
  yoğunluğu MPS atayıcı çökmesine yol açıyor ve minik-nesne doğruluğu düşürüyor.
