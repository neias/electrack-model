# Etiketleme Kuralları — PCB Komponent Tespiti

Bu belge, tutarlı ground-truth üretmek için etiketleme kurallarını tanımlar. Tutarlı etiketleme, SC-001 (recall) ve SC-002 (FP oranı) metriklerinin anlamlı olması için kritiktir.

## Genel

- Format: **YOLO** (her görüntü için `<name>.txt`; satır: `class_id cx cy w h`, tümü [0-1] normalize).
- Kutu **eksen-hizalı** ve komponenti **sıkıca** sarmalı (gövde + belirgin bacaklar/pinler dahil, aşırı boşluk yok).
- Kısmen kapalı komponent: **görünen kısmı** sar; %50'den azı görünüyorsa ve tür belirsizse etiketleme (veya "belirsiz" olarak ayır — aşağıya bakın).

## Sınıflar (MVP)

| id | name | Kapsam / ipuçları |
|----|------|-------------------|
| 0 | `ic` | Entegre/IC gövdeleri (DIP, çok bacaklı kare/dikdörtgen paketler). |
| 1 | `electrolytic_capacitor` | Silindirik elektrolitik kondansatör (kutup bandı görünür). |
| 2 | `connector_header` | Konnektör/header sıraları, pin başlıkları, soketler. |
| 3 | `led` | LED (kubbe/renkli gövde). |
| 4 | `through_hole_resistor` | Delikli (through-hole) direnç (renk bantlı, eksenli). |
| 5 | `to92_transistor` | TO-92 paket transistör (yarım-daire plastik gövde, 3 bacak). |

> Sınıf kümesi `datasets/yolo/data.yaml`'dan yönetilir. Yeni tür (ör. SMD) eklerken bu tabloyu ve `data.yaml`'ı birlikte güncelleyin (bkz. `docs/adding-a-class.md`).

## "Bilinmeyen" politikası (FR-004)

- Model çıktısında `unknown` **eğitilen bir sınıf değildir**; eşik-türetimli bir etikettir (research.md R5).
- Bu nedenle **belirsiz/eğitilmemiş komponentleri kutu olarak etiketlemeyin** (6 sınıftan biri değillerse). Onları arka planda bırakın; model düşük sınıf-güveniyle `unknown` üretmeyi öğrenir.
- İsteğe bağlı: Değerlendirme için ayrı bir `ambiguous/` klasöründe, belirsiz nesneleri içeren görüntüler toplanabilir (unknown davranışı ölçümü — SC-004).

## Negatif (komponentsiz) görüntüler — FR-007 / SC-002 için ZORUNLU

- Boş yüzey, komponent-olmayan nesne (kalem, parmak, arka plan) görüntüleri **boş etiket dosyası** (`.txt` var, içi boş) ile eklenir.
- Bunlar yanlış-pozitif bastırmayı eğitir ve kabul kümesinde FP oranı ölçümü için gereklidir.

## Zorlu koşullar (FR-008)

Kabul ve eğitim kümeleri şu koşullardan gerçek örnekler içermeli: zayıf/aşırı ışık, yansıma, çok yakın/uzak kart, kısmi kapanma/üst üste binme, kalabalık kartlar.

## Kalite kontrol

- Her kutu geçerli bir `class_id` (data.yaml kümesinde) ve [0-1] koordinat taşımalı (`datasets.py` doğrular).
- Şüpheli/çift etiketleri gözden geçirin; NMS mantığı tek nesne = tek kutu varsayar.
