# Veri Kartı — fpic-component (FICS-PCB Image Collection)

## Köken
- **Kaynak:** FICS-PCB Image Collection — **FPIC-Component** (CandleLabAI / PCBSegClassNet).
  Dataset Ninja aynası üzerinden indirildi.
- **URL:** https://datasetninja.com/fpic-component · orijinal: https://github.com/CandleLabAI/PCBSegClassNet
- **Lisans:** **CC BY-NC-ND 4.0** — ⚠️ **NonCommercial + NoDerivatives**.
  Bu kaynak **yalnız KİŞİSEL/ARAŞTIRMA** kullanımı için katıldı. Ticari ürünleşme
  hedeflenirse bu küme setten ÇIKARILMALI ve model CC BY 4.0 kaynaklarla yeniden
  eğitilmelidir. (Diğer 3 kaynağımız CC BY 4.0'dır.)
- **İndirilen format:** Supervisely (`meta.json` + `<split>/ann/*.json`), geometri
  **bitmap** (base64+zlib PNG maske + `origin`). `convert.py:convert_supervisely_bitmap`
  her maskeyi eksen-hizalı bbox'a indirir (segmentasyon → tespit).

## Kapsam
- **Görüntü:** 6260 (kaynak split: train 5008 / val 1252). 73 gerçek PCB'den (ön+arka)
  yüksek çözünürlüklü DSLR çekimlerinden **768×768 yamalar**.
- **Kaynak sınıf sayısı:** 25 (referans-tanımlayıcı harfleri: R, C, U, J, Q, LED, …).
- **Yoğunluk:** ortalama **4.7 kutu/görüntü** (maks 79) — yamalandığı için görüntü
  başına ılımlı; ama parçalar **gerçek, minik, yoğun-yerleşimli SMD**. Tam da canlı
  kamera domain-gap'ini (temiz stüdyo verisi ↔ gerçek yoğun kart) kapatmak için eklendi.

## Taksonomi eşlemesi → bizim 6 kanonik tür
`convert.py` → `CLASS_MAPPINGS["fpic"]`. Kutu sayıları (train+val, dönüştürme sonrası korunan):

| kaynak | anlam | kanonik tür | kutu |
|--------|-------|-------------|------|
| R | direnç | `resistor` | 9.312 |
| RA | direnç dizisi | `resistor` | 542 |
| RN | direnç ağı | `resistor` | 513 |
| C | kondansatör | `capacitor` | 8.837 |
| U | IC | `ic` | 3.330 |
| IC | IC | `ic` | 603 |
| J | jak/soket | `connector` | 1.489 |
| P | fiş/konnektör | `connector` | 667 |
| Q | transistör | `transistor` | 893 |
| QA | transistör dizisi | `transistor` | 127 |
| LED | LED | `led` | 74 |
| L, D, CR, CRA, TP, V, M, T, BTN, SW, FB, F, S, JP | indüktör/diyot/test-noktası/mekanik/anahtar/jumper vb. | — (kapsam dışı) | 3.252 (düşürüldü) |

**Kanonik toplam korunan:** 26.387 kutu (resistor 10.367 · capacitor 8.837 · ic 3.933 ·
connector 2.156 · transistor 1.020 · led 74). Düşürülen: 3.252. Boş-maske: 0.

## Split yönlendirmesi
Kaynakta **test split yok**. Dönüştürücü:
- FPIC `train` (5008) → bizim **train**.
- FPIC `val` (1252) → dosya adının kararlı hash'iyle ~50/50 **val (634) ↔ test (618)**
  (`_stable_fraction`, seed 1337). Böylece birleşik kabul **test** seti yoğun gerçek
  kart görür. Train↔test kart-düzeyinde ayrık (train yalnız FPIC-train'den gelir).

## Notlar / kısıtlar
- **connector +2.156, transistor +1.020**: hafızadaki zayıf sınıfları (connector ~0.42,
  transistor ~0.38) doğrudan besler.
- `led` yine az (74) — LED azlığı sürüyor; ileride LED-ağırlıklı kaynak gerekebilir.
- Yamalar 768×768 → **imgsz 768 eğitimiyle birebir** hizalı; minik SMD recall'una yardım eder.
- Yoğunluk ılımlı (maks 79 « electronic'in ~200'ü), bu yüzden MPS `TaskAlignedAssigner`
  çökme riskini artırmaz (bkz. training-env-constraints).
