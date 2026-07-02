# Data Model: PCB Komponent Tespiti

Bu belge, özelliğin veri varlıklarını (entity), alanlarını, ilişkilerini ve doğrulama kurallarını tanımlar. Uygulama/koda özgü tipler yerine mantıksal şema verilmiştir; somut serileştirme sözleşmeleri `contracts/` altındadır.

## 1. ComponentClass (Komponent Türü)

Tanınabilir komponent kategorisi. Tür kümesi `data.yaml` içinden yönetilir ve genişletilebilir (FR-009).

| Alan | Tip | Kurallar |
|------|-----|----------|
| `id` | integer | 0'dan başlayan, kararlı sınıf indeksi. Yeni sınıflar sona eklenir; mevcut id'ler değişmez. |
| `name` | string | Kanonik ad (snake_case). Benzersiz. |
| `display_name` | string | İnsan-okur ad (TR/EN). |

**MVP sınıf kümesi** (FR-003):

| id | name | display_name |
|----|------|--------------|
| 0 | `ic` | Entegre / IC |
| 1 | `capacitor` | Kondansatör |
| 2 | `connector` | Konnektör |
| 3 | `led` | LED |
| 4 | `resistor` | Direnç |
| 5 | `transistor` | Transistör |

> Sınıflar GENEL komponent türleridir. Kamera karesinden elektrolitik/seramik ya da delikli/SMD alt-türü güvenilir ayırt edilemediğinden ve mevcut açık veri kümeleri (bkz. `convert.py` CLASS_MAPPINGS) genel türle etiketlendiğinden, taksonomi genel tutulmuştur.

**"unknown" (bilinmeyen)**: Eğitilmiş bir sınıf id'si DEĞİLDİR; çıktı sözleşmesinde ayrı bir etiket olarak yüzeylenir (bkz. Detection.class_name). Model bir nesneyi bulur ama hiçbir sınıfı güven eşiğini geçemezse `unknown` üretilir (FR-004). Karar mantığı `research.md` → "Unknown/background stratejisi"nde.

**Doğrulama**:
- `data.yaml` içindeki sınıf sayısı = model başlık (head) sınıf sayısı.
- Yeni sınıf eklemek yalnızca `data.yaml` + veri + yeniden eğitim gerektirir; id yeniden kullanılmaz.

## 2. Image / Frame (Görüntü / Kamera Karesi)

Modele girdi olan tek bir PCB görüntüsü (FR-001, FR-001a).

| Alan | Tip | Kurallar |
|------|-----|----------|
| `pixels` | RGB raster | Herhangi bir çözünürlük; en-boy oranı korunur. Ön-işleme (yeniden boyutlandırma + normalize) model tarafında yapılır. |
| `width`, `height` | integer | Orijinal piksel boyutları (kutu ölçekleme referansı; kutular normalize döndüğünden uygulama bunlarla piksel'e çevirebilir). |

**Doğrulama**: 3-kanallı RGB; alfa/gri-tonlama girdide RGB'ye normalize edilir.

## 3. Detection (Tespit) — Model çıktısı birimi

Model çıktısının tek bir birimi (FR-002). Serileştirme sözleşmesi: `contracts/detection-output.schema.json`.

| Alan | Tip | Kurallar |
|------|-----|----------|
| `bbox` | float[4] | Normalize [0-1], eksen-hizalı, köşe biçimi `[x_min, y_min, x_max, y_max]`. `0 ≤ x_min < x_max ≤ 1`, `0 ≤ y_min < y_max ≤ 1`. |
| `class_name` | string | 6 MVP sınıf adından biri veya `"unknown"`. |
| `class_id` | integer \| null | Sınıf id'si; `unknown` için `null`. |
| `confidence` | float | [0, 1]. Model-içi eşiğin üzerinde (FR-002a). |

**Kurallar**:
- Liste, model-içi güven eşiği + NMS sonrası yalnızca kesinleşmiş tespitleri içerir (FR-002a).
- Boş yüzey / komponent-olmayan girdi → hayali tespit yok; liste boş olabilir (FR-007).
- Aynı nesne için tek kutu (NMS ile çakışmalar bastırılır).

## 4. AnnotatedImage (Etiketli Görüntü) — Eğitim/değerlendirme verisi

Veri kümesindeki bir görüntü ve onun ground-truth etiketleri.

| Alan | Tip | Kurallar |
|------|-----|----------|
| `image_path` | path | `datasets/yolo/images/<split>/<name>.jpg` |
| `label_path` | path | `datasets/yolo/labels/<split>/<name>.txt` (YOLO formatı) |
| `boxes` | GroundTruthBox[] | 0+ kutu. Boş dosya = negatif (komponentsiz) görüntü. |
| `split` | enum | `train` \| `val` \| `test` |
| `source` | string | Köken (açık küme adı veya "own_capture"); izlenebilirlik için. |

### GroundTruthBox

| Alan | Tip | Kurallar |
|------|-----|----------|
| `class_id` | integer | Geçerli `ComponentClass.id`. |
| `cx, cy, w, h` | float | YOLO formatı: normalize merkez + boyut [0-1]. (Değerlendirmede köşe xyxy'ye çevrilir.) |

**Doğrulama**:
- Her `class_id` `data.yaml` kümesinde olmalı.
- Koordinatlar [0-1]; `w>0, h>0`.
- Negatif görüntüler (boş label) kabul edilir ve FP bastırma eğitimi için gereklidir.

## 5. Dataset (Veri Kümesi)

Etiketli görüntülerin, deterministik bölünmüş koleksiyonu (FR-010, FR-013).

| Alan | Tip | Kurallar |
|------|-----|----------|
| `data_yaml` | path | Sınıf listesi + train/val/test yolları. Genişletme noktası. |
| `splits` | map | train/val/test → görüntü sayıları. Deterministik (sabit seed). |
| `class_distribution` | map | Sınıf başına örnek sayısı (dengesizlik takibi). |

**Alt tür — AcceptanceSet (Kabul Kümesi)**: Sabit, değişmeyen test kümesi. Kural: ≥~20 farklı kart, ~300+ etiketli görüntü, her MVP türünden yeterli örnek (FR-013). Bir kez dondurulur; model seçiminden etkilenmez.

## 6. EvaluationReport (Değerlendirme Raporu)

Sabit test kümesi üzerinde ölçülen kabul çıktısı (FR-011, SC-001, SC-002). Şema: `contracts/eval-report.schema.json`.

| Alan | Tip | Kurallar |
|------|-----|----------|
| `model_id` | string | Değerlendirilen artefakt (ör. `.mlpackage` hash/sürüm). |
| `dataset_id` | string | Kabul kümesi sürümü. |
| `overall_recall` | float | Mikro-ortalama recall @ IoU 0.5. Kabul: ≥ 0.80 (SC-001). |
| `false_positive_rate` | float | Üretilen tespitler içinde yanlış-pozitif oranı. Kabul: < 0.10 (SC-002). |
| `per_class` | map | Sınıf başına recall / precision / destek (support). |
| `unknown_behavior` | object | Belirsiz örneklerde yüksek-güvenli yanlış sınıflandırma sayısı (SC-004). |
| `latency_fps` | object | Hedef donanımda ~50 komponentli karede FPS/ms (SC-003). |
| `verdict` | enum | `PASS` \| `FAIL` (tüm kabul eşikleri karşılanırsa PASS). |

## İlişkiler

```text
ComponentClass 1..* ──< data.yaml >── 1 Dataset
Dataset 1 ──< contains >── * AnnotatedImage ──< has >── * GroundTruthBox >── 1 ComponentClass
Image (input) ──[model]──> * Detection >── 0..1 ComponentClass (veya "unknown")
AcceptanceSet ──[evaluate]──> 1 EvaluationReport
```

## Durum geçişleri (model artefaktı yaşam döngüsü)

```text
raw data → converted (YOLO) → split (train/val/test) → trained (.pt)
         → exported (.mlpackage) → evaluated (EvaluationReport)
         → PASS ? released : iterate (veri/hiperparametre)
```
