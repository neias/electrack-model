# Quickstart & Doğrulama Kılavuzu: PCB Komponent Tespiti

Bu kılavuz, hattı uçtan uca çalıştırıp özelliğin çalıştığını kanıtlayan doğrulama senaryolarını içerir. Uygulama kodu/gövdeleri burada yoktur (bkz. `tasks.md` + implementasyon). Referanslar: [plan.md](./plan.md), [data-model.md](./data-model.md), [contracts/](./contracts/).

## Önkoşullar

- macOS, Apple Silicon (hedef: **M4 Mac Mini**).
- Python 3.11 + sanal ortam.
- Bağımlılıklar: `ultralytics`, `torch` (MPS), `coremltools`, `opencv-python`, `pyyaml`, `pytest`, `jsonschema`.
- (Opsiyonel) Roboflow/veri indirme için ağ — yalnızca `prepare-data` aşamasında; çıkarım çevrimdışıdır.

## Hat aşamaları (CLI)

Tek giriş noktası `src/electrack/cli.py`; alt komutlar:

| Komut | Amaç | Ana çıktı |
|-------|------|-----------|
| `prepare-data` | Açık kümeleri indir/dönüştür + kendi veriyi birleştir → YOLO formatı | `datasets/yolo/` + `data.yaml` |
| `train` | Konfig ile YOLO eğitimi | `models/weights/best.pt` |
| `export` | `.pt` → `.mlpackage` (NMS + eşik gömülü) | `models/export/electrack.mlpackage` |
| `evaluate` | Sabit kabul kümesinde metrik + rapor | `reports/eval-<model_id>.json` |

---

## Senaryo 1 — Uçtan uca "duman testi" (küçük küme)

**Amaç**: Hattın bütünüyle çalıştığını hızlıca kanıtlamak (US2).

1. Küçük örnek küme ile veri hazırla:
   `python -m electrack.cli prepare-data --config configs/smoke.yaml`
   **Beklenen**: `datasets/yolo/data.yaml` oluşur; `images/` ve `labels/` train/val/test dolu; negatif (boş etiket) görüntüler dahil.
2. 1 epoch eğit:
   `python -m electrack.cli train --config src/electrack/training/config/mvp.yaml --epochs 1`
   **Beklenen**: `models/weights/best.pt` üretilir; hata yok.
3. Core ML'e aktar:
   `python -m electrack.cli export --weights models/weights/best.pt`
   **Beklenen**: `models/export/electrack.mlpackage` oluşur; NMS + eşik gömülü (bkz. [model-io.md](./contracts/model-io.md)).
4. Değerlendir:
   `python -m electrack.cli evaluate --model models/export/electrack.mlpackage --dataset datasets/acceptance`
   **Beklenen**: `reports/eval-*.json`, [`eval-report.schema.json`](./contracts/eval-report.schema.json) şemasına uyar.

**Başarı ölçütü**: 4 adım hatasız; şema doğrulaması geçer. (Metrik değerleri küçük kümede düşük olabilir — burada amaç boru hattı bütünlüğü.)

---

## Senaryo 2 — Tespit çıktısı sözleşmesi (US1)

**Amaç**: Model çıktısının tüketen uygulama sözleşmesine uyduğunu kanıtlamak.

1. Örnek bir PCB görüntüsünde çıkarım al (referans sarmalayıcı):
   `python -m electrack.cli infer --model models/export/electrack.mlpackage --image samples/board_01.jpg --json out.json`
2. Çıktıyı şemaya karşı doğrula:
   `python -m electrack.cli validate-output out.json` (veya `jsonschema` ile [detection-output.schema.json](./contracts/detection-output.schema.json)).

**Başarı ölçütü**:
- Her tespitte `bbox` normalize [0-1] xyxy; `class_name` 6 sınıf veya `unknown`; `confidence` [0,1] (FR-002).
- IC/kondansatör/konnektör/LED/direnç/TO-92 görünürse doğru türle ve makul kutuyla döner.

---

## Senaryo 3 — Yanlış-pozitif bastırma & "bilinmeyen" (US1, SC-002/SC-004)

1. **Boş yüzey / komponent-olmayan** görüntüde çıkarım:
   `... infer --image samples/empty_surface.jpg` → **Beklenen**: `detections` boş veya çok düşük güven yok (hayali tespit yok — FR-007).
2. **Belirsiz/eğitilmemiş nesne**: `... infer --image samples/ambiguous.jpg` → **Beklenen**: yüksek-güvenli yanlış tür yerine `class_name="unknown"` (FR-004).

**Başarı ölçütü**: Boş girdide FP yok; belirsiz nesnede `unknown` üretilir.

---

## Senaryo 4 — Kabul değerlendirmesi (SC-001, SC-002, SC-003)

`python -m electrack.cli evaluate --model models/export/electrack.mlpackage --dataset datasets/acceptance --measure-latency`

**Başarı ölçütü (verdict = PASS)**:
- `overall_recall ≥ 0.80` (mikro-ortalama @ IoU 0.5) — SC-001.
- `false_positive_rate < 0.10` — SC-002.
- `latency.fps ≥ 15` (~50 komponent, Apple M4 Mac Mini) — SC-003.
- `unknown_behavior.high_conf_misclassifications` düşük — SC-004.

Rapor `reports/eval-<model_id>.json` olarak yazılır ve [şemaya](./contracts/eval-report.schema.json) uyar.

---

## Senaryo 5 — Yeni tür ekleme (US3, SC-005)

1. `datasets/yolo/data.yaml` sınıf listesine yeni sınıfı ekle (ör. `6: smd_resistor`) ve etiketli veriyi koy.
2. Yeniden eğit → aktar → değerlendir (kod/mimari değişikliği YOK).

**Başarı ölçütü**: Yeni türü tanıyan model üretilir; mevcut türlerin recall'ü kabul eşiğinin altına düşmez (SC-005). Sözleşme yapısı değişmez (yalnızca `class_name` enum'u genişler).

---

## Tekrar üretilebilirlik kontrolü (SC-006)

Aynı `data.yaml` + sabit seed ile `train`+`export`+`evaluate` yeniden çalıştırıldığında kabul metrikleri karşılaştırılabilir (küçük varyans içinde) sonuç verir. Model id/hash ve dataset id raporda kayıtlıdır.
