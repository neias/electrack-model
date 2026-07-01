---
description: "Task list for PCB Komponent Tespiti implementation"
---

# Tasks: PCB Komponent Tespiti (Cihaz-Üstü Nesne Tespit Modeli)

**Input**: Design documents from `/specs/001-pcb-component-detection/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Bu özellik değerlendirme/kabul odaklı olduğundan (FR-011, SC-001..SC-004) yalnızca doğrudan gereksinime hizmet eden kontrat/entegrasyon testleri dahil edilmiştir (tam TDD değil).

**Organization**: Görevler kullanıcı hikâyelerine göre gruplanmıştır; her hikâye bağımsız test edilebilir bir artış (increment) oluşturur.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Paralel çalışabilir (farklı dosya, bağımlılık yok)
- **[Story]**: US1/US2/US3 — ilgili kullanıcı hikâyesi
- Her görevde tam dosya yolu vardır

## Path Conventions

Tek proje (ML hattı): `src/electrack/`, `tests/`, `datasets/`, `models/`, `reports/` repo kökünde (bkz. plan.md → Project Structure).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Proje iskeleti ve temel yapı

- [X] T001 Repo yapısını oluştur (`src/electrack/{data,labeling,training,export,inference,eval}/`, `datasets/{raw,yolo,acceptance}/`, `models/{weights,export}/`, `reports/`, `tests/{contract,integration,unit}/`) plan.md'ye göre
- [X] T002 Python 3.11 projesini başlat: `pyproject.toml` + bağımlılıklar (ultralytics, torch, coremltools, opencv-python, pillow, numpy, pyyaml, jsonschema, pytest) ve `.gitignore` (datasets/, models/, reports/) repo kökünde
- [X] T003 [P] Lint/format (ruff + black) ve pytest yapılandırması `pyproject.toml` içinde
- [X] T004 [P] Komponent sınıf kayıt defterini ve `datasets/yolo/data.yaml` iskeletini (6 MVP sınıf) `src/electrack/data/classes.py` + `datasets/yolo/data.yaml` içinde tanımla
- [X] T005 [P] Etiketleme kurallarını (sınıf tanımları, kutu kuralları, "bilinmeyen" politikası, negatif örnekler) `src/electrack/labeling/guidelines.md` içine yaz
- [X] T006 [P] Eşik/sabit değerleri (varsayılan `det_threshold`, `class_threshold`, IoU=0.5, girdi boyutu) `src/electrack/config_constants.py` içinde merkezileştir

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Herhangi bir modeli üretebilmek için gereken çekirdek altyapı (tüm hikâyeler buna bağlıdır)

**⚠️ CRITICAL**: Bu faz bitmeden hiçbir kullanıcı hikâyesi çalışması başlayamaz

- [X] T007 Konfig yönetimi + determinizm/seed yardımcılarını `src/electrack/config.py` içinde uygula
- [X] T008 [P] Loglama altyapısını `src/electrack/logging_setup.py` içinde yapılandır
- [X] T009 [P] Veri kümesi keşfi + doğrulama (AnnotatedImage/GroundTruthBox şeması, `data.yaml` tutarlılığı, koordinat sınırları) `src/electrack/data/datasets.py` içinde uygula
- [X] T010 [P] Açık veri kümesi → YOLO dönüştürücüleri + sınıf-eşleme tabloları (COCO/VOC/Roboflow) `src/electrack/data/convert.py` içinde uygula
- [X] T011 Deterministik train/val/test bölünmesini (hash-tabanlı, sabit seed) `src/electrack/data/splits.py` içinde uygula
- [X] T012 YOLO eğitim giriş noktasını (konfig-odaklı) `src/electrack/training/train.py` + `src/electrack/training/config/mvp.yaml` içinde uygula
- [X] T013 Core ML dışa aktarımını (`nms=True`, gömülü `det/class` eşikleri) `src/electrack/export/to_coreml.py` içinde uygula
- [X] T014 CLI iskeletini alt komutlarla (`prepare-data|train|export|evaluate|infer`) `src/electrack/cli.py` içinde uygula (gövdeler ilgili hikâye fazlarında bağlanır)

**Checkpoint**: Altyapı hazır — kullanıcı hikâyeleri başlayabilir

---

## Phase 3: User Story 1 - Bir kamera karesinde komponentleri tespit etme (Priority: P1) 🎯 MVP

**Goal**: Tüketen uygulamaya, bir kareden normalize [0-1] xyxy kutu + sınıf (6 tür veya `unknown`) + güven döndüren, çevrimdışı çalışan bir çıkarım yolu ve çalışan bir model sağlamak.

**Independent Test**: Bir PCB görüntüsü verildiğinde çıktı `detection-output.schema.json`'a uyar; boş yüzeyde hayali tespit üretmez; belirsiz nesnede `unknown` döner (Senaryo 2–3, quickstart.md).

### Tests for User Story 1

- [X] T015 [P] [US1] Kontrat testi: çıkarım çıktısının `contracts/detection-output.schema.json`'a uyduğunu doğrula → `tests/contract/test_detection_output.py`
- [X] T016 [P] [US1] Entegrasyon testi: örnek kart, boş yüzey ve belirsiz nesne görüntülerinde çıkarım davranışı → `tests/integration/test_inference.py`

### Implementation for User Story 1

- [X] T017 [US1] Çıkarım sarmalayıcısını uygula: herhangi çözünürlükte RGB ön-işleme + letterbox, Core ML çalıştırma, letterbox'ı geri alıp orijinal kareye göre **normalize [0-1] xyxy** kutu → `src/electrack/inference/detector.py`
- [X] T018 [US1] İki-eşik mantığını (`det_threshold` altı → tespit yok/FP bastırma; `class_threshold` altı → `unknown`) `src/electrack/inference/detector.py` içinde uygula (research.md R5)
- [X] T019 [US1] Çıktı serileştirmesini (`bbox`/`class_name`/`class_id`/`confidence`) `detection-output` sözleşmesine göre `src/electrack/inference/detector.py` içinde uygula
- [X] T020 [US1] `infer` CLI alt komutunu (`--model --image --json`) `src/electrack/cli.py` içinde bağla
- [X] T021 [P] [US1] `validate-output` yardımcısını (jsonschema ile çıktı doğrulama) `src/electrack/inference/validate.py` içinde uygula
- [ ] T022 [P] [US1] US1 manuel doğrulaması için küçük örnek görüntü kümesi + negatifler (boş yüzey, belirsiz nesne) `datasets/raw/samples/` altında hazırla

**Checkpoint**: US1 tek başına çalışır — model bir kareden sözleşme-uyumlu tespit üretir, FP bastırır, `unknown` işaretler.

---

## Phase 4: User Story 2 - Modeli tekrarlanabilir biçimde üretme (Priority: P2)

**Goal**: Veriden başlayıp taşınabilir modeli ve kabul metriklerini (recall@IoU0.5, FP oranı, FPS, verdict) tekrar üretilebilir biçimde üreten uçtan-uca hattı ve değerlendirme/kabul sürecini sağlamak.

**Independent Test**: Aynı `data.yaml` + seed ile `prepare-data → train → export → evaluate` çalıştırıldığında `eval-report.schema.json`'a uyan, karşılaştırılabilir metrikli bir rapor üretilir (Senaryo 1 & 4, quickstart.md).

### Tests for User Story 2

- [X] T023 [P] [US2] Kontrat testi: değerlendirme raporunun `contracts/eval-report.schema.json`'a uyduğunu doğrula → `tests/contract/test_eval_report.py`
- [X] T024 [P] [US2] Entegrasyon testi: uçtan-uca tekrar üretilebilirlik (küçük küme → 1 epoch train → export → evaluate ×2, karşılaştır) → `tests/integration/test_pipeline_repro.py`

### Implementation for User Story 2

- [X] T025 [P] [US2] Metrikleri (mikro-ortalama recall@IoU0.5, FP oranı, per-class precision/recall/support) `src/electrack/eval/metrics.py` içinde uygula
- [X] T026 [US2] `unknown_behavior` metriğini (belirsiz kümede yüksek-güvenli yanlış sınıflandırma + unknown recall — SC-004) `src/electrack/eval/metrics.py` içinde uygula
- [X] T027 [P] [US2] Hedef donanımda gecikme/FPS ölçümünü (~50 komponent) `src/electrack/eval/latency.py` içinde uygula
- [X] T028 [US2] Kabul verdict + rapor yazıcısını (eşikler → PASS/FAIL, `eval-report` şeması) `src/electrack/eval/acceptance.py` içinde uygula
- [X] T029 [US2] `evaluate` CLI alt komutunu (`--model --dataset --measure-latency`) `src/electrack/cli.py` içinde bağla
- [X] T030 [US2] `prepare-data` CLI alt komutunu (convert + merge + deterministik split) `src/electrack/cli.py` içinde bağla
- [ ] T031 [P] [US2] Sabit kabul kümesini dondur (≥~20 kart / ~300+ etiketli görüntü, her türden yeterli örnek) + veri kartı → `datasets/acceptance/` + `datasets/acceptance/DATASET_CARD.md`
- [X] T032 [P] [US2] Veri kümesi/model sürüm manifestini (id/hash, tekrar üretilebilirlik) `src/electrack/data/manifest.py` içinde uygula

**Checkpoint**: US1 ve US2 bağımsız çalışır — hat tekrar üretilebilir ve kabul raporu verdict üretir.

---

## Phase 5: User Story 3 - Yeni komponent türü ekleme (Priority: P3)

**Goal**: Tür kümesini yalnızca `data.yaml` + veri + yeniden eğitim ile, hat/mimari yeniden yazılmadan genişletebilmek.

**Independent Test**: `data.yaml`'a yeni sınıf eklenip veri konularak yeniden eğitildiğinde model yeni türü tanır ve mevcut türlerin recall'ü kabul eşiğinin altına düşmez (Senaryo 5, quickstart.md).

### Tests for User Story 3

- [X] T033 [P] [US3] Regresyon testi: yeni sınıf eklendiğinde mevcut sınıf recall'ünün eşiğin üstünde kaldığını (parametrik sınıf listesi) doğrula → `tests/integration/test_add_class.py`

### Implementation for User Story 3

- [X] T034 [US3] Sınıf kayıt defteri/enum'unun tamamen `data.yaml`'dan türetildiğini garanti et (sabit-kodlu sınıf yok) → `src/electrack/data/classes.py`
- [X] T035 [US3] `training`/`export`/`eval` yollarını sınıf listesini yalnızca `data.yaml`'dan okuyacak şekilde denetle/refaktör et → `src/electrack/training/train.py`, `src/electrack/export/to_coreml.py`, `src/electrack/eval/metrics.py`
- [X] T036 [P] [US3] Yeni sınıf ekleme iş akışı belgesini (data.yaml düzenle → veri ekle → yeniden eğit → değerlendir) `docs/adding-a-class.md` içine yaz

**Checkpoint**: Tüm hikâyeler bağımsız çalışır; tür kümesi veri-odaklı genişletilebilir.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Birden çok hikâyeyi etkileyen iyileştirmeler

- [X] T037 [P] Eşik tarama yardımcısını (SC-001 recall + SC-002 FP'yi aynı anda sağlayan `det/class` eşiklerini seç) `src/electrack/eval/threshold_sweep.py` içinde uygula
- [X] T038 [P] Zorlu koşullar için augmentasyon ayarını (ışık/yansıma/ölçek/kapanma — FR-008) `src/electrack/training/config/mvp.yaml` içinde ayarla
- [ ] T039 [P] Her açık kaynak için veri kartlarını (lisans, kapsam, taksonomi eşlemesi) `datasets/raw/<source>/DATASET_CARD.md` içine yaz
- [X] T040 [P] Proje README + kullanım kılavuzunu `README.md` içine yaz
- [X] T041 [P] convert/splits/metrics için birim testleri `tests/unit/` altında yaz
- [ ] T042 quickstart.md doğrulama senaryolarını (1–5) uçtan uca çalıştır ve sonuçları not et

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Bağımlılık yok — hemen başlar
- **Foundational (Phase 2)**: Setup'a bağlı — TÜM hikâyeleri bloklar
- **User Stories (Phase 3+)**: Foundational tamamlanınca başlar
  - US1 → US2 → US3 öncelik sırası; personel varsa paralel
- **Polish (Phase 6)**: Hedeflenen hikâyeler bitince

### User Story Dependencies

- **US1 (P1)**: Foundational'dan sonra başlar; başka hikâyeye bağımlı değil (MVP)
- **US2 (P2)**: Foundational'dan sonra başlar; US1 ile aynı `detector`/model artefaktını kullanabilir ama değerlendirme yolu bağımsız test edilebilir
- **US3 (P3)**: Foundational'dan sonra başlar; US1/US2 hattını parametrik kullanır, bağımsız test edilebilir

### Within Each User Story

- Testler (dahil edilmişse) implementasyondan önce yazılır ve önce başarısız olur
- detector.py çekirdeği (T017) → eşik/serileştirme (T018–T019) → CLI bağlama (T020)
- metrics (T025) → acceptance/rapor (T028) → CLI (T029)

### Parallel Opportunities

- Setup: T003, T004, T005, T006 paralel
- Foundational: T008, T009, T010 paralel (T007 sonrası); T011/T012/T013 modül-bazlı ayrı dosyalar
- US1: T015, T016 (testler) paralel; T021, T022 paralel
- US2: T023, T024 paralel; T025/T027/T031/T032 paralel
- Polish: T037–T041 büyük ölçüde paralel

---

## Parallel Example: User Story 1

```bash
# US1 testlerini birlikte başlat:
Task: "Kontrat testi detection-output.schema.json — tests/contract/test_detection_output.py"
Task: "Entegrasyon testi çıkarım davranışı — tests/integration/test_inference.py"

# US1 yardımcı görevlerini birlikte başlat:
Task: "validate-output yardımcısı — src/electrack/inference/validate.py"
Task: "Örnek + negatif görüntü kümesi — datasets/raw/samples/"
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1: Setup
2. Phase 2: Foundational (KRİTİK — tüm hikâyeleri bloklar; ilk modeli üretir)
3. Phase 3: US1
4. **DUR ve DOĞRULA**: Örnek kart/boş yüzey/belirsiz nesnede çıkarım + sözleşme uyumu (quickstart Senaryo 2–3)
5. Hazırsa demo (canlı tespit)

### Incremental Delivery

1. Setup + Foundational → temel hazır
2. US1 → bağımsız test → MVP demo (çevrimdışı canlı tespit)
3. US2 → kabul değerlendirmesi + tekrar üretilebilirlik (SC-001/002/003 verdict)
4. US3 → yeni tür ekleme (SC-005)
5. Polish → eşik tarama, augmentasyon, dokümantasyon, quickstart doğrulama

### Parallel Team Strategy

Foundational bittikten sonra: Geliştirici A → US1 (çıkarım/sözleşme), Geliştirici B → US2 (değerlendirme/kabul), Geliştirici C → US3 (genişletilebilirlik). Stories bağımsız entegre olur.

---

## Notes

- [P] = farklı dosya, bağımlılık yok
- [Story] etiketi izlenebilirlik için görevleri hikâyeye bağlar
- Her hikâye bağımsız tamamlanabilir/test edilebilir
- Testlerin implementasyondan önce başarısız olduğunu doğrula
- Her görev veya mantıksal grup sonrası commit (repo git'e alınırsa)
- Kabul eşikleri: recall ≥ %80 (mikro @ IoU0.5), FP < %10, FPS ≥ 15 (M4 Mac Mini)
