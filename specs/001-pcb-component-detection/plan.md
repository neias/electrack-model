# Implementation Plan: PCB Komponent Tespiti (Cihaz-Üstü Nesne Tespit Modeli)

**Branch**: `001-pcb-component-detection` | **Date**: 2026-07-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-pcb-component-detection/spec.md`

## Summary

PCB kamera karelerinde 6 MVP komponent türünü (+ "bilinmeyen") tespit edip sınıflandıran, Apple Silicon M4 Mac Mini üzerinde tamamen çevrimdışı ve gerçek zamanlı (≥15 FPS, ~50 komponent) çalışan bir nesne tespit modeli ve bu modeli tekrarlanabilir biçimde üreten eğitim/değerlendirme hattı. Teknik yaklaşım: Ultralytics YOLO ailesi (nano/small ölçek) ile PyTorch üzerinde eğitim; Core ML (`.mlpackage`) formatına dışa aktarım ile Neural Engine/GPU hızlandırması; normalize [0-1] köşe kutu çıktısı, model-içi güven eşiği + NMS; recall@IoU0.5 (mikro-ortalama) ve yanlış-pozitif oranı ile kabul değerlendirmesi. Tür kümesi `data.yaml` üzerinden veri + yeniden eğitim ile genişletilebilir; hat yeniden yazılmaz.

## Technical Context

**Language/Version**: Python 3.11 (eğitim/değerlendirme/dışa-aktarım hattı)

**Primary Dependencies**: Ultralytics (YOLO11), PyTorch (MPS backend), coremltools (Core ML dışa aktarımı), OpenCV + Pillow (görüntü G/Ç), NumPy, PyYAML (veri kümesi konfig), pytest (hat testleri)

**Storage**: Dosya sistemi — veri kümeleri (`datasets/`, YOLO formatı görüntü+etiket), model artefaktları (`.pt` ağırlıkları, `.mlpackage` dışa aktarım), değerlendirme raporları (JSON/CSV). Veritabanı yok.

**Testing**: pytest (hat/veri şeması/kontrat testleri) + özel değerlendirme harness'i (metrik hesaplama: recall@IoU0.5, FP oranı)

**Target Platform**: macOS (Apple Silicon, M4 Mac Mini) — çıkarım Core ML runtime; eğitim yerel Apple Silicon (MPS) veya harici CUDA makinesi, ancak dağıtılan model Core ML `.mlpackage`

**Project Type**: ML modeli + tekrarlanabilir eğitim/değerlendirme hattı (tek proje, çok aşamalı CLI/script)

**Performance Goals**: Hedef donanımda ~50 komponentli karede ≥15 FPS (≤66 ms/kare) uçtan-uca çıkarım (ön-işleme + inference + son-işleme dahil)

**Constraints**: Tamamen çevrimdışı/cihaz-üstü (hiçbir görüntü/tespit ağa çıkmaz); çıktı normalize [0-1] eksen-hizalı köşe kutu (xyxy); model içeride ayarlanabilir güven eşiği + NMS uygular; girdi herhangi bir çözünürlükte RGB; tür kümesi genişletilebilir (hat yeniden yazılmadan)

**Scale/Scope**: 6 MVP sınıf + "bilinmeyen"; kare başına ~50 komponent; sabit kabul test kümesi ≥~20 kart / ~300+ etiketli görüntü; SMD vb. ile genişletilebilir

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` doldurulmamış bir şablondur (bağlayıcı ilke tanımlı değil). Bu nedenle projeye özgü zorunlu kapı (gate) yoktur. Yine de aşağıdaki genel mühendislik ilkeleri gözetilir ve ihlal edilmez:

- **Tekrar üretilebilirlik**: Tüm eğitim/değerlendirme adımları sabit tohum (seed), sürümlenmiş konfig ve deterministik veri bölünmeleriyle çalışır (FR-013, SC-006).
- **Çevrimdışılık**: Çıkarım hattı hiçbir dış servise bağımlı değildir (FR-005). Eğitim aşaması internet kullanabilir (veri/ağırlık indirme) ama dağıtılan artefakt bağımsızdır.
- **Genişletilebilirlik**: Tür kümesi veri odaklı (`data.yaml`); yeni sınıf kod/mimari değişikliği gerektirmez (FR-009).
- **Basitlik / YAGNI**: MVP için tek bir tespit modeli; ayrı sınıflandırıcı ikinci aşama eklenmez (gerekmedikçe).

**İlk değerlendirme**: PASS (ihlal yok). **Post-tasarım yeniden değerlendirme**: aşağıda Phase 1 sonrası tekrar PASS.

## Project Structure

### Documentation (this feature)

```text
specs/001-pcb-component-detection/
├── plan.md              # Bu dosya (/speckit-plan çıktısı)
├── research.md          # Phase 0 çıktısı
├── data-model.md        # Phase 1 çıktısı
├── quickstart.md        # Phase 1 çıktısı
├── contracts/           # Phase 1 çıktısı
│   ├── detection-output.schema.json   # Model çıktı sözleşmesi (tespit listesi)
│   ├── model-io.md                    # Model girdi/çıktı tensör sözleşmesi
│   └── eval-report.schema.json        # Değerlendirme raporu şeması
└── tasks.md             # Phase 2 çıktısı (/speckit-tasks — bu komut üretmez)
```

### Source Code (repository root)

```text
src/electrack/
├── data/
│   ├── datasets.py          # Veri kümesi keşfi, doğrulama, YOLO data.yaml üretimi
│   ├── convert.py           # Açık veri kümesi → YOLO formatı dönüştürücüler (COCO/VOC/Roboflow)
│   └── splits.py            # Deterministik train/val/test bölünmesi
├── labeling/
│   └── guidelines.md        # Etiketleme kuralları (sınıf tanımları, kutu kuralları, "bilinmeyen" politikası)
├── training/
│   ├── train.py             # YOLO eğitim giriş noktası (konfig odaklı)
│   └── config/
│       └── mvp.yaml         # Eğitim hiperparametreleri + augmentasyon
├── export/
│   └── to_coreml.py         # .pt → .mlpackage (NMS + eşik gömülü) dışa aktarım
├── inference/
│   └── detector.py          # Referans çıkarım sarmalayıcı (girdi ön-işleme, çıktı normalize kutu)
├── eval/
│   ├── metrics.py           # recall@IoU0.5 (mikro), FP oranı, per-class rapor
│   └── acceptance.py        # Kabul eşiklerine göre geç/kal + rapor üretimi
└── cli.py                   # Alt komutlar: prepare-data | train | export | evaluate

datasets/                    # Veri kümeleri (git-ignore; DVC/manifest ile izlenir)
├── raw/                     # İndirilen/toplanan ham veri
├── yolo/                    # YOLO formatında birleştirilmiş küme (images/ + labels/)
│   └── data.yaml            # Sınıf listesi + bölünme yolları (genişletme noktası)
└── acceptance/              # Sabit kabul test kümesi (~20 kart / ~300+ görüntü)

models/
├── weights/                 # Eğitim çıktısı .pt ağırlıkları
└── export/                  # Dağıtılabilir .mlpackage

reports/                     # Değerlendirme raporları (JSON/CSV, sürümlü)

tests/
├── contract/                # detection-output & eval-report şema uyumluluk testleri
├── integration/             # uçtan-uca: küçük küme → train (1 epoch) → export → evaluate
└── unit/                    # metrics, convert, splits birim testleri
```

**Structure Decision**: Tek proje (ML hattı). `src/electrack/` altında aşama-bazlı modüller (data → training → export → inference → eval) ve tek bir `cli.py` orkestrasyonu. Veri, model ve raporlar kod dışında sürümlenir (git-ignore + manifest). Bu düzen; genişletilebilirliği (`data.yaml`), tekrar üretilebilirliği (konfig + deterministik bölünme) ve çevrimdışı dağıtımı (`models/export/*.mlpackage`) doğrudan destekler.

## Complexity Tracking

> Constitution Check PASS — ihlal yok; bu tablo boş bırakılmıştır.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
