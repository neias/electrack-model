# electrack

**[Türkçe](#türkçe) · [English](#english)**

PCB kamera karelerinde komponent tespiti — Apple Silicon (M4) üzerinde çevrimdışı, gerçek zamanlı nesne tespit modeli + tekrarlanabilir eğitim/değerlendirme hattı.
PCB component detection from camera frames — an offline, real-time object-detection model for Apple Silicon (M4) + a reproducible training/evaluation pipeline.

---

## Türkçe

PCB kamera karelerinde 6 komponent türünü (+ türetilen `unknown`) tespit edip sınıflandıran, **Apple Silicon M4 Mac Mini** üzerinde tamamen **çevrimdışı** ve gerçek zamanlı (≥15 FPS) çalışan bir nesne tespit modeli ve onu üreten **tekrarlanabilir eğitim/değerlendirme hattı**.

Spec Kit belgeleri: [`specs/001-pcb-component-detection/`](specs/001-pcb-component-detection/) (spec, plan, tasks, research, data-model, contracts, quickstart).

### Sınıflar (MVP)

`ic`, `electrolytic_capacitor`, `connector_header`, `led`, `through_hole_resistor`, `to92_transistor` + türetilen `unknown`. Tür kümesi `datasets/yolo/data.yaml`'dan genişletilir — bkz. [docs/adding-a-class.md](docs/adding-a-class.md).

### Kurulum

```sh
./run.sh setup            # venv oluşturur + bağımlılıkları kurar (.[train,export,dev])
```

veya elle:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # çekirdek + geliştirme (değerlendirme/çıkarım-sonrası)
pip install -e '.[train,export]'   # eğitim + Core ML dışa aktarımı (ağır)
```

> Not: `coremltools` Python 3.13'ü desteklemez; 3.9–3.12 kullanın (`PYBIN=python3.11 ./run.sh setup`). Çekirdek değerlendirme/geometri/sözleşme mantığı ağır bağımlılık olmadan çalışır ve test edilir; `train`/`export`/`infer` ultralytics/coremltools gerektirir.

### Hat (CLI)

```sh
./run.sh prepare-data                      # veri doğrula/hazırla → datasets/yolo/
./run.sh train                             # YOLO eğitimi → models/weights/
./run.sh export --weights models/weights/mvp/weights/best.pt
./run.sh infer --model models/export/electrack.mlpackage --image board.jpg
./run.sh evaluate --model models/export/electrack.mlpackage --dataset datasets/acceptance
./run.sh validate-output out.json
```

### Çıktı sözleşmesi

Her tespit: normalize [0-1] köşe kutu `[x_min,y_min,x_max,y_max]` + `class_name` + `class_id` (unknown için `null`) + `confidence`. Şema: [`contracts/detection-output.schema.json`](specs/001-pcb-component-detection/contracts/detection-output.schema.json). Model içeride NMS + eşik uygular ([`contracts/model-io.md`](specs/001-pcb-component-detection/contracts/model-io.md)).

### Kabul ölçütleri

- Recall ≥ %80 (mikro-ortalama @ IoU 0.5) — SC-001
- Yanlış-pozitif oranı < %10 — SC-002
- ≥ 15 FPS (~50 komponent, M4 Mac Mini) — SC-003

### Testler

```sh
./run.sh test              # tümü (ağır olanlar 'heavy' işaretli, deps/model yoksa atlanır)
./run.sh lint              # ruff + black --check
```

### Proje yapısı

```text
src/electrack/{data,training,export,inference,eval}/  # aşama modülleri + cli.py
datasets/{raw,yolo,acceptance}/                       # veri (git-ignore + kartlar)
models/{weights,export}/  reports/  tests/{contract,integration,unit}/
```

---

## English

An object-detection model that detects and classifies 6 component types (+ a derived `unknown`) from PCB camera frames, running fully **offline** and in real time (≥15 FPS) on an **Apple Silicon M4 Mac Mini**, together with a **reproducible training/evaluation pipeline** that produces it.

Spec Kit documents: [`specs/001-pcb-component-detection/`](specs/001-pcb-component-detection/) (spec, plan, tasks, research, data-model, contracts, quickstart).

### Classes (MVP)

`ic`, `electrolytic_capacitor`, `connector_header`, `led`, `through_hole_resistor`, `to92_transistor` + a derived `unknown`. The class set is extended from `datasets/yolo/data.yaml` — see [docs/adding-a-class.md](docs/adding-a-class.md).

### Setup

```sh
./run.sh setup            # creates venv + installs dependencies (.[train,export,dev])
```

or manually:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # core + dev (evaluation / post-processing)
pip install -e '.[train,export]'   # training + Core ML export (heavy)
```

> Note: `coremltools` does not support Python 3.13; use 3.9–3.12 (`PYBIN=python3.11 ./run.sh setup`). The core evaluation/geometry/contract logic runs and is tested without the heavy dependencies; `train`/`export`/`infer` require ultralytics/coremltools.

### Pipeline (CLI)

```sh
./run.sh prepare-data                      # validate/prepare data → datasets/yolo/
./run.sh train                             # YOLO training → models/weights/
./run.sh export --weights models/weights/mvp/weights/best.pt
./run.sh infer --model models/export/electrack.mlpackage --image board.jpg
./run.sh evaluate --model models/export/electrack.mlpackage --dataset datasets/acceptance
./run.sh validate-output out.json
```

### Output contract

Each detection: normalized [0-1] corner box `[x_min,y_min,x_max,y_max]` + `class_name` + `class_id` (`null` for unknown) + `confidence`. Schema: [`contracts/detection-output.schema.json`](specs/001-pcb-component-detection/contracts/detection-output.schema.json). The model applies NMS + thresholding internally ([`contracts/model-io.md`](specs/001-pcb-component-detection/contracts/model-io.md)).

### Acceptance criteria

- Recall ≥ 80% (micro-average @ IoU 0.5) — SC-001
- False-positive rate < 10% — SC-002
- ≥ 15 FPS (~50 components, M4 Mac Mini) — SC-003

### Tests

```sh
./run.sh test              # all (heavy tests are marked 'heavy', skipped without deps/model)
./run.sh lint              # ruff + black --check
```

### Project structure

```text
src/electrack/{data,training,export,inference,eval}/  # stage modules + cli.py
datasets/{raw,yolo,acceptance}/                       # data (git-ignored + cards)
models/{weights,export}/  reports/  tests/{contract,integration,unit}/
```

---

## Lisans / License

<!-- TODO: lisans ekleyin / add a license (e.g. MIT) -->
