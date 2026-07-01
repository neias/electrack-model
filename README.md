# electrack — PCB Komponent Tespiti

PCB kamera karelerinde 6 komponent türünü (+ `unknown`) tespit edip sınıflandıran, **Apple Silicon M4 Mac Mini** üzerinde tamamen **çevrimdışı** ve gerçek zamanlı (≥15 FPS) çalışan bir nesne tespit modeli ve onu üreten **tekrarlanabilir eğitim/değerlendirme hattı**.

Spec Kit belgeleri: [`specs/001-pcb-component-detection/`](specs/001-pcb-component-detection/) (spec, plan, tasks, research, data-model, contracts, quickstart).

## Sınıflar (MVP)

`ic`, `electrolytic_capacitor`, `connector_header`, `led`, `through_hole_resistor`, `to92_transistor` + türetilen `unknown`. Tür kümesi `datasets/yolo/data.yaml`'dan genişletilir — bkz. [docs/adding-a-class.md](docs/adding-a-class.md).

## Kurulum

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # çekirdek + geliştirme (değerlendirme/çıkarım-sonrası)
pip install -e '.[train,export]'   # eğitim + Core ML dışa aktarımı (ağır)
```

> Çekirdek değerlendirme/geometri/sözleşme mantığı ağır bağımlılık olmadan çalışır ve test edilir; `train`/`export`/`infer` ultralytics/coremltools gerektirir.

## Hat (CLI)

```sh
python -m electrack.cli prepare-data      # veri doğrula/hazırla → datasets/yolo/
python -m electrack.cli train             # YOLO eğitimi → models/weights/
python -m electrack.cli export --weights models/weights/mvp/weights/best.pt
python -m electrack.cli infer --model models/export/electrack.mlpackage --image board.jpg
python -m electrack.cli evaluate --model models/export/electrack.mlpackage --dataset datasets/acceptance
python -m electrack.cli validate-output out.json
```

## Çıktı sözleşmesi

Her tespit: normalize [0-1] köşe kutu `[x_min,y_min,x_max,y_max]` + `class_name` + `class_id` (unknown için `null`) + `confidence`. Şema: [`contracts/detection-output.schema.json`](specs/001-pcb-component-detection/contracts/detection-output.schema.json). Model içeride NMS + eşik uygular (`contracts/model-io.md`).

## Kabul ölçütleri

- Recall ≥ %80 (mikro @ IoU 0.5) — SC-001
- Yanlış-pozitif oranı < %10 — SC-002
- ≥ 15 FPS (~50 komponent, M4 Mac Mini) — SC-003

## Testler

```sh
pytest              # tümü (ağır olanlar 'heavy' işaretli, deps yoksa atlanır)
pytest -m "not heavy"
```

## Yapı

```text
src/electrack/{data,training,export,inference,eval}/  # aşama modülleri + cli.py
datasets/{raw,yolo,acceptance}/                       # veri (git-ignore + kartlar)
models/{weights,export}/  reports/  tests/{contract,integration,unit}/
```
