"""electrack CLI — tek giriş noktası (T014).

Alt komutlar: prepare-data | train | export | evaluate | infer | validate-output
Ağır aşamalar (train/export/infer) bağımlılıkları LAZY yükler; çekirdek komutlar
(validate-output) bağımsız çalışır.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from electrack.logging_setup import get_logger

log = get_logger("electrack.cli")


def _cmd_prepare_data(args) -> int:
    from electrack.data.classes import ClassRegistry
    from electrack.data.datasets import validate_split

    registry = ClassRegistry.from_data_yaml(Path(args.data_yaml))
    log.info("Sınıflar (%d): %s", registry.num_classes, registry.names())
    yolo_root = Path(args.data_yaml).parent
    total_issues = 0
    for split in ("train", "val", "test"):
        img_dir = yolo_root / "images" / split
        lbl_dir = yolo_root / "labels" / split
        if not img_dir.exists():
            log.warning("Bölünme yok, atlanıyor: %s", img_dir)
            continue
        stats = validate_split(img_dir, lbl_dir, registry)
        log.info(
            "%s: %d görüntü, %d kutu, %d negatif, %d sorun",
            split,
            stats.num_images,
            stats.num_boxes,
            stats.num_negative_images,
            len(stats.issues),
        )
        total_issues += len(stats.issues)
        for issue in stats.issues[:20]:
            log.error("  %s:%d %s", issue.label_path, issue.line_no, issue.message)
    if total_issues:
        log.error("Toplam %d etiket sorunu. Düzeltin.", total_issues)
        return 1
    log.info("Veri doğrulama başarılı.")
    return 0


def _cmd_train(args) -> int:
    from electrack.training.train import train

    best = train(Path(args.config), epochs=args.epochs)
    print(str(best))
    return 0


def _cmd_export(args) -> int:
    from electrack.export.to_coreml import export_coreml

    out = export_coreml(Path(args.weights), det_threshold=args.det_threshold)
    print(str(out))
    return 0


def _cmd_infer(args) -> int:
    from electrack.config import ThresholdConfig
    from electrack.inference.detector import Detector

    thr = ThresholdConfig(det_threshold=args.det_threshold, class_threshold=args.class_threshold)
    det = Detector(Path(args.model), thresholds=thr)
    out = det.predict_path(Path(args.image))
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
        log.info("Yazıldı: %s", args.json)
    else:
        print(text)
    return 0


def _cmd_validate_output(args) -> int:
    from electrack.inference.validate import ContractError, validate_file

    try:
        validate_file(Path(args.file))
    except ContractError as e:
        log.error("Sözleşme ihlali: %s", e)
        return 1
    log.info("Çıktı sözleşmeye uygun: %s", args.file)
    return 0


def _cmd_evaluate(args) -> int:
    # Değerlendirme, çıkarım (Detector) gerektirir; ağır bağımlılık lazy.
    log.info(
        "Değerlendirme: model=%s dataset=%s (gerçek çalıştırma için model + "
        "bağımlılıklar gereklidir).",
        args.model,
        args.dataset,
    )
    log.info(
        "Ölçüm mantığı: electrack.eval.metrics + electrack.eval.acceptance. "
        "quickstart.md Senaryo 4'e bakın."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="electrack", description="PCB komponent tespiti hattı")
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("prepare-data", help="Veri kümesini doğrula/hazırla")
    pd.add_argument("--data-yaml", default="datasets/yolo/data.yaml")
    pd.set_defaults(func=_cmd_prepare_data)

    tr = sub.add_parser("train", help="Modeli eğit")
    tr.add_argument("--config", default="src/electrack/training/config/mvp.yaml")
    tr.add_argument("--epochs", type=int, default=None)
    tr.set_defaults(func=_cmd_train)

    ex = sub.add_parser("export", help="Core ML'e aktar")
    ex.add_argument("--weights", required=True)
    ex.add_argument("--det-threshold", type=float, default=0.25)
    ex.set_defaults(func=_cmd_export)

    inf = sub.add_parser("infer", help="Bir görüntüde çıkarım")
    inf.add_argument("--model", required=True)
    inf.add_argument("--image", required=True)
    inf.add_argument("--json", default=None)
    inf.add_argument("--det-threshold", type=float, default=0.25)
    inf.add_argument("--class-threshold", type=float, default=0.50)
    inf.set_defaults(func=_cmd_infer)

    vo = sub.add_parser("validate-output", help="Tespit çıktısını sözleşmeye göre doğrula")
    vo.add_argument("file")
    vo.set_defaults(func=_cmd_validate_output)

    ev = sub.add_parser("evaluate", help="Kabul değerlendirmesi")
    ev.add_argument("--model", required=True)
    ev.add_argument("--dataset", default="datasets/acceptance")
    ev.add_argument("--measure-latency", action="store_true")
    ev.set_defaults(func=_cmd_evaluate)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
