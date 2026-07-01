"""Kabul verdict + rapor yazıcısı (US2: FR-011, eval-report sözleşmesi).

Kabul eşikleri (config_constants):
- overall_recall >= 0.80 (SC-001)
- false_positive_rate < 0.10 (SC-002)
- latency fps >= 15 (SC-003)
"""

from __future__ import annotations

import json
from pathlib import Path

from electrack.config_constants import (
    ACCEPT_FP_RATE_MAX,
    ACCEPT_FPS_MIN,
    ACCEPT_RECALL_MIN,
    IOU_MATCH_THRESHOLD,
)
from electrack.eval.metrics import AggregateMetrics, per_class_report


def compute_verdict(
    overall_recall: float,
    false_positive_rate: float,
    fps: float | None,
) -> str:
    ok = overall_recall >= ACCEPT_RECALL_MIN and false_positive_rate < ACCEPT_FP_RATE_MAX
    if fps is not None:
        ok = ok and fps >= ACCEPT_FPS_MIN
    return "PASS" if ok else "FAIL"


def build_report(
    model_id: str,
    dataset_id: str,
    agg: AggregateMetrics,
    id_to_name,
    thresholds,
    latency: dict | None = None,
    unknown: dict | None = None,
) -> dict:
    fps = latency.get("fps") if latency else None
    report = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "iou_threshold": IOU_MATCH_THRESHOLD,
        "det_threshold": thresholds.det_threshold,
        "class_threshold": thresholds.class_threshold,
        "overall_recall": round(agg.overall_recall, 6),
        "false_positive_rate": round(agg.false_positive_rate, 6),
        "per_class": per_class_report(agg, id_to_name),
        "verdict": compute_verdict(agg.overall_recall, agg.false_positive_rate, fps),
    }
    if unknown is not None:
        report["unknown_behavior"] = unknown
    if latency is not None:
        report["latency"] = latency
    return report


def write_report(report: dict, out_dir: Path = Path("reports")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval-{report['model_id']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
