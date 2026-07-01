"""Eşik tarama — SC-001 (recall) + SC-002 (FP) kısıtlarını birlikte sağlayan
`det_threshold`/`class_threshold` çalışma noktasını seçer (Polish: T037).

Saf Python: doğrulama kümesindeki ham tahminler (skorlarıyla) ve GT verildiğinde,
eşik ızgarasında recall/FP hesaplayıp kabul kısıtlarını karşılayan en iyi noktayı
döndürür.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from electrack.config import ThresholdConfig
from electrack.config_constants import ACCEPT_FP_RATE_MAX, ACCEPT_RECALL_MIN
from electrack.eval.metrics import ImageEval, evaluate


@dataclass
class SweepPoint:
    det_threshold: float
    class_threshold: float
    recall: float
    fp_rate: float
    feasible: bool


def _frange(start: float, stop: float, step: float) -> list[float]:
    out, v = [], start
    while v <= stop + 1e-9:
        out.append(round(v, 4))
        v += step
    return out


def sweep(
    build_images: Callable[[ThresholdConfig], list[ImageEval]],
    det_grid: list[float] | None = None,
    class_grid: list[float] | None = None,
) -> tuple[list[SweepPoint], SweepPoint | None]:
    """Eşik ızgarasını tara.

    `build_images(thresholds)`: verilen eşiklerle görüntü-bazlı tahminleri üreten
    fonksiyon (ham skorları eşiklere göre etiketler). Bu, çıkarım-sonrası mantığı
    (derive_label) uygulayan tarafça sağlanır.

    Döndürür: (tüm noktalar, en iyi uygun nokta). En iyi = kısıtları sağlayan,
    recall'i en yüksek (eşitlikte FP en düşük) nokta.
    """
    det_grid = det_grid or _frange(0.10, 0.50, 0.05)
    class_grid = class_grid or _frange(0.30, 0.80, 0.05)
    points: list[SweepPoint] = []
    for dt in det_grid:
        for ct in class_grid:
            if ct < dt:
                continue  # class_threshold >= det_threshold olmalı.
            images = build_images(ThresholdConfig(det_threshold=dt, class_threshold=ct))
            agg = evaluate(images)
            recall = agg.overall_recall
            fp = agg.false_positive_rate
            feasible = recall >= ACCEPT_RECALL_MIN and fp < ACCEPT_FP_RATE_MAX
            points.append(SweepPoint(dt, ct, round(recall, 6), round(fp, 6), feasible))

    feasible_points = [p for p in points if p.feasible]
    best = None
    if feasible_points:
        best = max(feasible_points, key=lambda p: (p.recall, -p.fp_rate))
    return points, best
