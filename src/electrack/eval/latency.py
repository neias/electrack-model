"""Hedef donanımda gecikme/FPS ölçümü (US2: SC-003, FR-012).

~50 komponentli kareler üzerinde uçtan-uca (ön-işleme + inference + NMS) süreyi
ölçer. `time.perf_counter` kullanır (Date/random yasağı bu ortamda değil; gerçek
donanımda çalışır). Ağır bağımlılık gerektiren kısım (gerçek model çağrısı)
`Detector` üzerinden gelir.
"""

from __future__ import annotations

import time
from pathlib import Path

from electrack.config_constants import ACCEPT_FPS_MIN


def measure_latency(detector, image_paths: list[Path], warmup: int = 3) -> dict:
    """Verilen görüntülerde ortalama FPS/ms ölç. Hedef: fps >= 15."""
    if not image_paths:
        raise ValueError("Ölçüm için en az bir görüntü gerekli.")

    # Isınma (ilk çağrılar model yüklemesi nedeniyle yavaştır).
    for p in image_paths[: min(warmup, len(image_paths))]:
        detector.predict_path(p)

    n = 0
    total = 0.0
    max_objects = 0
    for p in image_paths:
        t0 = time.perf_counter()
        out = detector.predict_path(p)
        total += time.perf_counter() - t0
        n += 1
        max_objects = max(max_objects, len(out.get("detections", [])))

    ms = (total / n) * 1000.0
    fps = n / total if total > 0 else 0.0
    return {
        "fps": round(fps, 2),
        "ms_per_frame": round(ms, 2),
        "objects_per_frame": max_objects,
        "hardware": "Apple M4 Mac Mini",
        "meets_target": fps >= ACCEPT_FPS_MIN,
    }
