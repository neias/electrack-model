"""Merkezi sabitler: eşikler, girdi boyutu, kabul ölçütleri.

Bu değerler hattın tek doğru kaynağıdır. Operasyonel eşikler (det/class)
`eval/threshold_sweep.py` ile SC-001 + SC-002'yi sağlayacak biçimde ayarlanabilir;
buradaki değerler makul başlangıç varsayılanlarıdır (research.md R4/R5).
"""

from __future__ import annotations

# --- Çıkarım eşikleri (iki-eşik mantığı, research.md R5) ---
# Genel tespit eşiği: altındaki adaylar elenir (hayali tespit yok — FR-007).
DEFAULT_DET_THRESHOLD: float = 0.25
# Sınıf kesinlik eşiği: det üstü ama bunun altı → "unknown" (FR-004).
DEFAULT_CLASS_THRESHOLD: float = 0.50

# --- Değerlendirme ---
IOU_MATCH_THRESHOLD: float = 0.50  # SC-001: eşleştirme IoU eşiği (sabit).

# --- Kabul ölçütleri (spec Success Criteria) ---
ACCEPT_RECALL_MIN: float = 0.80  # SC-001
ACCEPT_FP_RATE_MAX: float = 0.10  # SC-002
ACCEPT_FPS_MIN: float = 15.0  # SC-003

# --- Model girdisi ---
MODEL_INPUT_SIZE: int = 640  # Kare (letterbox) — research.md R1/R4.

# --- Etiketler ---
UNKNOWN_LABEL: str = "unknown"

# --- Tekrar üretilebilirlik ---
DEFAULT_SEED: int = 1337
