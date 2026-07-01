"""Veri kümesi/model sürüm manifesti (US2: SC-006, FR-013).

İçerik-tabanlı kararlı id/hash üretir — böylece rapordaki `model_id`/`dataset_id`
belirli bir artefaktı tekrar üretilebilir biçimde tanımlar.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_dir(root: Path, patterns: Iterable[str] = ("*.jpg", "*.png", "*.txt")) -> str:
    """Bir dizindeki içerikten kararlı bir hash (dosya adı + içerik)."""
    root = Path(root)
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    files = sorted(files, key=lambda p: str(p.relative_to(root)))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(hash_file(p).encode("utf-8"))
    return h.hexdigest()


def short_id(full_hash: str, n: int = 12) -> str:
    return full_hash[:n]


def dataset_id(root: Path) -> str:
    return "ds_" + short_id(hash_dir(root))


def model_id(weights_or_package: Path) -> str:
    p = Path(weights_or_package)
    digest = hash_dir(p) if p.is_dir() else hash_file(p)
    return "m_" + short_id(digest)
