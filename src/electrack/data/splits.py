"""Deterministik train/val/test bölünmesi (FR-013, SC-006).

Bölünme, dosya adının kararlı bir hash'ine göre yapılır — böylece küme büyüse bile
mevcut atamalar değişmez ve çalıştırmalar arası tekrar üretilebilir olur (rastgele
karıştırma yerine hash-tabanlı deterministik atama).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _stable_fraction(key: str, seed: int) -> float:
    """key+seed'den [0,1) aralığında kararlı bir sayı üret (platformdan bağımsız)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    # İlk 8 hex hane → 32-bit tamsayı → [0,1)
    return int(h[:8], 16) / 0x100000000


def assign_split(
    key: str,
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> str:
    """Tek bir örneği (dosya adı/id) train/val/test'e deterministik ata."""
    if not (0 <= val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("val_frac + test_frac < 1 olmalı ve negatif olmamalı.")
    f = _stable_fraction(key, seed)
    if f < test_frac:
        return "test"
    if f < test_frac + val_frac:
        return "val"
    return "train"


def split_dataset(
    keys: Iterable[str],
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, list[str]]:
    """Anahtar kümesini deterministik bölünmelere ayır."""
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for k in keys:
        result[assign_split(k, seed, val_frac, test_frac)].append(k)
    for v in result.values():
        v.sort()
    return result


def split_counts(splits: dict[str, list[str]]) -> tuple[int, int, int]:
    return len(splits["train"]), len(splits["val"]), len(splits["test"])
