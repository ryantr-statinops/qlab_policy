"""Chronological 60/20/20 split by bar count.

Splitting by index (not date) keeps the ratio exact regardless of holidays.
Because bars are sorted by date, the split is still strictly chronological.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def split_indices(n: int, train: float = 0.60, val: float = 0.20, test: float = 0.20) -> dict[str, range]:
    total = train + val + test
    if not (0 < train < 1 and 0 < val < 1 and 0 < test < 1):
        raise ValueError("split fractions must be in (0, 1)")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    n_test = n - n_train - n_val

    return {
        "train": range(0, n_train),
        "val": range(n_train, n_train + n_val),
        "test": range(n_train + n_val, n_train + n_val + n_test),
    }


def split_bars(bars: Sequence[T], train: float = 0.60, val: float = 0.20, test: float = 0.20) -> dict[str, list[T]]:
    idx = split_indices(len(bars), train, val, test)
    return {k: [bars[i] for i in r] for k, r in idx.items()}