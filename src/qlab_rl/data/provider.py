"""Daily OHLCV provider with a local CSV cache.

Source is Yahoo Finance's public chart endpoint (no API key). The Provider
interface is intentionally small so a vnstock-backed implementation can be
dropped in later without touching env or policy code.

stdlib only: csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

FIELDS = ["date", "open", "high", "low", "close", "volume"]


class Bar:
    """One daily bar."""

    __slots__ = ("date", "open", "high", "low", "close", "volume")

    def __init__(self, date: str, o: float, h: float, l: float, c: float, v: float):
        self.date = date
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Bar({self.date}, c={self.close})"


class DataProvider(Protocol):
    def load(self) -> list[Bar]: ...


def write_csv(bars: list[Bar], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for b in bars:
            writer.writerow([b.date, b.open, b.high, b.low, b.close, int(b.volume)])


def read_csv(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            bars.append(
                Bar(
                    row["date"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume") or 0),
                )
            )
    return bars