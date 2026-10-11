"""Daily OHLCV provider with a local CSV cache.

Source is Yahoo Finance's public chart endpoint (no API key). The Provider
interface is intentionally small so a vnstock-backed implementation can be
dropped in later without touching env or policy code.

stdlib only: csv.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import urllib.request
from pathlib import Path
from typing import Protocol

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

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


# --------------------------------------------------------------------- fetch


def _epoch(date_str: str) -> int:
    return int(
        dt.datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )


def fetch_yahoo(ticker: str, start: str, end: str | None = None) -> list[Bar]:
    p1 = _epoch(start)
    p2 = _epoch(end) if end else int(dt.datetime.now(dt.timezone.utc).timestamp())
    url = CHART_URL.format(ticker=ticker) + f"?period1={p1}&period2={p2}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"yahoo error for {ticker}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"yahoo returned no data for {ticker}")

    res = results[0]
    stamps = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    o, h, l, c, v = (quote.get(k) or [] for k in ("open", "high", "low", "close", "volume"))

    bars: list[Bar] = []
    for i, ts in enumerate(stamps):
        close = c[i] if i < len(c) else None
        if close is None:
            continue  # holiday / incomplete bar
        date = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")
        bars.append(
            Bar(
                date,
                o[i] if i < len(o) and o[i] is not None else close,
                h[i] if i < len(h) and h[i] is not None else close,
                l[i] if i < len(l) and l[i] is not None else close,
                float(close),
                float(v[i]) if i < len(v) and v[i] is not None else 0.0,
            )
        )
    return bars


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