"""Download daily OHLCV into data/raw/<TICKER>.csv.

    python scripts\fetch_data.py
    python scripts\fetch_data.py --ticker QQQ
    python scripts\fetch_data.py --refresh
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from qlab_rl.config import load_data, resolve
from qlab_rl.data.provider import YahooCsvProvider


def main() -> int:
    cfg = load_data()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["ticker"])
    ap.add_argument("--start", default=cfg["start"])
    ap.add_argument("--end", default=cfg.get("end") or None)
    ap.add_argument("--cache-dir", default=cfg["cache_dir"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    provider = YahooCsvProvider(
        args.ticker, args.start, args.end, resolve(args.cache_dir)
    )
    if provider.cache_path.exists() and not args.refresh:
        print(f"cache hit: {provider.cache_path}  (use --refresh to re-download)")
        bars = provider.load()
    else:
        print(f"downloading {args.ticker} from {args.start} -> today ...")
        try:
            bars = provider.load(refresh=True)
        except Exception as exc:
            print(f"[FAIL] {exc}")
            return 1
        print(f"written: {provider.cache_path}")

    if not bars:
        print("[FAIL] zero bars fetched")
        return 1

    print(f"\nbars : {len(bars)}")
    print(f"range: {bars[0].date} -> {bars[-1].date}")
    print(f"first: close={bars[0].close:.2f}")
    print(f"last : close={bars[-1].close:.2f}")
    bad = [b.date for b in bars if b.close <= 0 or b.high < b.low]
    print(f"sanity: {'OK' if not bad else f'{len(bad)} malformed bars'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())