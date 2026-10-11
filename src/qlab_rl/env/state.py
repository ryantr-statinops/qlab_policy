"""Observation builder.

At bar index i we observe using bars[0..i] only, then the chosen exposure earns
the return from bar i to bar i+1. No lookahead.

Observation layout, with the default lookbacks:

    0 return_1d     1 return_5d     2 return_20d
    3 vol_20d       annualised volatility over vol_window bars
    4 drawdown_20d  close / rolling_max - 1  (<= 0)
    5 exposure      exposure currently held, in [-1, 1]

The return block has len(ret_windows) entries, so the vector length is
len(ret_windows) + 3. `feature_names` mirrors the layout exactly.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..data.provider import Bar


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _pstdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


class StateBuilder:
    def __init__(
        self,
        bars: list[Bar],
        ret_windows: Sequence[int] = (1, 5, 20),
        vol_window: int = 20,
        drawdown_window: int = 20,
        trading_days_per_year: int = 252,
    ) -> None:
        self.bars = bars
        self.closes = [b.close for b in bars]
        self.ret_windows = list(ret_windows)
        self.vol_window = vol_window
        self.drawdown_window = drawdown_window
        self.annualiser = math.sqrt(trading_days_per_year)
        # smallest index at which every lookback is fully available
        self.warmup = max(self.ret_windows + [vol_window, drawdown_window])

    # ----------------------------------------------------------- components

    def _return(self, i: int, window: int) -> float | None:
        if i - window < 0:
            return None
        prev = self.closes[i - window]
        if prev <= 0:
            return None
        return self.closes[i] / prev - 1.0

    def _vol(self, i: int) -> float | None:
        """Annualised volatility of daily returns ending at bar i."""
        if i - self.vol_window + 1 < 1:
            return None
        rets = [
            self.closes[k] / self.closes[k - 1] - 1.0
            for k in range(i - self.vol_window + 1, i + 1)
        ]
        return _pstdev(rets) * self.annualiser

    def _drawdown(self, i: int) -> float | None:
        if i - self.drawdown_window + 1 < 0:
            return None
        window = self.closes[i - self.drawdown_window + 1 : i + 1]
        peak = max(window)
        if peak <= 0:
            return None
        return self.closes[i] / peak - 1.0

    # ------------------------------------------------------------- observe

    def observe(self, i: int, exposure: float = 0.0) -> list[float] | None:
        """Return the observation at bar i, or None if still warming up.

        Layout is `[*ret_windows, vol, drawdown, exposure]`, so its length is
        len(ret_windows) + 3 and it stays consistent with `feature_names`.
        """
        if i < self.warmup or i >= len(self.bars):
            return None
        rets = [self._return(i, w) for w in self.ret_windows]
        vol = self._vol(i)
        dd = self._drawdown(i)
        if any(r is None for r in rets) or vol is None or dd is None:
            return None
        return list(rets) + [vol, dd, exposure]

    @property
    def feature_names(self) -> list[str]:
        return (
            [f"return_{w}d" for w in self.ret_windows]
            + [f"vol_{self.vol_window}d", f"drawdown_{self.drawdown_window}d", "exposure"]
        )