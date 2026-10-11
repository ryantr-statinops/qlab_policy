"""Non-LLM reference policies. These define the bar the LLM has to clear."""

from __future__ import annotations

import random
from typing import Sequence

from .base import PolicyContext


class RulePolicy:
    """Exposes the same `act` signature as LLMPolicy."""

    def __init__(self, name: str, fn, needs_state: bool = False) -> None:
        self.name = name
        self._fn = fn
        self.needs_state = needs_state
        self._sma: list[float] = []
        self._sma_n = 0
        self._rng = random.Random(42)

    def reset(self) -> None:
        self._sma = []
        self._rng.seed(42)

    def act(self, obs: Sequence[float], ctx: PolicyContext) -> float:
        return self._fn(obs, ctx, self)


# ----------------------------------------------------------------- helpers


def _sma(prices: list[float], n: int) -> float | None:
    if len(prices) < n:
        return None
    return sum(prices[-n:]) / n


# ---------------------------------------------------------------- policies


def buy_hold(obs, ctx, self) -> float:
    return 1.0


def always_flat(obs, ctx, self) -> float:
    return 0.0


def random_exposure(obs, ctx, self) -> float:
    choices = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return choices[self._rng.randrange(len(choices))]


def sma20_long_flat(obs, ctx, self) -> float:
    """Long when price is above its 20-bar SMA, otherwise flat."""
    price = ctx.price
    self._sma.append(price)
    ma = _sma(self._sma, 20)
    if ma is None:
        return 0.0
    return 1.0 if price > ma else 0.0


def sma20_cross(obs, ctx, self) -> float:
    """Long above SMA20, short below - the classic always-in trend follower."""
    price = ctx.price
    self._sma.append(price)
    ma = _sma(self._sma, 20)
    if ma is None:
        return 0.0
    return 1.0 if price > ma else -1.0


def fixed_half(obs, ctx, self) -> float:
    return 0.5


def make(name: str) -> RulePolicy:
    table = {
        "buy_hold": buy_hold,
        "always_flat": always_flat,
        "random": random_exposure,
        "sma20_long_flat": sma20_long_flat,
        "sma20_cross": sma20_cross,
        "fixed_half": fixed_half,
    }
    if name not in table:
        raise ValueError(f"unknown rule policy {name!r}; choose from {sorted(table)}")
    return RulePolicy(name, table[name], needs_state=True)


ALL_NAMES = ["buy_hold", "sma20_cross", "sma20_long_flat", "fixed_half", "always_flat", "random"]