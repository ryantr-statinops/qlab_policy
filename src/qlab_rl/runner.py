"""Shared run harness: build an env, drive a policy through it."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import load_env
from .data.provider import Bar
from .env.splits import split_bars
from .env.trading_env import TradingEnv

RECORD_FIELDS = [
    "step", "date", "price", "action", "reward", "equity",
    "gross_return", "cost", "turnover", "response", "parse_ok",
]


def build_env(bars: list[Bar], label: str, overrides: dict[str, Any] | None = None) -> TradingEnv:
    cfg = load_env()
    env_cfg = dict(cfg["env"])
    state_cfg = cfg["state"]
    env_cfg.update(overrides or {})
    return TradingEnv(
        bars,
        fee_bps=env_cfg["fee_bps"],
        lambda_risk=env_cfg["lambda_risk"],
        warmup=env_cfg.get("warmup"),
        liquidate_at_end=env_cfg.get("liquidate_at_end", True),
        trading_days_per_year=env_cfg.get("trading_days_per_year", 252),
        ret_windows=tuple(state_cfg["ret_windows"]),
        vol_window=state_cfg["vol_window"],
        drawdown_window=state_cfg["drawdown_window"],
        label=label,
    )


def load_segment(
    bars: list[Bar], split: str, train: float = 0.60, val: float = 0.20, test: float = 0.20
) -> list[Bar]:
    parts = split_bars(bars, train, val, test)
    if split not in parts:
        raise ValueError(f"split must be one of {sorted(parts)}, got {split!r}")
    return parts[split]