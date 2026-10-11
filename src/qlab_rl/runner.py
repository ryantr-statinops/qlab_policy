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


def run_policy(
    env: TradingEnv,
    policy,
    on_step=None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Drive `policy` through `env` until done (or `max_steps`).

    Returns env metrics plus `parse_rate`, `steps`, `action_hist`, `truncated`.
    """
    from .policies.base import PolicyContext

    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    ctx = PolicyContext(env.builder, env.warmup)
    step = 0
    parse_ok = 0
    action_hist: dict[float, int] = {}

    while not env.done and (max_steps is None or step < max_steps):
        ctx.index = env.i
        action = policy.act(obs, ctx)
        next_obs, reward, done, info = env.step(action)

        text = ""
        if hasattr(policy, "responses") and policy.responses:
            text = policy.responses[-1]
        # LLM policies report whether the action was actually understood, which
        # is what parse_rate must mean. Rule policies have no parse step.
        if hasattr(policy, "last_parse_ok"):
            ok = bool(policy.last_parse_ok)
        else:
            ok = bool(text) and not text.startswith("<ERROR>")
        parse_ok += int(ok)
        action_hist[action] = action_hist.get(action, 0) + 1

        if on_step:
            on_step(
                {
                    "step": step,
                    "date": info["date"],
                    "price": info["price"],
                    "action": action,
                    "reward": reward,
                    "equity": env.equity,
                    "gross_return": info["gross_return"],
                    "cost": info["cost"],
                    "turnover": info["turnover"],
                    "response": text.replace("\n", " ")[:400],
                    "parse_ok": int(ok),
                }
            )

        obs = next_obs
        step += 1

    metrics = env.metrics()
    metrics["parse_rate"] = (parse_ok / step) if step else float("nan")
    metrics["steps"] = step
    metrics["action_hist"] = action_hist
    metrics["truncated"] = bool(max_steps is not None and step >= max_steps)
    return metrics


def write_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)