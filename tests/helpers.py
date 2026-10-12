"""Shared fixtures for mirrored tests. No network, stdlib unittest only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.data.provider import Bar
from qlab_rl.env.trading_env import TradingEnv

SMALL_LOOKBACK = dict(ret_windows=(1,), vol_window=2, drawdown_window=2)
SMALL_LOOKBACK_WARMUP = dict(ret_windows=(1,), vol_window=2, drawdown_window=2, warmup=2)


def make_ramp_bars(n=40, step=1.001):
    """Deterministic ramp: close[i] = 100 * step^i."""
    bars, price = [], 100.0
    for i in range(n):
        bars.append(Bar(f"2020-01-{i % 28 + 1:02d}", price, price, price, price, 1000.0))
        price *= step
    return bars


def make_close_bars(closes):
    """Wrap a close series into minimal bars."""
    return [Bar(f"2020-01-{i + 1:02d}", c, c, c, c, 1000.0) for i, c in enumerate(closes)]


def flat_bars(n=80, price=100.0):
    return [Bar(f"2020-01-{(i % 28) + 1:02d}", price, price, price, price, 1.0) for i in range(n)]


def tiny_ramp_env(n=40, **kwargs):
    kw = dict(fee_bps=10.0, lambda_risk=0.0, **SMALL_LOOKBACK)
    kw.update(kwargs)
    return TradingEnv(make_ramp_bars(n), **kw)


def tiny_close_env(closes, **kwargs):
    kw = dict(fee_bps=10.0, lambda_risk=0.0, liquidate_at_end=False, **SMALL_LOOKBACK_WARMUP)
    kw.update(kwargs)
    return TradingEnv(make_close_bars(closes), **kw)


class FakeClient:
    """Stands in for LLMClient. Records prompts, replays canned text."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.models = [{"id": "fake/model"}]

    def resolve_model(self):
        return "fake/model"

    def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        if isinstance(text, Exception):
            raise text
        return {"choices": [{"message": {"content": text}}]}

    @staticmethod
    def text_of(response):
        return response["choices"][0]["message"]["content"]


BASE_CFG = {
    "base_url": "http://localhost:1234/v1",
    "model": "",
    "temperature": 0.0,
    "max_tokens": 160,
    "timeout_sec": 60,
    "retries": 2,
    "reasoning_effort": "none",
}

PRESETS = {
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "",
                 "api_key_env": "", "auto_detect": True, "timeout_sec": 60},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "qwen/qwen3-4b-instruct",
                   "api_key_env": "OPENROUTER_API_KEY",
                   "auto_detect": False, "timeout_sec": 90,
                   "extra_headers": {"X-Title": "qlab_rl"}},
}
