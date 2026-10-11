"""Single-asset trading environment.

Accounting convention (deliberately simple, matches common RL trading setups):

    At bar i you observe state from bars[0..i] and choose exposure e in [-1, 1].
    Fee is charged on the traded notional: cost = fee_rate * |e - e_prev|.
    That exposure earns the next bar's return: gross = e * (close[i+1]/close[i] - 1).
    net_t = gross - cost
    reward_t = net_t - lambda * vol_20(i) * |e|

Equity is multiplicative: equity *= (1 + net_t). Shorting is symmetric, so
negative returns while short are profits.
"""

from __future__ import annotations

from typing import Any

from ..data.provider import Bar
from .state import StateBuilder


class TradingEnv:
    def __init__(
        self,
        bars: list[Bar],
        fee_bps: float = 10.0,
        lambda_risk: float = 0.5,
        warmup: int | None = None,
        liquidate_at_end: bool = True,
        trading_days_per_year: int = 252,
        ret_windows: tuple[int, ...] = (1, 5, 20),
        vol_window: int = 20,
        drawdown_window: int = 20,
        label: str = "full",
    ) -> None:
        if len(bars) < 3:
            raise ValueError("need at least 3 bars")
        self.bars = bars
        self.fee_rate = fee_bps / 10_000.0
        self.lambda_risk = lambda_risk
        self.liquidate_at_end = liquidate_at_end
        self.label = label

        self.builder = StateBuilder(
            bars,
            ret_windows=ret_windows,
            vol_window=vol_window,
            drawdown_window=drawdown_window,
            trading_days_per_year=trading_days_per_year,
        )
        self.warmup = max(warmup or 0, self.builder.warmup)
        self.last_index = len(bars) - 1

        self.i = self.warmup
        self.equity = 1.0
        self.exposure = 0.0
        self.done = False
        self.equity_curve: list[float] = []
        self.net_returns: list[float] = []
        self.rewards: list[float] = []
        self.actions: list[float] = []

    # ----------------------------------------------------------------- core

    def reset(self) -> list[float]:
        self.i = self.warmup
        self.equity = 1.0
        self.exposure = 0.0
        self.done = False
        self.equity_curve = [1.0]
        self.net_returns = []
        self.rewards = []
        self.actions = []
        obs = self.observe()
        assert obs is not None, "warmup too small for state builder"
        return obs

    def observe(self) -> list[float] | None:
        return self.builder.observe(self.i, self.exposure)

    def step(self, exposure: float) -> tuple[list[float] | None, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode finished; call reset() first")
        if self.i + 1 > self.last_index:
            raise RuntimeError("step() called past the last bar")

        price = self.bars[self.i].close
        next_price = self.bars[self.i + 1].close
        vol = self.builder._vol(self.i) or 0.0

        turnover = abs(exposure - self.exposure)
        cost = self.fee_rate * turnover
        ret = next_price / price - 1.0 if price > 0 else 0.0
        gross = exposure * ret
        net = gross - cost

        self.equity *= 1.0 + net
        reward = net - self.lambda_risk * vol * abs(exposure)

        self.exposure = exposure
        self.net_returns.append(net)
        self.rewards.append(reward)
        self.actions.append(exposure)
        self.equity_curve.append(self.equity)
        self.i += 1

        info = {
            "date": self.bars[self.i].date,
            "price": self.bars[self.i].close,
            "gross_return": gross,
            "cost": cost,
            "turnover": turnover,
        }

        if self.i >= self.last_index:
            if self.liquidate_at_end and self.exposure != 0.0:
                self.equity *= 1.0 - self.fee_rate * abs(self.exposure)
                self.equity_curve[-1] = self.equity
                self.exposure = 0.0
                info["final_liquidation_fee"] = self.fee_rate * abs(self.actions[-1])
            self.done = True

        return self.observe(), reward, self.done, info