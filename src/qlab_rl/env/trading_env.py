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

import math
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
        self.trading_days_per_year = trading_days_per_year
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

    # -------------------------------------------------------------- metrics

    @property
    def total_return(self) -> float:
        return self.equity - 1.0

    @property
    def max_drawdown(self) -> float:
        peak = -float("inf")
        worst = 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            if peak > 0:
                worst = min(worst, eq / peak - 1.0)
        return worst

    @property
    def annualiser(self) -> float:
        return math.sqrt(self.trading_days_per_year)

    @property
    def sharpe(self) -> float:
        rets = self.net_returns
        if len(rets) < 2:
            return float("nan")
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        if sd == 0:
            return float("nan")
        return (mu / sd) * self.annualiser

    @property
    def cagr(self) -> float:
        n = len(self.net_returns)
        if n <= 0 or self.equity <= 0:
            return float("nan")
        years = n / self.trading_days_per_year
        if years <= 0:
            return float("nan")
        return (self.equity ** (1.0 / years)) - 1.0

    @property
    def profit_factor(self) -> float:
        gains = sum(r for r in self.net_returns if r > 0)
        losses = -sum(r for r in self.net_returns if r < 0)
        if losses == 0:
            return float("inf") if gains > 0 else float("nan")
        return gains / losses

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        if mdd == 0:
            return float("inf")
        return self.cagr / mdd

    def metrics(self) -> dict[str, Any]:
        flat = sum(1 for a in self.actions if a == 0.0)
        n = len(self.actions)
        return {
            "label": self.label,
            "n_bars": n,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "calmar": self.calmar,
            "mean_reward": (sum(self.rewards) / n) if n else float("nan"),
            "flat_ratio": (flat / n) if n else float("nan"),
            "mean_exposure": (sum(self.actions) / n) if n else float("nan"),
            "turnover": sum(
                abs(a - b) for a, b in zip([0.0] + self.actions, self.actions)
            ),
            "final_equity": self.equity,
        }

    def format_metrics(self) -> str:
        m = self.metrics()

        def pct(x: float) -> str:
            return "n/a" if x != x else f"{x * 100:6.2f}%"

        def num(x: float) -> str:
            return "n/a" if x != x else f"{x:6.2f}"

        return (
            f"{m['label']:<14} bars={m['n_bars']:>5}  "
            f"ret={pct(m['total_return'])}  sharpe={num(m['sharpe'])}  "
            f"cagr={pct(m['cagr'])}  maxDD={pct(m['max_drawdown'])}  "
            f"PF={num(m['profit_factor'])}  calmar={num(m['calmar'])}  "
            f"flat={pct(m['flat_ratio'])}  expo={num(m['mean_exposure'])}"
        )