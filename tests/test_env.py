"""Environment accounting, verified against hand-computed numbers.

The golden test is deliberate: every figure below was calculated by hand from

    cost  = fee_rate * |e - e_prev|
    net_t = e * (close[i+1]/close[i] - 1) - cost
    equity *= (1 + net_t)

If this test fails, nothing downstream means anything.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.data.provider import Bar
from qlab_rl.env.trading_env import TradingEnv


def make_bars(closes):
    """Wrap a close series into minimal bars."""
    return [
        Bar(f"2020-01-{i + 1:02d}", c, c, c, c, 1000.0)
        for i, c in enumerate(closes)
    ]


# Golden tests need a state builder that fits inside a 4-bar series, so the
# lookbacks are shrunk to their minimum. That does not change the accounting.
SMALL_LOOKBACK = dict(ret_windows=(1,), vol_window=2, drawdown_window=2, warmup=2)


def tiny_env(closes, fee_bps=10.0, lambda_risk=0.0, liquidate_at_end=False):
    return TradingEnv(
        make_bars(closes), fee_bps=fee_bps, lambda_risk=lambda_risk,
        liquidate_at_end=liquidate_at_end, **SMALL_LOOKBACK,
    )


def flat_bars(n=80, price=100.0):
    return [Bar(f"2020-01-{(i % 28) + 1:02d}", price, price, price, price, 1.0) for i in range(n)]


class TestGoldenAccounting(unittest.TestCase):
    def test_hand_computed_single_step(self):
        """price 100 -> 101, go long full at 10bps.

        turnover = |1.0 - 0| = 1.0  -> cost = 0.001 * 1.0 = 0.001
        ret      = 101/100 - 1      = 0.01
        gross    = 1.0 * 0.01       = 0.01
        net      = 0.01 - 0.001     = 0.009
        equity   = 1 * 1.009
        """
        env = tiny_env([100.0, 100.0, 100.0, 101.0])
        env.reset()
        obs, reward, done, info = env.step(1.0)
        self.assertAlmostEqual(info["turnover"], 1.0, places=12)
        self.assertAlmostEqual(info["cost"], 0.001, places=12)
        self.assertAlmostEqual(info["gross_return"], 0.01, places=12)
        self.assertAlmostEqual(reward, 0.009, places=12)
        self.assertAlmostEqual(env.equity, 1.009, places=12)

    def test_second_step_pays_no_turnover_cost(self):
        """Holding 1.0 across the next bar costs nothing further."""
        env = tiny_env([100.0, 100.0, 100.0, 101.0, 102.0])
        env.reset()
        env.step(1.0)
        obs, reward, done, info = env.step(1.0)
        self.assertAlmostEqual(info["turnover"], 0.0, places=12)
        self.assertAlmostEqual(info["cost"], 0.0, places=12)
        self.assertAlmostEqual(info["gross_return"], 102.0 / 101.0 - 1.0, places=12)
        expected = 1.009 * (1.0 + 102.0 / 101.0 - 1.0)
        self.assertAlmostEqual(env.equity, expected, places=12)
        self.assertTrue(done)

    def test_short_side_is_symmetric(self):
        """Shorting a 1% fall earns the same gross as longing a 1% rise."""
        env = tiny_env([100.0, 100.0, 100.0, 99.0])
        env.reset()
        obs, reward, done, info = env.step(-1.0)
        self.assertAlmostEqual(info["gross_return"], -1.0 * (99.0 / 100.0 - 1.0), places=12)
        self.assertAlmostEqual(info["gross_return"], 0.01, places=12)
        self.assertAlmostEqual(env.equity, 1.009, places=12)

    def test_half_turnover_halves_the_fee(self):
        env = tiny_env([100.0, 100.0, 100.0, 100.0])
        env.reset()
        obs, reward, done, info = env.step(0.5)
        self.assertAlmostEqual(info["turnover"], 0.5, places=12)
        self.assertAlmostEqual(info["cost"], 0.0005, places=12)
        # price is unchanged, so the whole move is the fee
        self.assertAlmostEqual(env.equity, 0.9995, places=12)

    def test_reducing_exposure_also_pays_a_fee(self):
        """Going from +1.0 to 0.0 is turnover 1.0, not free."""
        env = tiny_env([100.0, 100.0, 100.0, 100.0, 100.0])
        env.reset()
        env.step(1.0)
        obs, reward, done, info = env.step(0.0)
        self.assertAlmostEqual(info["turnover"], 1.0, places=12)
        self.assertAlmostEqual(info["cost"], 0.001, places=12)

    def test_final_liquidation_charges_the_remaining_exposure(self):
        env = tiny_env([100.0, 100.0, 100.0, 101.0], liquidate_at_end=True)
        env.reset()
        env.step(1.0)  # lands on the last index, force-liquidates
        self.assertEqual(env.exposure, 0.0)
        # entry fee 0.001, then the exit fee 0.001 on an unchanged price
        self.assertAlmostEqual(env.equity, 1.009 * 0.999, places=12)


class TestRiskTerm(unittest.TestCase):
    """reward = net - lambda * vol * |e|, and lambda must not touch equity."""

    def _rollout(self, closes, lam):
        env = tiny_env(closes, fee_bps=0.0, lambda_risk=lam)
        obs = env.reset()
        rewards, nets = [], []
        while not env.done:
            before = len(env.net_returns)
            obs, reward, done, info = env.step(1.0)
            rewards.append(reward)
            nets.append(env.net_returns[before])
        return rewards, nets, list(env.equity_curve), env

    def _volley_series(self):
        # alternating +/-2% closes give a large, easy-to-check volatility
        return [100.0 * (1.0 + (0.02 if i % 2 else -0.02)) for i in range(60)]

    def test_lambda_zero_means_reward_equals_net_return(self):
        rewards, nets, _, _ = self._rollout(self._volley_series(), 0.0)
        for r, n in zip(rewards, nets):
            self.assertAlmostEqual(r, n, places=12)

    def test_lambda_only_shifts_reward_by_the_risk_term(self):
        closes = self._volley_series()
        r0, n0, eq0, env0 = self._rollout(closes, 0.0)
        r1, n1, eq1, _ = self._rollout(closes, 1.0)
        self.assertEqual(n0, n1, "lambda must not change net returns")
        self.assertEqual(eq0, eq1, "lambda must not change the equity path")
        self.assertNotEqual(r0, r1, "lambda must change the reward")

        # the gap must equal lambda * vol_i * |e| at each bar
        for k, (a, b) in enumerate(zip(r0, r1)):
            bar = env0.warmup + k
            vol = env0.builder._vol(bar)
            self.assertAlmostEqual(a - b, vol, places=12)

    def test_volatility_is_annualised(self):
        """Realised vol over the window, scaled by sqrt(trading_days_per_year).

        The window std is the population std, matching StateBuilder._pstdev.
        """
        import math

        env0 = tiny_env(self._volley_series(), fee_bps=0.0, lambda_risk=0.0)
        env0.reset()
        i = env0.warmup
        rets = [
            env0.bars[k].close / env0.bars[k - 1].close - 1.0
            for k in range(i - env0.builder.vol_window + 1, i + 1)
        ]
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / len(rets))
        self.assertAlmostEqual(env0.builder._vol(i), sd * math.sqrt(252), places=12)
        self.assertGreater(env0.builder._vol(i), 0.5, "alternating +/-2% must be volatile")


class TestLifecycle(unittest.TestCase):
    def test_reset_restores_starting_state(self):
        env = TradingEnv(flat_bars(80, 100.0), fee_bps=10.0)
        obs = env.reset()
        self.assertEqual(env.equity, 1.0)
        self.assertEqual(env.exposure, 0.0)
        self.assertEqual(obs[5], 0.0)
        while not env.done:
            env.step(0.0)
        env.reset()
        self.assertEqual(env.equity, 1.0)
        self.assertEqual(len(env.equity_curve), 1)

    def test_stepping_past_the_end_raises(self):
        env = TradingEnv(flat_bars(80, 100.0), fee_bps=10.0)
        env.reset()
        while not env.done:
            env.step(0.0)
        with self.assertRaises(RuntimeError):
            env.step(0.0)

    def test_length_of_equity_curve(self):
        n = 80
        env = TradingEnv(flat_bars(n, 100.0), fee_bps=10.0)
        env.reset()
        while not env.done:
            env.step(0.0)
        # 1 initial point + (last_index - warmup) steps
        self.assertEqual(len(env.equity_curve), 1 + (env.last_index - env.warmup))

    def test_metrics_on_a_flat_losing_series(self):
        env = TradingEnv(
            [Bar(f"2020-01-{(i % 28) + 1:02d}", 100 - i, 100 - i, 100 - i, 100 - i, 1.0)
             for i in range(80)],
            fee_bps=10.0, lambda_risk=0.0,
        )
        env.reset()
        while not env.done:
            env.step(1.0)  # long a falling market
        m = env.metrics()
        self.assertLess(m["total_return"], 0)
        self.assertLess(m["max_drawdown"], 0)

    def test_buy_hold_on_rising_series_is_profitable(self):
        bars = [Bar(f"2020-01-{(i % 28) + 1:02d}", 100 + i, 100 + i, 100 + i, 100 + i, 1.0)
                for i in range(80)]
        env = TradingEnv(bars, fee_bps=10.0, lambda_risk=0.0)
        env.reset()
        while not env.done:
            env.step(1.0)
        self.assertGreater(env.total_return, 0.2)


if __name__ == "__main__":
    unittest.main()