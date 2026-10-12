"""Non-LLM reference policies. Mirrors src/qlab_rl/policies/rule.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.env.trading_env import TradingEnv
from qlab_rl.policies.base import PolicyContext
from qlab_rl.policies.rule import ALL_NAMES, _sma, make

from tests.helpers import SMALL_LOOKBACK, make_ramp_bars


def ctx_at(prices, index):
    from qlab_rl.data.provider import Bar
    bars = [Bar(f"2020-01-{(i % 28) + 1:02d}", p, p, p, p, 1.0) for i, p in enumerate(prices)]
    env = TradingEnv(bars, **SMALL_LOOKBACK)
    env.reset()
    return PolicyContext(env.builder, index)


class TestMake(unittest.TestCase):
    def test_all_names_make_round_trip(self):
        for name in ALL_NAMES:
            p = make(name)
            self.assertEqual(p.name, name)

    def test_unknown_name_rejected(self):
        with self.assertRaises(ValueError):
            make("nope")

    def test_all_names_covers_six(self):
        self.assertEqual(len(ALL_NAMES), 6)
        self.assertIn("buy_hold", ALL_NAMES)


class TestFixedPolicies(unittest.TestCase):
    def test_buy_hold_always_long(self):
        p = make("buy_hold")
        c = ctx_at([100.0] * 10, 2)
        self.assertEqual(p.act([0.0], c), 1.0)

    def test_always_flat(self):
        p = make("always_flat")
        c = ctx_at([100.0] * 10, 2)
        self.assertEqual(p.act([0.0], c), 0.0)

    def test_fixed_half(self):
        p = make("fixed_half")
        c = ctx_at([100.0] * 10, 2)
        self.assertEqual(p.act([0.0], c), 0.5)


class TestRandom(unittest.TestCase):
    def test_random_is_deterministic_and_resets(self):
        p = make("random")
        c = ctx_at([100.0] * 10, 2)
        first = [p.act([0.0], c) for _ in range(10)]
        p.reset()
        second = [p.act([0.0], c) for _ in range(10)]
        self.assertEqual(first, second)
        for v in first:
            self.assertIn(v, [-1.0, -0.5, 0.0, 0.5, 1.0])


class TestSma(unittest.TestCase):
    def test_sma_helper_needs_n(self):
        self.assertIsNone(_sma([1.0, 2.0], 3))
        self.assertEqual(_sma([1.0, 2.0, 3.0], 3), 2.0)

    def test_sma_policies_flat_before_20_bars(self):
        for name in ("sma20_long_flat", "sma20_cross"):
            p = make(name)
            c = ctx_at([100.0] * 10, 2)
            for _ in range(5):
                self.assertEqual(p.act([0.0], c), 0.0)

    def test_long_flat_goes_long_above_sma(self):
        from qlab_rl.data.provider import Bar
        from qlab_rl.env.trading_env import TradingEnv
        prices = [100.0] * 20 + [200.0]
        bars = [Bar(f"2020-01-{(i % 28) + 1:02d}", p, p, p, p, 1.0) for i, p in enumerate(prices)]
        env = TradingEnv(bars, **SMALL_LOOKBACK)
        env.reset()
        p = make("sma20_long_flat")
        ctx = PolicyContext(env.builder, 0)
        out = None
        for i in range(len(prices)):
            ctx.index = i
            out = p.act([0.0], ctx)
        self.assertEqual(out, 1.0)

    def test_cross_goes_short_below_sma(self):
        from qlab_rl.data.provider import Bar
        from qlab_rl.env.trading_env import TradingEnv
        prices = [200.0] * 20 + [100.0]
        bars = [Bar(f"2020-01-{(i % 28) + 1:02d}", p, p, p, p, 1.0) for i, p in enumerate(prices)]
        env = TradingEnv(bars, **SMALL_LOOKBACK)
        env.reset()
        p = make("sma20_cross")
        ctx = PolicyContext(env.builder, 0)
        out = None
        for i in range(len(prices)):
            ctx.index = i
            out = p.act([0.0], ctx)
        self.assertEqual(out, -1.0)

    def test_ramp_bars_smoke(self):
        bars = make_ramp_bars(30)
        self.assertEqual(len(bars), 30)


if __name__ == "__main__":
    unittest.main()
