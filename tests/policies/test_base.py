"""PolicyContext read-only view. Mirrors src/qlab_rl/policies/base.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.env.trading_env import TradingEnv
from qlab_rl.policies.base import PolicyContext

from tests.helpers import SMALL_LOOKBACK, make_ramp_bars


class TestPromptContext(unittest.TestCase):
    def test_context_date_and_price_track_the_bar(self):
        bars = make_ramp_bars(40)
        env = TradingEnv(bars, **SMALL_LOOKBACK)
        env.reset()
        ctx = PolicyContext(env.builder, env.warmup)
        self.assertEqual(ctx.price, bars[env.warmup].close)
        ctx.index = env.warmup + 1
        self.assertEqual(ctx.price, bars[env.warmup + 1].close)

    def test_context_carries_episode_counters_and_builder(self):
        bars = make_ramp_bars(10)
        env = TradingEnv(bars, **SMALL_LOOKBACK)
        env.reset()
        ctx = PolicyContext(env.builder, env.warmup, episode=2, total_episodes=5)
        self.assertIs(ctx.builder, env.builder)
        self.assertEqual(ctx.episode, 2)
        self.assertEqual(ctx.total_episodes, 5)
        self.assertEqual(ctx.date, bars[env.warmup].date)


if __name__ == "__main__":
    unittest.main()
