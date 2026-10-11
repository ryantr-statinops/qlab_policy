"""State builder: no lookahead, correct windows, warmup respected."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.data.provider import Bar
from qlab_rl.env.state import StateBuilder


def make_bars(n=200, step=1.001):
    """Deterministic ramp: close[i] = 100 * step^i."""
    bars = []
    price = 100.0
    for i in range(n):
        bars.append(Bar(f"2020-01-{i % 28 + 1:02d}", price, price, price, price, 1000.0))
        price *= step
    return bars


class TestStateBuilder(unittest.TestCase):
    def setUp(self):
        self.bars = make_bars(100)
        self.b = StateBuilder(self.bars, ret_windows=(1, 5, 20), vol_window=20, drawdown_window=20)

    def test_warmup_is_max_lookback(self):
        self.assertEqual(self.b.warmup, 20)

    def test_none_before_warmup(self):
        for i in range(0, 20):
            self.assertIsNone(self.b.observe(i), f"bar {i} should not be observable")

    def test_observable_at_warmup(self):
        obs = self.b.observe(20)
        self.assertIsNotNone(obs)
        self.assertEqual(len(obs), 6)

    def test_return_1d_on_ramp(self):
        # close[i]/close[i-1] - 1 with price *= 1.001 per bar
        obs = self.b.observe(50, exposure=0.0)
        self.assertAlmostEqual(obs[0], 0.001, places=9)

    def test_return_5d_on_ramp(self):
        obs = self.b.observe(50, exposure=0.0)
        self.assertAlmostEqual(obs[1], 1.001 ** 5 - 1, places=9)

    def test_exposure_passthrough(self):
        obs = self.b.observe(50, exposure=-0.5)
        self.assertEqual(obs[5], -0.5)

    def test_drawdown_is_non_positive_at_peak(self):
        obs = self.b.observe(50, exposure=0.0)
        # a monotonically rising ramp sits at its rolling peak -> drawdown 0
        self.assertAlmostEqual(obs[4], 0.0, places=12)

    def test_drawdown_negative_after_drop(self):
        """Peak at i=39, then a steady 2%/bar decline: drawdown must turn negative."""
        bars = make_bars(60)
        price = bars[39].close
        for i in range(40, 60):
            price *= 0.98
            bars[i] = Bar(bars[i].date, price, price, price, price, 1000.0)
        b = StateBuilder(bars)
        obs = b.observe(59, exposure=0.0)
        self.assertIsNotNone(obs)
        # window [40..59] peaks at its first bar, so the peak is close[40]
        self.assertAlmostEqual(obs[4], 0.98 ** 19 - 1, places=9)

    def test_flat_series_has_zero_vol(self):
        bars = [Bar("2020-01-01", 100, 100, 100, 100, 1.0) for _ in range(60)]
        b = StateBuilder(bars)
        obs = b.observe(59, 0.0)
        self.assertIsNotNone(obs)
        self.assertEqual(obs[3], 0.0)
        self.assertEqual(obs[0], 0.0)

    def test_observation_uses_no_future_data(self):
        """Truncating the series must not change an earlier observation."""
        a = self.b.observe(50, 0.25)
        short_builder = StateBuilder(self.bars[:51])
        b = short_builder.observe(50, 0.25)
        self.assertEqual(a, b, "observation changed when future bars were removed")


if __name__ == "__main__":
    unittest.main()