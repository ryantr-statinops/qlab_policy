"""Chronological 60/20/20 split by bar count."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.env.splits import split_bars, split_indices


class TestSplit(unittest.TestCase):
    def test_ratio_is_exact(self):
        idx = split_indices(1000)
        self.assertEqual(len(idx["train"]), 600)
        self.assertEqual(len(idx["val"]), 200)
        self.assertEqual(len(idx["test"]), 200)

    def test_splits_are_ordered_and_disjoint(self):
        idx = split_indices(1000)
        self.assertEqual(idx["train"].start, 0)
        self.assertEqual(idx["val"].start, idx["train"].stop)
        self.assertEqual(idx["test"].start, idx["val"].stop)
        self.assertEqual(idx["test"].stop, 1000)

    def test_bars_preserve_chronological_order(self):
        bars = list(range(1000))
        parts = split_bars(bars)
        self.assertEqual(parts["train"][0], 0)
        self.assertEqual(parts["test"][-1], 999)
        self.assertLess(parts["train"][-1], parts["val"][0])
        self.assertLess(parts["val"][-1], parts["test"][0])

    def test_tiny_series_does_not_produce_empty_splits(self):
        parts = split_bars(list(range(3)))
        self.assertTrue(all(len(v) >= 1 for v in parts.values()))

    def test_bad_fractions_rejected(self):
        with self.assertRaises(ValueError):
            split_indices(100, train=0.5, val=0.2, test=0.2)


if __name__ == "__main__":
    unittest.main()