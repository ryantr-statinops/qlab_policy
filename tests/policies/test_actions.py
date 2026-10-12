"""Action sets for discrete exposures. Mirrors src/qlab_rl/policies/actions.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.policies.actions import action_set, label_for, labels, nearest, value_of


class TestActionSets(unittest.TestCase):
    def test_five_actions_cover_minus_one_to_one(self):
        vals = [v for _, v in action_set(5)]
        self.assertEqual(vals, [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_three_actions_are_long_only(self):
        vals = [v for _, v in action_set(3)]
        self.assertEqual(vals, [0.0, 0.5, 1.0])

    def test_unsupported_size_rejected(self):
        with self.assertRaises(ValueError):
            action_set(7)

    def test_nearest_snaps_to_legal_value(self):
        self.assertEqual(nearest(5, 0.33), 0.5)
        self.assertEqual(nearest(5, -0.9), -1.0)
        self.assertEqual(nearest(3, -0.7), 0.0)

    def test_value_of_and_label_for_round_trip(self):
        for name, value in action_set(5):
            self.assertEqual(value_of(5, name), value)
            self.assertEqual(label_for(5, value), name)


if __name__ == "__main__":
    unittest.main()
