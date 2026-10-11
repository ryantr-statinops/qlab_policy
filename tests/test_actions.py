"""Action sets and parsing the model's reply back into an exposure."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.policies.actions import action_set, label_for, labels, nearest, value_of
from qlab_rl.policies.llm import parse_action


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


class TestParseAction(unittest.TestCase):
    def test_canonical_format(self):
        text = "ACTION: LONG_HALF\nREASON: Trend is up but volatility is high."
        self.assertEqual(parse_action(text, 5), (0.5, "LONG_HALF"))

    def test_lowercase_and_colonless(self):
        self.assertEqual(parse_action("action: short_all", 5)[0], -1.0)

    def test_action_with_dash_separator(self):
        self.assertEqual(parse_action("ACTION - FLAT", 5)[0], 0.0)

    def test_illegal_label_is_not_accepted(self):
        """BUY is a trading word but not one of the legal actions."""
        exposure, label = parse_action("ACTION: BUY\nREASON: whatever", 5)
        self.assertIsNone(exposure)
        self.assertEqual(label, "UNPARSED")

    def test_bare_label_in_prose_is_found(self):
        self.assertEqual(parse_action("I think LONG_ALL is too risky, FLAT is better.", 5)[0], 0.0)

    def test_prose_only_returns_none(self):
        exposure, label = parse_action("The market looks uncertain right now.", 5)
        self.assertIsNone(exposure)
        self.assertEqual(label, "UNPARSED")

    def test_empty_response_returns_none(self):
        self.assertEqual(parse_action("", 5)[0], None)

    def test_short_label_ignored_when_set_is_long_only(self):
        """SHORT_ALL is legal in the 5-action set but must not be picked in 3."""
        self.assertIsNone(parse_action("ACTION: SHORT_ALL", 3)[0])


if __name__ == "__main__":
    unittest.main()