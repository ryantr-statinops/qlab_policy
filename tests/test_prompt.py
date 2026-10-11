"""Shape of the prompt handed to the model."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.policies.actions import labels
from qlab_rl.policies.llm import build_user_message, parse_action

OBS = [0.0125, 0.0340, 0.0712, 0.1520, -0.0180, 0.0]


class TestPrompt(unittest.TestCase):
    def test_contains_every_state_row(self):
        msg = build_user_message(OBS, "2021-06-15", 425.18, 5)
        for row in ("return_1d", "return_5d", "return_20d", "vol_20d", "drawdown_20d", "exposure"):
            self.assertIn(row, msg)

    def test_contains_date_price_and_all_options(self):
        msg = build_user_message(OBS, "2021-06-15", 425.18, 5)
        self.assertIn("2021-06-15", msg)
        self.assertIn("425.18", msg)
        for label in labels(5):
            self.assertIn(label, msg)

    def test_returns_are_shown_as_percentages(self):
        msg = build_user_message(OBS, "2021-06-15", 425.18, 5)
        self.assertIn("+1.25%", msg)
        self.assertIn("-1.80%", msg)

    def test_feature_names_can_be_overridden(self):
        msg = build_user_message(
            [0.01, 0.2, 0.0],
            "2021-06-15", 425.18, 5,
            names=["return_1d", "vol_20d", "exposure"],
        )
        self.assertIn("vol_20d", msg)
        self.assertNotIn("return_20d", msg)

    def test_name_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_user_message(OBS, "2021-06-15", 425.18, 5, names=["only_one"])

    def test_memories_are_injected(self):
        msg = build_user_message(
            OBS, "2021-06-15", 425.18, 5,
            memories=[{"state": "vol high", "action": "FLAT", "outcome": "sharpe 1.4"}],
        )
        self.assertIn("FLAT", msg)
        self.assertIn("sharpe 1.4", msg)

    def test_prompt_round_trips_through_the_parser(self):
        """The parser must read exactly what the prompt asks for."""
        msg = build_user_message(OBS, "2021-06-15", 425.18, 5)
        fake_reply = "ACTION: LONG_ALL\nREASON: Momentum is strong."
        self.assertEqual(parse_action(fake_reply, 5)[0], 1.0)
        self.assertIn("ACTION:", msg)


if __name__ == "__main__":
    unittest.main()