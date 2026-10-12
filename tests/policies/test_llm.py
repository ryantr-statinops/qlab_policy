"""LLM policy: prompt shape, reply parsing, policy plumbing.

Mirrors src/qlab_rl/policies/llm.py. Uses FakeClient from tests.helpers
so no LM Studio server is required.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.policies.actions import labels
from qlab_rl.policies.llm import LLMPolicy, build_user_message, parse_action
from qlab_rl.runner import run_policy

from tests.helpers import FakeClient, tiny_ramp_env

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


class TestLLMPolicyPlumbing(unittest.TestCase):
    def test_well_formed_replies_become_actions(self):
        client = FakeClient(["ACTION: LONG_ALL\nREASON: Momentum is up."] * 50)
        env = tiny_ramp_env()
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy)

        self.assertGreater(m["steps"], 0)
        self.assertEqual(m["parse_rate"], 1.0)
        self.assertEqual(list(m["action_hist"]), [1.0], "always LONG_ALL -> one action")
        self.assertEqual(len(client.prompts), m["steps"], "one LLM call per bar")

    def test_garbage_falls_back_to_flat_and_is_counted(self):
        client = FakeClient(["I am not sure what to do here."] * 50)
        env = tiny_ramp_env()
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy)

        self.assertEqual(m["parse_rate"], 0.0)
        self.assertEqual(list(m["action_hist"]), [0.0], "unparsable -> FLAT")

    def test_network_error_does_not_kill_the_episode(self):
        boom = RuntimeError("connection reset")
        client = FakeClient([boom] * 50)
        env = tiny_ramp_env()
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy)

        self.assertGreater(m["steps"], 0, "episode must survive server errors")
        self.assertEqual(list(m["action_hist"]), [0.0])
        self.assertTrue(policy.responses[0].startswith("<ERROR>"))

    def test_prompt_carries_the_real_state(self):
        client = FakeClient(["ACTION: FLAT"] * 50)
        env = tiny_ramp_env()
        policy = LLMPolicy(client, n_actions=5)
        run_policy(env, policy)

        first = client.prompts[0]
        self.assertEqual(first[0]["role"], "system")
        self.assertEqual(first[1]["role"], "user")
        user = first[1]["content"]
        self.assertIn("ACTION:", user)
        for label in ("SHORT_ALL", "SHORT_HALF", "FLAT", "LONG_HALF", "LONG_ALL"):
            self.assertIn(label, user)

    def test_action_set_size_changes_the_options_shown(self):
        client = FakeClient(["ACTION: FLAT"] * 50)
        env = tiny_ramp_env()
        policy = LLMPolicy(client, n_actions=3)
        run_policy(env, policy)
        user = client.prompts[0][1]["content"]
        self.assertIn("LONG_ALL", user)
        self.assertNotIn("SHORT_ALL", user)


if __name__ == "__main__":
    unittest.main()
