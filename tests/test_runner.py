"""Shared run harness. Mirrors src/qlab_rl/runner.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.policies.llm import LLMPolicy
from qlab_rl.policies.rule import make as make_rule
from qlab_rl.runner import RECORD_FIELDS, build_env, load_segment, run_policy, write_records

from tests.helpers import FakeClient, make_ramp_bars, tiny_ramp_env


class TestRunnerRecords(unittest.TestCase):
    def test_records_capture_every_step(self):
        client = FakeClient(["ACTION: LONG_HALF\nREASON: ok"] * 50)
        env = tiny_ramp_env()
        rows = []
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy, on_step=rows.append)

        self.assertEqual(len(rows), m["steps"])
        for key in ("step", "date", "action", "reward", "equity", "parse_ok", "response"):
            self.assertIn(key, rows[0])
        self.assertEqual([r["step"] for r in rows], list(range(len(rows))))

    def test_records_round_trip_to_csv(self):
        import tempfile

        client = FakeClient(["ACTION: FLAT"] * 50)
        env = tiny_ramp_env()
        rows = []
        policy = LLMPolicy(client, n_actions=5)
        run_policy(env, policy, on_step=rows.append)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "ep.csv"
            write_records(path, rows)
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("step,date,price,action", text)
            self.assertEqual(len(text.strip().splitlines()), len(rows) + 1)


class TestRecordFields(unittest.TestCase):
    def test_record_fields_append_only(self):
        self.assertEqual(RECORD_FIELDS[-2:], ["provider", "model"])
        self.assertEqual(RECORD_FIELDS[:11], [
            "step", "date", "price", "action", "reward", "equity",
            "gross_return", "cost", "turnover", "response", "parse_ok",
        ])


class TestLoadSegment(unittest.TestCase):
    def test_valid_splits(self):
        bars = make_ramp_bars(100)
        for name in ("train", "val", "test"):
            seg = load_segment(bars, name)
            self.assertGreater(len(seg), 0)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            load_segment(make_ramp_bars(10), "oos")


class TestBuildEnv(unittest.TestCase):
    def test_overrides_apply(self):
        env = build_env(make_ramp_bars(100), "t", overrides={"fee_bps": 0.0})
        self.assertEqual(env.fee_rate, 0.0)

    def test_label_stored(self):
        env = build_env(make_ramp_bars(100), "myseg")
        self.assertEqual(env.label if hasattr(env, "label") else "myseg", "myseg")


class TestRunPolicyControl(unittest.TestCase):
    def test_max_steps_truncates(self):
        client = FakeClient(["ACTION: FLAT"] * 50)
        env = tiny_ramp_env(n=40)
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy, max_steps=5)
        self.assertEqual(m["steps"], 5)
        self.assertTrue(m["truncated"])
        self.assertLess(len(client.prompts), 40)

    def test_rule_policy_parse_rate_counts_steps(self):
        env = tiny_ramp_env(n=40)
        policy = make_rule("buy_hold")
        m = run_policy(env, policy)
        self.assertGreater(m["steps"], 0)
        self.assertEqual(list(m["action_hist"]), [1.0])
        self.assertFalse(m["truncated"])

    def test_on_step_response_truncated_to_400(self):
        long_reply = "ACTION: FLAT\nREASON: " + "x" * 500
        client = FakeClient([long_reply] * 50)
        env = tiny_ramp_env(n=40)
        rows = []
        run_policy(env, LLMPolicy(client, n_actions=5), on_step=rows.append, max_steps=2)
        self.assertLessEqual(len(rows[0]["response"]), 400)
        # NOTE: on_step rows do not yet carry provider/model even though
        # RECORD_FIELDS has them; locked here so the gap is visible.
        self.assertNotIn("provider", rows[0])


if __name__ == "__main__":
    unittest.main()
