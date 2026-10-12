"""Shared run harness. Mirrors src/qlab_rl/runner.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.policies.llm import LLMPolicy
from qlab_rl.runner import RECORD_FIELDS, run_policy, write_records

from tests.helpers import FakeClient, tiny_ramp_env


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


if __name__ == "__main__":
    unittest.main()
