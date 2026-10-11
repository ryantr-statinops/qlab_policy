"""LLM policy path, driven by a fake client.

LM Studio is not available in CI, so LLMClient is replaced by a stub that
returns canned replies. This proves the plumbing - prompt assembly, parsing,
fallback on garbage, per-step records, action histogram - works end to end
before a real model is ever loaded.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.data.provider import Bar
from qlab_rl.env.trading_env import TradingEnv
from qlab_rl.policies.base import PolicyContext
from qlab_rl.policies.llm import LLMPolicy
from qlab_rl.runner import run_policy, write_records


class FakeClient:
    """Stands in for LLMClient. Records prompts, replays canned text."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.models = [{"id": "fake/model"}]

    def resolve_model(self):
        return "fake/model"

    def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        if isinstance(text, Exception):
            raise text
        return {"choices": [{"message": {"content": text}}]}

    @staticmethod
    def text_of(response):
        return response["choices"][0]["message"]["content"]


def make_bars(n=40):
    bars, price = [], 100.0
    for i in range(n):
        bars.append(Bar(f"2020-01-{i + 1:02d}", price, price, price, price, 1000.0))
        price *= 1.001
    return bars


SMALL_LOOKBACK = dict(ret_windows=(1,), vol_window=2, drawdown_window=2)


def tiny_env():
    return TradingEnv(make_bars(40), fee_bps=10.0, lambda_risk=0.0, **SMALL_LOOKBACK)


class TestLLMPolicyPlumbing(unittest.TestCase):
    def test_well_formed_replies_become_actions(self):
        client = FakeClient(["ACTION: LONG_ALL\nREASON: Momentum is up."] * 50)
        env = tiny_env()
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy)

        self.assertGreater(m["steps"], 0)
        self.assertEqual(m["parse_rate"], 1.0)
        self.assertEqual(list(m["action_hist"]), [1.0], "always LONG_ALL -> one action")
        self.assertEqual(len(client.prompts), m["steps"], "one LLM call per bar")

    def test_garbage_falls_back_to_flat_and_is_counted(self):
        client = FakeClient(["I am not sure what to do here."] * 50)
        env = tiny_env()
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy)

        self.assertEqual(m["parse_rate"], 0.0)
        self.assertEqual(list(m["action_hist"]), [0.0], "unparsable -> FLAT")

    def test_network_error_does_not_kill_the_episode(self):
        boom = RuntimeError("connection reset")
        client = FakeClient([boom] * 50)
        env = tiny_env()
        policy = LLMPolicy(client, n_actions=5)
        m = run_policy(env, policy)

        self.assertGreater(m["steps"], 0, "episode must survive server errors")
        self.assertEqual(list(m["action_hist"]), [0.0])
        self.assertTrue(policy.responses[0].startswith("<ERROR>"))

    def test_prompt_carries_the_real_state(self):
        client = FakeClient(["ACTION: FLAT"] * 50)
        env = tiny_env()
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
        env = tiny_env()
        policy = LLMPolicy(client, n_actions=3)
        run_policy(env, policy)
        user = client.prompts[0][1]["content"]
        self.assertIn("LONG_ALL", user)
        self.assertNotIn("SHORT_ALL", user)

    def test_records_capture_every_step(self):
        client = FakeClient(["ACTION: LONG_HALF\nREASON: ok"] * 50)
        env = tiny_env()
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
        env = tiny_env()
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


class TestPromptContext(unittest.TestCase):
    def test_context_date_and_price_track_the_bar(self):
        bars = make_bars(40)
        env = TradingEnv(bars, **SMALL_LOOKBACK)
        env.reset()
        ctx = PolicyContext(env.builder, env.warmup)
        self.assertEqual(ctx.price, bars[env.warmup].close)
        ctx.index = env.warmup + 1
        self.assertEqual(ctx.price, bars[env.warmup + 1].close)


if __name__ == "__main__":
    unittest.main()