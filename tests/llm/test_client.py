"""LLM client. Mirrors src/qlab_rl/llm/client.py. No network: urlopen stubbed."""

import io
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.llm.client import LLMClient, LLMError


class AuthTest(unittest.TestCase):
    def test_local_sends_no_auth(self):
        c = LLMClient()
        self.assertNotIn("Authorization", c._headers(True))

    def test_cloud_sends_bearer_and_extra(self):
        c = LLMClient(api_key="sk-x", extra_headers={"X-Title": "t"})
        h = c._headers(True)
        self.assertEqual(h["Authorization"], "Bearer sk-x")
        self.assertEqual(h["X-Title"], "t")

    def test_errors_never_echo_key(self):
        c = LLMClient(provider="openrouter", api_key="sk-secret",
                      api_key_env="OPENROUTER_API_KEY", model="x/y")
        err = c._http_error(401, "https://x", "detail sk-secret")
        self.assertIn("$OPENROUTER_API_KEY", str(err))
        self.assertNotIn("sk-secret", str(err))


class RetryTest(unittest.TestCase):
    def test_429_then_success(self):
        calls = []

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 429, "slow down", {}, io.BytesIO(b"busy"))
            return Resp()

        orig = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            c = LLMClient(model="m", retries=2)
            out = c.chat([{"role": "user", "content": "hi"}])
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(len(calls), 2)
        self.assertEqual(LLMClient.text_of(out), "hi")

    def test_auto_detect_off_requires_model(self):
        c = LLMClient(provider="openrouter", auto_detect=False)
        with self.assertRaises(LLMError) as ctx:
            c.resolve_model()
        self.assertIn("explicit model", str(ctx.exception))

    def test_404_names_model_and_provider(self):
        c = LLMClient(provider="openrouter", model="bad/slug")
        err = c._http_error(404, "https://x", "nope")
        self.assertIn("bad/slug", str(err))
        self.assertIn("openrouter", str(err))


class ContentTest(unittest.TestCase):
    def test_text_of_joins_blocks(self):
        resp = {"choices": [{"message": {"content": [
            {"type": "text", "text": "ACTION: "},
            {"type": "text", "text": "LONG_HALF"},
        ]}}]}
        self.assertEqual(LLMClient.text_of(resp), "ACTION: LONG_HALF")

    def test_text_of_plain_string(self):
        resp = {"choices": [{"message": {"content": "hi"}}]}
        self.assertEqual(LLMClient.text_of(resp), "hi")

    def test_text_of_no_choices_raises(self):
        with self.assertRaises(LLMError):
            LLMClient.text_of({"choices": []})

    def test_reasoning_of(self):
        self.assertEqual(LLMClient.reasoning_of({"choices": []}), "")
        resp = {"choices": [{"message": {"reasoning_content": "trace"}}]}
        self.assertEqual(LLMClient.reasoning_of(resp), "trace")


class TestBuilders(unittest.TestCase):
    def test_from_config(self):
        cfg = {"base_url": "http://x/v1", "model": "m", "temperature": 0.1,
               "max_tokens": 10, "timeout_sec": 5, "retries": 1, "reasoning_effort": "none"}
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.base_url, "http://x/v1")
        self.assertEqual(c.timeout, 5)

    def test_from_provider(self):
        from tests.helpers import BASE_CFG, PRESETS
        import os
        os.environ["OPENROUTER_API_KEY"] = "dummy"
        try:
            from qlab_rl.config import resolve_provider
            cfg = resolve_provider("openrouter", BASE_CFG, PRESETS, {})
            c = LLMClient.from_provider(cfg)
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)
        self.assertEqual(c.provider, "openrouter")
        self.assertIn("openrouter.ai", c.base_url)


class TestErrors(unittest.TestCase):
    def test_402_mentions_credit(self):
        c = LLMClient()
        self.assertIn("credit", str(c._http_error(402, "https://x", "poor")).lower())

    def test_generic_error_truncates(self):
        c = LLMClient()
        err = c._http_error(500, "https://x", "z" * 500)
        self.assertIn("500", str(err))

    def test_hint_per_provider(self):
        self.assertIn("LM Studio", LLMClient(provider="lmstudio")._hint())
        self.assertIn("Ollama", LLMClient(provider="ollama")._hint())
        self.assertIn("vllm", LLMClient(provider="vllm")._hint())
        self.assertIn("openrouter", LLMClient(provider="openrouter")._hint())

    def test_retry_after_header(self):
        class E:
            headers = {"Retry-After": "3"}
        self.assertEqual(LLMClient._retry_after(E(), 0), 3.0)
        self.assertEqual(LLMClient._retry_after(Exception("x"), 1), 2.0)

    def test_chat_includes_reasoning_and_extra_body(self):
        seen = {}

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def fake(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            return Resp()

        orig = urllib.request.urlopen
        urllib.request.urlopen = fake
        try:
            c = LLMClient(model="m", reasoning_effort="none", extra_body={"k": "v"}, retries=0)
            c.chat([{"role": "user", "content": "hi"}])
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(seen["body"]["reasoning_effort"], "none")
        self.assertEqual(seen["body"]["k"], "v")


if __name__ == "__main__":
    unittest.main()
