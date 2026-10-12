"""LLM client plumbing moved from test_llm_providers.py.

Mirrors src/qlab_rl/llm/client.py. No network: urlopen is stubbed.
Full coverage lands in C8; this commit only moves existing tests.
"""

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


if __name__ == "__main__":
    unittest.main()
