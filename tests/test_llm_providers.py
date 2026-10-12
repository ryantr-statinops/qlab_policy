"""Provider plumbing: merge precedence, auth headers, error mapping.

No network. urllib.request.urlopen is stubbed per test.
"""

import io
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.config import resolve_provider
from qlab_rl.llm.client import LLMClient, LLMError


BASE = {
    "base_url": "http://localhost:1234/v1",
    "model": "",
    "temperature": 0.0,
    "max_tokens": 160,
    "timeout_sec": 60,
    "retries": 2,
    "reasoning_effort": "none",
}

PRESETS = {
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "",
                 "api_key_env": "", "auto_detect": True, "timeout_sec": 60},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "qwen/qwen3-4b-instruct",
                   "api_key_env": "OPENROUTER_API_KEY",
                   "auto_detect": False, "timeout_sec": 90,
                   "extra_headers": {"X-Title": "qlab_rl"}},
}


class MergeTest(unittest.TestCase):
    def setUp(self):
        import os
        self._saved = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "dummy"

    def tearDown(self):
        import os
        if self._saved is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self._saved

    def test_cli_beats_preset_beats_default(self):
        r = resolve_provider("openrouter", BASE, PRESETS,
                             {"model": "x/y", "base_url": None,
                              "temperature": None, "max_tokens": None,
                              "timeout": None})
        self.assertEqual(r["model"], "x/y")  # CLI wins
        self.assertEqual(r["base_url"], "https://openrouter.ai/api/v1")  # preset
        self.assertEqual(r["temperature"], 0.0)  # default
        self.assertFalse(r["auto_detect"])


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

    def test_missing_key_names_env_var(self):
        import os
        os.environ.pop("OPENROUTER_API_KEY", None)
        with self.assertRaises(LLMError) as ctx:
            resolve_provider("openrouter", BASE, PRESETS, {})
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))


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

    def test_record_fields_append_only(self):
        from qlab_rl.runner import RECORD_FIELDS
        self.assertEqual(RECORD_FIELDS[-2:], ["provider", "model"])
        self.assertEqual(RECORD_FIELDS[:11], [
            "step", "date", "price", "action", "reward", "equity",
            "gross_return", "cost", "turnover", "response", "parse_ok",
        ])


if __name__ == "__main__":
    unittest.main()
