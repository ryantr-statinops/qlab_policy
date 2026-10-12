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


if __name__ == "__main__":
    unittest.main()
