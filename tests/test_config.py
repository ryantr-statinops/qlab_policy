"""Config loading and provider merging. Mirrors src/qlab_rl/config.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.config import (
    load,
    load_data,
    load_env,
    load_policy,
    resolve,
    resolve_provider,
)
from qlab_rl.llm.client import LLMError

from tests.helpers import BASE_CFG, PRESETS


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
        r = resolve_provider("openrouter", BASE_CFG, PRESETS,
                             {"model": "x/y", "base_url": None,
                              "temperature": None, "max_tokens": None,
                              "timeout": None})
        self.assertEqual(r["model"], "x/y")  # CLI wins
        self.assertEqual(r["base_url"], "https://openrouter.ai/api/v1")  # preset
        self.assertEqual(r["temperature"], 0.0)  # default
        self.assertFalse(r["auto_detect"])


class MissingKeyTest(unittest.TestCase):
    def test_missing_key_names_env_var(self):
        import os
        os.environ.pop("OPENROUTER_API_KEY", None)
        with self.assertRaises(LLMError) as ctx:
            resolve_provider("openrouter", BASE_CFG, PRESETS, {})
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))


class TestLoad(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load("no_such_config_xyz")

    def test_load_data_has_ticker(self):
        d = load_data()
        self.assertIn("ticker", d)

    def test_load_env_sections(self):
        cfg = load_env()
        self.assertIn("env", cfg)
        self.assertIn("split", cfg)
        self.assertIn("state", cfg)
        self.assertIn("fee_bps", cfg["env"])

    def test_load_policy_sections(self):
        cfg = load_policy()
        self.assertIn("llm", cfg)
        self.assertIn("policy", cfg)
        self.assertIn("prompt", cfg)


class TestResolve(unittest.TestCase):
    def test_absolute_passthrough(self):
        p = Path("/tmp/x.csv")
        self.assertEqual(resolve(p), p)

    def test_relative_against_root(self):
        from qlab_rl.config import ROOT
        self.assertEqual(resolve("data/raw/SPY.csv"), ROOT / "data/raw/SPY.csv")


class TestResolveProviderEdges(unittest.TestCase):
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

    def test_unknown_provider_rejected(self):
        with self.assertRaises(LLMError):
            resolve_provider("nope", BASE_CFG, PRESETS, {})

    def test_base_url_override_wins(self):
        r = resolve_provider("lmstudio", BASE_CFG, PRESETS, {"base_url": "http://x:1/v1"})
        self.assertEqual(r["base_url"], "http://x:1/v1")

    def test_timeout_and_retries_defaults(self):
        r = resolve_provider("lmstudio", BASE_CFG, PRESETS, {})
        self.assertEqual(r["timeout_sec"], 60)
        self.assertEqual(r["retries"], 2)
        self.assertEqual(r["provider"], "lmstudio")
        self.assertTrue(r["auto_detect"])

    def test_extra_headers_copied(self):
        r = resolve_provider("openrouter", BASE_CFG, PRESETS, {})
        self.assertEqual(r["extra_headers"], {"X-Title": "qlab_rl"})
        r["extra_headers"]["X-Title"] = "mut"
        self.assertEqual(PRESETS["openrouter"]["extra_headers"], {"X-Title": "qlab_rl"})

    def test_local_needs_no_key(self):
        r = resolve_provider("lmstudio", BASE_CFG, PRESETS, {})
        self.assertEqual(r["api_key"], "")
        self.assertEqual(r["api_key_env"], "")


if __name__ == "__main__":
    unittest.main()
