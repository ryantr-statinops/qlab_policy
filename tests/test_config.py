"""Provider preset merging. Mirrors src/qlab_rl/config.resolve_provider."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_rl.config import resolve_provider
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


if __name__ == "__main__":
    unittest.main()
