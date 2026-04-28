import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_advisors_cli.config import AdvisorsConfig, load_config


class ConfigDefaultsTests(unittest.TestCase):
    def test_default_members_and_chair_use_requested_models(self):
        cfg = AdvisorsConfig()

        self.assertEqual(
            cfg.members,
            [
                "codex/gpt-5.2",
                "gemini/gemini-2.5-flash",
                "codex/gpt-5.4",
                "ollama/gemma4:latest",
            ],
        )
        self.assertEqual(cfg.chairman, "codex/gpt-5.5")

    def test_load_config_accepts_neutral_chair_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                """
[general]
chair = "gemini/gemini-2.5-flash"
chairman = "codex/gpt-5.2"
""".strip(),
                encoding="utf-8",
            )
            with patch("llm_advisors_cli.config.CONFIG_PATH", config_path):
                cfg = load_config()

        self.assertEqual(cfg.chairman, "gemini/gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
