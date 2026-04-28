import unittest
from unittest.mock import patch

from llm_advisors_cli.config import AdvisorsConfig
from llm_advisors_cli.providers import (
    _parse_gemini_output,
    ask_gemini,
    sanitize_provider_output,
)


class ProviderOutputTests(unittest.TestCase):
    def test_sanitize_provider_output_removes_terminal_controls(self):
        raw = "Thinking...\x1b[7D\x1b[K necessary\x07\nFinal answer"

        cleaned = sanitize_provider_output(raw)

        self.assertNotIn("\x1b", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertIn("Final answer", cleaned)

    def test_parse_gemini_json_output_uses_response_field(self):
        raw = '{"response": "Final\\nanswer", "stats": {"models": []}}'

        self.assertEqual(_parse_gemini_output(raw), "Final\nanswer")


class GeminiCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_uses_json_output_and_no_thinking_alias(self):
        cfg = AdvisorsConfig(thinking_enabled=False)
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["provider"] = provider
            captured["cmd"] = cmd
            return '{"response": "ok"}'

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await ask_gemini(
                "Question?",
                cfg,
                model_override="gemini-2.5-flash",
            )

        self.assertEqual(result.answer, "ok")
        self.assertEqual(result.provider, "gemini/gemini-2.5-flash")
        self.assertEqual(result.meta["runtime_model"], "gemini-2.5-flash-base")
        self.assertIn("--output-format", captured["cmd"])
        self.assertIn("json", captured["cmd"])
        self.assertEqual(captured["cmd"][2], "gemini-2.5-flash-base")
        self.assertTrue(captured["cmd"][-1].startswith("Answer directly."))


if __name__ == "__main__":
    unittest.main()
