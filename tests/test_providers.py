import unittest
from unittest.mock import patch

from llm_advisors_cli.advisors import ConcurrencyLimiter, _call_provider_async
from llm_advisors_cli.config import AdvisorsConfig, ProviderConfig
from llm_advisors_cli.providers import (
    _parse_gemini_output,
    ask_claude,
    ask_gemini,
    discover_claude_models,
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


class ClaudeCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_claude_model_override_uses_print_mode_and_model(self):
        cfg = AdvisorsConfig()
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["provider"] = provider
            captured["cmd"] = cmd
            return "ok"

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await ask_claude("Question?", cfg, model_override="sonnet")

        self.assertEqual(result.answer, "ok")
        self.assertEqual(result.provider, "claude/sonnet")
        self.assertEqual(result.meta["model"], "sonnet")
        self.assertEqual(captured["provider"], "claude")
        self.assertEqual(
            captured["cmd"],
            ["claude", "-p", "--model", "sonnet", "Question?"],
        )

    async def test_claude_configured_model_is_used_for_bare_provider(self):
        cfg = AdvisorsConfig()
        cfg.providers["claude"] = ProviderConfig(
            name="claude",
            model="opus",
            extra_args=["--print"],
        )
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["cmd"] = cmd
            return "ok"

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await ask_claude("Question?", cfg)

        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.meta["model"], "opus")
        self.assertNotIn("-p", captured["cmd"])
        self.assertEqual(
            captured["cmd"],
            ["claude", "--print", "--model", "opus", "Question?"],
        )

    async def test_claude_model_member_dispatches_with_override(self):
        cfg = AdvisorsConfig()
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["cmd"] = cmd
            return "ok"

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await _call_provider_async(
                "claude/opus",
                "Question?",
                cfg,
                ConcurrencyLimiter(cfg),
                stage="stage1",
                turn_index=1,
            )

        self.assertEqual(result.provider, "claude/opus")
        self.assertEqual(result.meta["model"], "opus")
        self.assertEqual(
            captured["cmd"],
            ["claude", "-p", "--model", "opus", "Question?"],
        )


class ClaudeDiscoveryTests(unittest.TestCase):
    def test_discover_claude_models_dedupes_configured_alias(self):
        cfg = AdvisorsConfig()
        cfg.providers["claude"] = ProviderConfig(name="claude", model="sonnet")

        self.assertEqual(discover_claude_models(cfg), ["sonnet", "opus", "haiku"])

    def test_discover_claude_models_lists_custom_configured_model_first(self):
        cfg = AdvisorsConfig()
        cfg.providers["claude"] = ProviderConfig(
            name="claude",
            model="claude-sonnet-4-6",
        )

        self.assertEqual(
            discover_claude_models(cfg),
            ["claude-sonnet-4-6", "sonnet", "opus", "haiku"],
        )


if __name__ == "__main__":
    unittest.main()
