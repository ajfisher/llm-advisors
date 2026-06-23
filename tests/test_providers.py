import unittest
from unittest.mock import patch

from llm_advisors_cli.advisors import ConcurrencyLimiter, _call_provider_async
from llm_advisors_cli.config import AdvisorsConfig, ProviderConfig
from llm_advisors_cli.exceptions import ProviderError
from llm_advisors_cli.providers import (
    ask_agy,
    ask_claude,
    discover_agy_models,
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


class AgyCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_agy_model_override_uses_print_mode_and_model(self):
        cfg = AdvisorsConfig()
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["provider"] = provider
            captured["cmd"] = cmd
            return "ok"

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await ask_agy(
                "Question?",
                cfg,
                model_override="Gemini 3.5 Flash (Low)",
            )

        self.assertEqual(result.answer, "ok")
        self.assertEqual(result.provider, "agy/Gemini 3.5 Flash (Low)")
        self.assertEqual(result.meta["model"], "Gemini 3.5 Flash (Low)")
        self.assertEqual(captured["provider"], "agy")
        self.assertEqual(
            captured["cmd"],
            [
                "agy",
                "--model",
                "Gemini 3.5 Flash (Low)",
                "-p",
                "Question?",
            ],
        )

    async def test_agy_preserves_extra_args_and_configured_model(self):
        cfg = AdvisorsConfig()
        cfg.providers["agy"] = ProviderConfig(
            name="agy",
            model="Gemini 3.1 Pro (Low)",
            extra_args=["--print", "--print-timeout", "30s"],
        )
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["cmd"] = cmd
            return "ok"

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await ask_agy("Question?", cfg)

        self.assertEqual(result.provider, "agy")
        self.assertEqual(result.meta["model"], "Gemini 3.1 Pro (Low)")
        self.assertNotIn("-p", captured["cmd"])
        self.assertEqual(
            captured["cmd"],
            [
                "agy",
                "--model",
                "Gemini 3.1 Pro (Low)",
                "--print-timeout",
                "30s",
                "--print",
                "Question?",
            ],
        )

    async def test_agy_model_member_dispatches_with_override(self):
        cfg = AdvisorsConfig()
        captured = {}

        async def fake_run(provider, cmd, cwd=None, cancel_event=None):
            captured["cmd"] = cmd
            return "ok"

        with patch("llm_advisors_cli.providers._run_cmd_async", fake_run):
            result = await _call_provider_async(
                "agy/Gemini 3.5 Flash (High)",
                "Question?",
                cfg,
                ConcurrencyLimiter(cfg),
                stage="stage1",
                turn_index=1,
            )

        self.assertEqual(result.provider, "agy/Gemini 3.5 Flash (High)")
        self.assertEqual(result.meta["model"], "Gemini 3.5 Flash (High)")
        self.assertEqual(
            captured["cmd"],
            [
                "agy",
                "--model",
                "Gemini 3.5 Flash (High)",
                "-p",
                "Question?",
            ],
        )


class AgyDiscoveryTests(unittest.TestCase):
    def test_discover_agy_models_parses_cli_output_and_dedupes_configured_model(self):
        cfg = AdvisorsConfig()
        cfg.providers["agy"] = ProviderConfig(
            name="agy",
            model="Gemini 3.5 Flash (Low)",
        )
        completed = type(
            "Completed",
            (),
            {
                "stdout": "\n".join(
                    [
                        "Gemini 3.5 Flash (Medium)",
                        "Gemini 3.5 Flash (Low)",
                        "Claude Sonnet 4.6 (Thinking)",
                    ]
                )
            },
        )()

        with patch("llm_advisors_cli.providers.subprocess.run", return_value=completed):
            models = discover_agy_models(cfg)

        self.assertEqual(
            models,
            [
                "Gemini 3.5 Flash (Low)",
                "Gemini 3.5 Flash (Medium)",
                "Claude Sonnet 4.6 (Thinking)",
            ],
        )

    def test_discover_agy_models_falls_back_to_default(self):
        cfg = AdvisorsConfig()

        with patch("llm_advisors_cli.providers.subprocess.run", side_effect=OSError):
            models = discover_agy_models(cfg)

        self.assertEqual(models, ["Gemini 3.5 Flash (Medium)"])


class UnsupportedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_provider_is_not_configured(self):
        cfg = AdvisorsConfig()

        with self.assertRaises(ProviderError):
            await _call_provider_async(
                "gemini/gemini-2.5-flash",
                "Question?",
                cfg,
                ConcurrencyLimiter(cfg),
                stage="stage1",
                turn_index=1,
            )


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
