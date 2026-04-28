import unittest
from unittest.mock import patch

from llm_advisors_cli.config import AdvisorsConfig
from llm_advisors_cli import web


class WebSelectionTests(unittest.TestCase):
    def _patch_discovery(self):
        return (
            patch("llm_advisors_cli.web.discover_codex_models", return_value=["gpt-5.5", "gpt-5.2", "gpt-5.4"]),
            patch("llm_advisors_cli.web.discover_gemini_models", return_value=["gemini-2.5-flash"]),
            patch("llm_advisors_cli.web.discover_ollama_models", return_value=["gemma4:latest", "llama3.1:8b"]),
        )

    def test_home_uses_advisor_slots_without_bare_provider_options(self):
        cfg = AdvisorsConfig(members=["codex", "claude", "gemini"], chairman="codex")
        patches = self._patch_discovery()
        with patch("llm_advisors_cli.web.load_config", return_value=cfg), patches[0], patches[1], patches[2]:
            response = web.app.test_client().get("/")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="advisor_1"', html)
        self.assertIn('name="advisor_4"', html)
        self.assertNotIn('name="members"', html)
        self.assertNotIn('<option value="codex"', html)
        self.assertNotIn('<option value="claude"', html)
        self.assertNotIn('<option value="gemini"', html)
        self.assertIn('<option value="codex/gpt-5.5"', html)
        self.assertIn('<option value="codex/gpt-5.2"', html)
        self.assertIn('<option value="codex/gpt-5.4"', html)
        self.assertIn('<option value="gemini/gemini-2.5-flash"', html)
        self.assertIn('<option value="ollama/gemma4:latest"', html)
        self.assertIn('<option value="codex/gpt-5.2" selected', html)
        self.assertIn('<option value="gemini/gemini-2.5-flash" selected', html)
        self.assertIn('<option value="codex/gpt-5.4" selected', html)
        self.assertIn('<option value="ollama/gemma4:latest" selected', html)
        self.assertIn('<option value="codex/gpt-5.5" selected', html)

    def test_start_conversation_reads_unique_advisor_slots(self):
        cfg = AdvisorsConfig(members=["codex"], chairman="codex")
        created = {}

        class FakeJob:
            def __init__(self, conversation_id, question, members, chair, turns, cfg):
                created["members"] = members
                created["chair"] = chair
                created["turns"] = turns

            def start(self):
                created["started"] = True

        patches = self._patch_discovery()
        client = web.app.test_client()
        with (
            patch("llm_advisors_cli.web.load_config", return_value=cfg),
            patch("llm_advisors_cli.web.generate_conversation_id", return_value="test-conversation"),
            patch("llm_advisors_cli.web.ConversationJob", FakeJob),
            patches[0],
            patches[1],
            patches[2],
        ):
            response = client.post(
                "/conversations",
                data={
                    "question": "Q",
                    "advisor_1": "codex/gpt-5.5",
                    "advisor_2": "codex/gpt-5.5",
                    "advisor_3": web.NONE_ADVISOR_VALUE,
                    "advisor_4": "gemini/gemini-2.5-flash",
                    "chair": "gemini/gemini-2.5-flash",
                    "turns": "2",
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(created["members"], ["codex/gpt-5.5", "gemini/gemini-2.5-flash"])
        self.assertEqual(created["chair"], "gemini/gemini-2.5-flash")
        self.assertEqual(created["turns"], 2)
        self.assertTrue(created["started"])

    def test_markdown_endpoint_matches_conversation_rendering(self):
        response = web.app.test_client().post("/markdown", json={"text": "**bold**"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("<strong>bold</strong>", response.get_json()["html"])


if __name__ == "__main__":
    unittest.main()
