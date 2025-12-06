# Coding Agent Guide

Notes for working on this repo as an automated coding agent.

## What the tool does
- CLI orchestrates a three-stage "council of LLMs" flow: first opinions → peer reviews → chairman synthesis.
- All inference is delegated to external CLIs (`codex`, `claude`, `gemini`, `ollama`); this project only builds prompts and shells out.
- Entry point: `src/llm_advisors_cli/cli.py`; provider commands live in `providers.py`; prompts and flow are in `advisors.py`; config parsing is in `config.py`.

## Running locally
- Requires Python 3.11+.
- Install editable for PATH access: `pip install -e .` then run `llm-advisors "<question>"`.
- Config file (optional) lives at `~/.config/llm_advisors/config.toml`; defaults are in code (`codex`, `claude`, `gemini`, `ollama` members; `codex` chairman).
- Use CLI flags `--members`, `--chairman`, `--show-intermediate` to override config at runtime. Ollama model overrides use the `ollama/<model>` naming.

## Developing and testing
- No automated tests yet; quick manual check: point providers at simple echo scripts via `providers.<name>.command` in the config, then run `llm-advisors --show-intermediate "test prompt"` to verify flow without calling real APIs.
- Provider errors bubble up as `ProviderError` with the underlying CLI stderr; keep error messages concise and actionable.
- Keep documentation in sync with behaviour; update `README.md` when adding flags or config fields.
- There are lingering type hints referring to `CouncilConfig`; they should map to `AdvisorsConfig` when you touch those areas.

## Style and maintenance
- Stay in ASCII; prefer small, clear functions and early exits.
- Respect the existing stage structure and prompt shapes unless intentionally redesigning them.
- When adding providers or flags, ensure they can be configured via TOML and overridden via CLI arguments.
