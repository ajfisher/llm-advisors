from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "llm_advisors"
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class ProviderConfig:
    name: str
    enabled: bool = True
    command: str | None = None  # override CLI command if needed
    model: str | None = None    # mainly for Ollama
    extra_args: List[str] = field(default_factory=list)


@dataclass
class AdvisorsConfig:
    members: List[str] = field(default_factory=lambda: ["codex", "claude", "gemini", "ollama"])
    chairman: str = "codex"
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)


def load_config() -> AdvisorsConfig:
    """Load config from ~/.config/llm_advisors/config.toml if it exists."""
    cfg = AdvisorsConfig()

    if tomllib is None or not CONFIG_PATH.exists():
        return cfg

    with CONFIG_PATH.open("rb") as f:
        data = tomllib.load(f)

    # general
    general = data.get("general", {})
    members = general.get("members")
    if isinstance(members, list):
        cfg.members = [str(m) for m in members if isinstance(m, str)]

    chairman = general.get("chairman")
    if isinstance(chairman, str):
        cfg.chairman = chairman

    # providers
    providers_section = data.get("providers", {})
    if isinstance(providers_section, dict):
        for name, raw in providers_section.items():
            if not isinstance(raw, dict):
                continue
            pc = ProviderConfig(
                name=name,
                enabled=bool(raw.get("enabled", True)),
                command=str(raw["command"]) if "command" in raw else None,
                model=str(raw["model"]) if "model" in raw else None,
                extra_args=[str(a) for a in raw.get("extra_args", []) if isinstance(a, str)],
            )
            cfg.providers[name] = pc

    return cfg

