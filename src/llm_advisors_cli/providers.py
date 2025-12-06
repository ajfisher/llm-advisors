from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import AdvisorsConfig, ProviderConfig
from .exceptions import ProviderError


@dataclass
class ProviderResult:
    provider: str
    answer: str
    meta: Dict[str, Any]


def _run_cmd(
    provider: str,
    cmd: List[str],
    cwd: Optional[str] = None,
) -> str:
    try:
        out = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        msg = f"CLI failed with code {e.returncode}. Command: {shlex.join(cmd)}"
        if stderr:
            msg += f"\nStderr: {stderr}"
        raise ProviderError(provider, msg, returncode=e.returncode)

    return (out.stdout or "").strip()


def _merge_provider_config(name: str, cfg: AdvisorsConfig) -> ProviderConfig:
    base = ProviderConfig(name=name)
    override = cfg.providers.get(name)
    if not override:
        return base

    return ProviderConfig(
        name=name,
        enabled=override.enabled if override.enabled is not None else base.enabled,
        command=override.command or base.command,
        model=override.model or base.model,
        extra_args=override.extra_args or base.extra_args,
    )


def ask_codex(
    prompt: str,
    cfg: CouncilConfig,
    cwd: Optional[str] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("codex", cfg)

    # Base command: codex exec ...
    cmd = [pcfg.command or "codex", "exec"]

    # Optional model from config.toml
    if pcfg.model:
        cmd.extend(["-m", pcfg.model])

    # Any extra args you’ve configured (e.g. --no-color)
    if pcfg.extra_args:
        cmd.extend(pcfg.extra_args)

    cmd.append(prompt)

    answer = _run_cmd("codex", cmd, cwd=cwd)
    return ProviderResult("codex", answer, {})



def ask_claude(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("claude", cfg)
    cmd = [pcfg.command or "claude"]
    # minimal: `claude -p prompt`. If you prefer `--output-format` etc, add here or in config.
    cmd.extend(pcfg.extra_args or ["-p"])
    cmd.append(prompt)
    answer = _run_cmd("claude", cmd, cwd=cwd)
    return ProviderResult("claude", answer, {})


def ask_gemini(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("gemini", cfg)
    cmd = [pcfg.command or "gemini"]
    cmd.extend(pcfg.extra_args or ["-p"])
    cmd.append(prompt)
    answer = _run_cmd("gemini", cmd, cwd=cwd)
    return ProviderResult("gemini", answer, {})


def ask_ollama(
    prompt: str,
    cfg: CouncilConfig,
    cwd: Optional[str] = None,
    model_override: Optional[str] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("ollama", cfg)
    model = model_override or pcfg.model or "llama3.2"
    cmd = [pcfg.command or "ollama", "run", model]
    cmd.extend(pcfg.extra_args or [])
    cmd.append(prompt)
    answer = _run_cmd("ollama", cmd, cwd=cwd)
    return ProviderResult(f"ollama/{model}", answer, {"model": model})


# registry

ProviderFn = callable


def get_provider_functions(cfg: AdvisorsConfig):
    """Return a {name: callable} mapping with config applied."""
    fns = {
        "codex": ask_codex,
        "claude": ask_claude,
        "gemini": ask_gemini,
        "ollama": ask_ollama,
    }

    # honour enabled = false in config
    return {name: fn for name, fn in fns.items() if cfg.providers.get(name, None) is None or cfg.providers[name].enabled}

