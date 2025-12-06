from __future__ import annotations

import asyncio
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


async def _run_cmd_async(
    provider: str,
    cmd: List[str],
    cwd: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise ProviderError(provider, f"Command not found: {cmd[0]}")

    communicate_task = asyncio.create_task(proc.communicate())
    cancel_task: Optional[asyncio.Task[None]] = None

    if cancel_event is not None:
        cancel_task = asyncio.create_task(cancel_event.wait())
        done, _ = await asyncio.wait(
            {communicate_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_task in done:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            communicate_task.cancel()
            raise ProviderError(provider, "Cancelled by user")

        cancel_task.cancel()

    stdout_b, stderr_b = await communicate_task
    stdout = (stdout_b or b"").decode().strip()
    stderr = (stderr_b or b"").decode().strip()

    if proc.returncode != 0:
        msg = f"CLI failed with code {proc.returncode}. Command: {shlex.join(cmd)}"
        if stderr:
            msg += f"\nStderr: {stderr}"
        raise ProviderError(provider, msg, returncode=proc.returncode)

    return stdout


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


async def ask_codex(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
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

    answer = await _run_cmd_async("codex", cmd, cwd=cwd, cancel_event=cancel_event)
    return ProviderResult("codex", answer, {})


async def ask_claude(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("claude", cfg)
    cmd = [pcfg.command or "claude"]
    cmd.extend(pcfg.extra_args or ["-p"])
    cmd.append(prompt)
    answer = await _run_cmd_async("claude", cmd, cwd=cwd, cancel_event=cancel_event)
    return ProviderResult("claude", answer, {})


async def ask_gemini(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("gemini", cfg)
    cmd = [pcfg.command or "gemini"]
    cmd.extend(pcfg.extra_args or ["-p"])
    cmd.append(prompt)
    answer = await _run_cmd_async("gemini", cmd, cwd=cwd, cancel_event=cancel_event)
    return ProviderResult("gemini", answer, {})


async def ask_ollama(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
    model_override: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("ollama", cfg)
    model = model_override or pcfg.model or "llama3.2"
    cmd = [pcfg.command or "ollama", "run", model]
    cmd.extend(pcfg.extra_args or [])
    cmd.append(prompt)
    answer = await _run_cmd_async("ollama", cmd, cwd=cwd, cancel_event=cancel_event)
    return ProviderResult(f"ollama/{model}", answer, {"model": model})


# registry

ProviderFn = callable


def get_provider_functions(cfg: AdvisorsConfig):
    """Return a {name: async callable} mapping with config applied."""
    fns = {
        "codex": ask_codex,
        "claude": ask_claude,
        "gemini": ask_gemini,
        "ollama": ask_ollama,
    }

    return {
        name: fn
        for name, fn in fns.items()
        if cfg.providers.get(name, None) is None or cfg.providers[name].enabled
    }
