from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AdvisorsConfig, ProviderConfig
from .exceptions import ProviderError


DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2",
]
DEFAULT_GEMINI_MODELS = [
    "auto-gemini-3",
    "auto-gemini-2.5",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
GEMINI_NO_THINKING_ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash-base",
    "gemini-3-flash-preview": "gemini-3-flash-base",
}

_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"
    r"|\][^\x07]*(?:\x07|\x1b\\)"
    r"|[@-Z\\-_]"
    r")"
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class ProviderResult:
    provider: str
    answer: str
    meta: Dict[str, Any]
    duration_seconds: float | None = None


def sanitize_provider_output(text: str) -> str:
    """Remove terminal control bytes that should never become prompt context."""
    if not text:
        return ""
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def _truncate_arg_for_error(arg: str, max_len: int = 320) -> str:
    clean = sanitize_provider_output(arg)
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len]}...<truncated {len(clean) - max_len} chars>"


def _format_command_for_error(cmd: List[str]) -> str:
    return shlex.join([_truncate_arg_for_error(part) for part in cmd])


def _has_cli_option(args: List[str], names: set[str]) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in args)


def _parse_gemini_output(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except Exception:
        return sanitize_provider_output(raw)

    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, str):
            return sanitize_provider_output(response)
        error = payload.get("error")
        if error:
            return sanitize_provider_output(json.dumps(error, ensure_ascii=False))
    return sanitize_provider_output(raw)


def _gemini_runtime_model(model: Optional[str], thinking_enabled: bool) -> Optional[str]:
    if thinking_enabled or model is None:
        return model
    return GEMINI_NO_THINKING_ALIASES.get(model, model)


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
    stdout = (stdout_b or b"").decode(errors="replace").strip()
    stderr = sanitize_provider_output((stderr_b or b"").decode(errors="replace"))

    if proc.returncode != 0:
        msg = f"CLI failed with code {proc.returncode}. Command: {_format_command_for_error(cmd)}"
        if stderr:
            msg += f"\nStderr: {stderr}"
        raise ProviderError(provider, msg, returncode=proc.returncode)

    return stdout


def _merge_provider_config(name: str, cfg: AdvisorsConfig) -> ProviderConfig:
    base = ProviderConfig(
        name=name,
        model=DEFAULT_CODEX_MODEL if name == "codex" else None,
    )
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
    model_override: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("codex", cfg)
    model = model_override or pcfg.model

    # Base command: codex exec ...
    cmd = [pcfg.command or "codex", "exec"]

    if not cfg.thinking_enabled:
        cmd.extend(["-c", 'model_reasoning_effort="low"'])

    # Default model, overridable via config.toml
    if model:
        cmd.extend(["-m", model])

    # Any extra args you've configured (e.g. --no-color)
    if pcfg.extra_args:
        cmd.extend(pcfg.extra_args)

    cmd.append(prompt)

    raw = await _run_cmd_async("codex", cmd, cwd=cwd, cancel_event=cancel_event)
    answer = sanitize_provider_output(raw)
    provider_name = f"codex/{model}" if model_override and model else "codex"
    return ProviderResult(provider_name, answer, {"model": model})


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
    raw = await _run_cmd_async("claude", cmd, cwd=cwd, cancel_event=cancel_event)
    answer = sanitize_provider_output(raw)
    return ProviderResult("claude", answer, {})


async def ask_gemini(
    prompt: str,
    cfg: AdvisorsConfig,
    cwd: Optional[str] = None,
    model_override: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> ProviderResult:
    pcfg = _merge_provider_config("gemini", cfg)
    cmd = [pcfg.command or "gemini"]
    requested_model = model_override or pcfg.model
    runtime_model = _gemini_runtime_model(requested_model, cfg.thinking_enabled)
    if runtime_model:
        cmd.extend(["-m", runtime_model])
    extra_args = pcfg.extra_args or ["-p"]
    if not _has_cli_option(extra_args, {"--output-format", "-o"}):
        cmd.extend(["--output-format", "json"])
    cmd.extend(extra_args)
    if not cfg.thinking_enabled:
        prompt = (
            "Answer directly. Do not include thinking, planning, hidden reasoning, "
            "or a 'Thinking...' section.\n\n"
            + prompt
        )
    cmd.append(prompt)
    raw = await _run_cmd_async("gemini", cmd, cwd=cwd, cancel_event=cancel_event)
    answer = _parse_gemini_output(raw)
    provider_name = f"gemini/{requested_model}" if model_override and requested_model else "gemini"
    meta = {"model": requested_model}
    if runtime_model != requested_model:
        meta["runtime_model"] = runtime_model
    return ProviderResult(provider_name, answer, meta)


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
    raw = await _run_cmd_async("ollama", cmd, cwd=cwd, cancel_event=cancel_event)
    answer = sanitize_provider_output(raw)
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


def _dedupe_models(models: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def discover_codex_models(cfg: AdvisorsConfig) -> List[str]:
    """Return model names available to the local Codex CLI."""
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    cache_path = codex_home / "models_cache.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            models: List[str] = []
            for item in data.get("models", []):
                if not isinstance(item, dict):
                    continue
                if item.get("visibility") != "list":
                    continue
                slug = item.get("slug")
                if isinstance(slug, str):
                    models.append(slug)
            if models:
                return _dedupe_models(models)
        except Exception:
            pass

    configured = cfg.providers.get("codex", ProviderConfig(name="codex")).model
    return _dedupe_models(([configured] if configured else []) + DEFAULT_CODEX_MODELS)


def _gemini_package_root(command: str) -> Optional[Path]:
    binary = shutil.which(command)
    if not binary:
        return None
    path = Path(binary).resolve()
    for parent in [path.parent, *path.parents]:
        if parent.name == "gemini-cli" and parent.parent.name == "@google":
            return parent
    return None


def discover_gemini_models(cfg: AdvisorsConfig) -> List[str]:
    """Return Gemini CLI model names from local bundled docs plus safe defaults."""
    pcfg = cfg.providers.get("gemini", ProviderConfig(name="gemini"))
    configured = [pcfg.model] if pcfg.model else []
    package_root = _gemini_package_root(pcfg.command or "gemini")
    docs_path = package_root / "bundle" / "docs" / "reference" / "configuration.md" if package_root else None
    models: List[str] = []
    if docs_path and docs_path.exists():
        try:
            text = docs_path.read_text(encoding="utf-8")
            for model in DEFAULT_GEMINI_MODELS:
                if f'"{model}"' in text:
                    models.append(model)
        except Exception:
            pass

    return _dedupe_models(configured + models + DEFAULT_GEMINI_MODELS)


def discover_ollama_models(cfg: AdvisorsConfig) -> List[str]:
    """Return a list of available ollama model names (no 'ollama/' prefix)."""
    pcfg = cfg.providers.get("ollama", ProviderConfig(name="ollama"))
    cmd = [pcfg.command or "ollama", "list", "--format", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(out.stdout or "[]")
        models: List[str] = []
        if isinstance(data, list):
            for item in data:
                name = item.get("name") if isinstance(item, dict) else None
                if isinstance(name, str):
                    models.append(name)
        if models:
            return models
    except Exception:
        # fallback to text parsing below
        pass

    # Text fallback: first column of `ollama list`
    fallback_cmd = [pcfg.command or "ollama", "list"]
    try:
        out = subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
        lines = (out.stdout or "").strip().splitlines()
        models: List[str] = []
        for line in lines:
            if not line or line.lower().startswith("name"):
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []
