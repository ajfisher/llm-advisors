from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .advisors import (
    AdvisorCall,
    ChairmanCall,
    ConcurrencyLimiter,
    ReviewCall,
    TurnRecord,
    alog,
    build_turn_prompt,
    stage1_first_opinions_async,
    stage2_reviews_async,
    stage3_chairman_async,
)
from .config import AdvisorsConfig, ProviderParallelismConfig
from .exceptions import ProviderError


def _default_ollama_mode(cfg: AdvisorsConfig) -> ProviderParallelismConfig:
    return cfg.parallelism.get("ollama", ProviderParallelismConfig())


def generate_conversation_id() -> str:
    now = datetime.now()
    suffix = os.urandom(4).hex()
    return f"{now:%Y%m%d-%H%M%S}-{suffix}"


def _config_summary(cfg: AdvisorsConfig, log_dir: Path, log_enabled: bool) -> Dict[str, Any]:
    return {
        "max_parallel": cfg.max_parallel,
        "ollama_parallel_mode": _default_ollama_mode(cfg).mode,
        "ollama_max_parallel": _default_ollama_mode(cfg).max_parallel,
        "logging_enabled": log_enabled,
        "log_dir": str(log_dir),
    }


@dataclass
class ConversationRun:
    conversation_id: str
    created_at: str
    question: str
    members: List[str]
    chairman: str
    turns_requested: int
    config: Dict[str, Any]
    turns: List[TurnRecord] = field(default_factory=list)
    error: Optional[str] = None


class ConversationLogger:
    def __init__(self, base_dir: Path, enabled: bool):
        self.base_dir = base_dir
        self.enabled = enabled
        self.conversation_dir: Optional[Path] = None

    def prepare(self, conversation_id: str) -> None:
        if not self.enabled:
            return
        self.conversation_dir = self.base_dir / conversation_id
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

    def write_meta(self, run: ConversationRun, version: str) -> None:
        if not self.enabled or self.conversation_dir is None:
            return
        meta_path = self.conversation_dir / "meta.json"
        payload = {
            "conversation_id": run.conversation_id,
            "created_at": run.created_at,
            "question": run.question,
            "members": run.members,
            "chairman": run.chairman,
            "turns": run.turns_requested,
            "config": run.config,
            "version": version,
        }
        if run.error:
            payload["error"] = run.error
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_turn(self, turn: TurnRecord) -> None:
        if not self.enabled or self.conversation_dir is None:
            return
        turn_path = self.conversation_dir / f"turn-{turn.turn_index:02d}.json"
        payload = {
            "turn_index": turn.turn_index,
            "turn_prompt": turn.turn_prompt,
            "advisors": [
                {
                    "provider": adv.provider,
                    "prompt": adv.prompt,
                    "answer": adv.answer,
                    "meta": adv.meta,
                }
                for adv in turn.opinions
            ],
            "reviews": [
                {
                    "provider": rev.provider,
                    "prompt": rev.prompt,
                    "review": rev.raw_review,
                }
                for rev in turn.reviews
            ],
            "chairman": {
                "provider": turn.chairman.provider,
                "prompt": turn.chairman.prompt,
                "answer": turn.chairman.answer,
            },
        }
        turn_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_error_log(self, message: str) -> None:
        if not self.enabled or self.conversation_dir is None:
            return
        error_path = self.conversation_dir / "error.log"
        error_path.write_text(message, encoding="utf-8")


async def run_conversation_async(
    question: str,
    members: List[str],
    chairman: str,
    turns: int,
    cfg: AdvisorsConfig,
    *,
    log_dir: Optional[Path] = None,
    log_enabled: bool = True,
    show_progress: bool = True,
    version: str = "0.2.0",
) -> ConversationRun:
    conversation_id = generate_conversation_id()
    created_at = datetime.now(timezone.utc).astimezone().isoformat()
    limiter = ConcurrencyLimiter(cfg)
    effective_log_dir = log_dir or cfg.logging.base_dir
    effective_log_enabled = log_enabled and cfg.logging.enabled
    run = ConversationRun(
        conversation_id=conversation_id,
        created_at=created_at,
        question=question,
        members=members,
        chairman=chairman,
        turns_requested=turns,
        config=_config_summary(cfg, effective_log_dir, effective_log_enabled),
    )

    logger = ConversationLogger(effective_log_dir, enabled=effective_log_enabled)
    logger.prepare(conversation_id)
    logger.write_meta(run, version=version)

    try:
        for turn_index in range(1, turns + 1):
            turn_prompt = build_turn_prompt(question, run.turns)
            if show_progress:
                alog(f"[turn {turn_index}/{turns}] Collecting first opinions from: {', '.join(members)} ...")
            opinions = await stage1_first_opinions_async(
                turn_prompt,
                members,
                cfg,
                limiter,
                log_progress=show_progress,
            )
            if show_progress:
                alog(f"[turn {turn_index}/{turns}] Requesting peer reviews from: {', '.join(members)} ...")
            reviews = await stage2_reviews_async(
                turn_prompt,
                opinions,
                members,
                cfg,
                limiter,
                log_progress=show_progress,
            )
            if show_progress:
                alog(f"[turn {turn_index}/{turns}] Asking chairman '{chairman}' to synthesize final answer ...")
            chairman_call = await stage3_chairman_async(
                turn_prompt,
                opinions,
                reviews,
                chairman,
                cfg,
                limiter,
                log_progress=show_progress,
            )

            turn_record = TurnRecord(
                turn_index=turn_index,
                opinions=opinions,
                reviews=reviews,
                chairman=chairman_call,
                turn_prompt=turn_prompt,
            )
            run.turns.append(turn_record)
            logger.write_turn(turn_record)

    except Exception as exc:  # pragma: no cover - defensive
        run.error = str(exc)
        logger.write_error_log(run.error)
        logger.write_meta(run, version=version)
        raise

    logger.write_meta(run, version=version)
    return run


def run_conversation(
    question: str,
    members: List[str],
    chairman: str,
    turns: int,
    cfg: AdvisorsConfig,
    *,
    log_dir: Optional[Path] = None,
    log_enabled: bool = True,
    show_progress: bool = True,
    version: str = "0.2.0",
) -> ConversationRun:
    return asyncio.run(
        run_conversation_async(
            question,
            members,
            chairman,
            turns,
            cfg,
            log_dir=log_dir,
            log_enabled=log_enabled,
            show_progress=show_progress,
            version=version,
        )
    )
