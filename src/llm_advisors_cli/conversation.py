from __future__ import annotations

import asyncio
import json
import os
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .advisors import (
    AdvisorCall,
    ChairmanCall,
    ConcurrencyLimiter,
    ReviewCall,
    TurnRecord,
    alog,
    ProgressEvent,
    stage1_first_opinions_async,
    stage2_reviews_async,
    stage3_chairman_async,
)
from .config import AdvisorsConfig, ProviderParallelismConfig
from .exceptions import ProviderError


def _default_ollama_mode(cfg: AdvisorsConfig) -> ProviderParallelismConfig:
    return cfg.parallelism.get("ollama", ProviderParallelismConfig())

# Roles for post-baseline turns
ROLE_POOL = [
    "Explorer",
    "Skeptic",
    "Synthesiser",
    "Pragmatist",
    "Theorist",
    "Contrarian",
]


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


def _turn_plan(turns_requested: int) -> List[Dict[str, Any]]:
    turns = max(1, min(turns_requested, 4))
    if turns == 1:
        return [{"kind": "baseline", "final": True}]
    if turns == 2:
        return [
            {"kind": "baseline", "final": False},
            {"kind": "convergence", "final": True},
        ]
    if turns == 3:
        return [
            {"kind": "baseline", "final": False},
            {"kind": "divergence", "final": False},
            {"kind": "convergence", "final": True},
        ]
    return [
        {"kind": "baseline", "final": False},
        {"kind": "divergence", "final": False},
        {"kind": "task_solve", "final": False},
        {"kind": "convergence", "final": True},
    ]


def _assign_roles(members: List[str]) -> Dict[str, str]:
    rng = random.Random()
    roles = ROLE_POOL.copy()
    rng.shuffle(roles)
    assignments: Dict[str, str] = {}
    for idx, member in enumerate(members):
        role = roles[idx % len(roles)]
        assignments[member] = role
    return assignments


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a response."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def _build_advisor_prompt(
    kind: str,
    original_question: str,
    *,
    role: Optional[str],
    baseline_summary: Optional[Dict[str, Any]],
    task_object: Optional[Dict[str, Any]],
    convergence_prep: Optional[Dict[str, Any]],
) -> str:
    if kind == "baseline":
        return f"""Provide the best possible answer to the question. Full freedom.

Question:
{original_question}
"""

    if kind == "divergence":
        return f"""
You MUST expand, challenge, or critique the previous consensus.
Do NOT repeat or mildly rephrase earlier answers.

Your assigned role: {role or 'Advisor'}.

Context (turn 1 summary):
{json.dumps(baseline_summary or {}, indent=2)}

Respond in:
1. NEW INSIGHTS
2. CHALLENGES / CRITIQUES
3. WHAT SHOULD CHANGE IN THE OVERALL ANSWER
""".strip()

    if kind == "task_solve":
        return f"""
You must address the chairman's tasks from the previous turn.
Your role this turn: {role or 'Advisor'}.

Tasks to solve:
{json.dumps(task_object or {}, indent=2)}

Respond in 3 parts:
1. SOLUTIONS OR OPTIONS for each subquestion/task
2. IMPLICATIONS / SECOND-ORDER EFFECTS
3. RECOMMENDATIONS to carry into the final convergence turn
""".strip()

    # convergence (final synthesis inputs)
    return f"""
Produce your final proposal, integrating all learning from previous turns.

Original question:
{original_question}

Turn 1 summary:
{json.dumps(baseline_summary or {}, indent=2)}

Task/strategic context:
{json.dumps(task_object or {}, indent=2)}

Convergence prep:
{json.dumps(convergence_prep or {}, indent=2)}

Respond in:
1. FINAL PROPOSAL
2. RATIONALE
3. RISKS / REMAINING GAPS
""".strip()


def _build_review_prompt(
    kind: str,
    base_question: str,
    opinions: List[AdvisorCall],
    *,
    context: Optional[str] = None,
) -> str:
    shuffled = list(opinions)
    random.shuffle(shuffled)

    labelled = []
    for idx, res in enumerate(shuffled):
        label = chr(ord("A") + idx)
        labelled.append(f"{label})\n{res.answer}")
    opinions_block = "\n\n".join(labelled)

    base = f"""
Question:
{base_question}

Advisor answers (labelled A, B, C...) presented anonymously:
{opinions_block}

You do not know who wrote which. Do not guess or infer authorship. Judge purely on quality and usefulness.
"""

    if context:
        base += f"\n\nContext:\n{context}\n"

    if kind == "divergence":
        instructions = """
Focus on NEWNESS and USEFULNESS of insights, not just polish.

Tasks:
1. Critique each answer with emphasis on novelty and usefulness.
2. Rank the answers from most to least valuable for moving the solution forward.
3. Recommend concrete directions to explore next turn.
"""
    elif kind == "task_solve":
        instructions = """
Tasks:
1. Identify the strongest solutions/options and why.
2. Flag conflicts or incompatible proposals.
3. List unresolved issues that must be addressed in the final convergence.
"""
    else:
        instructions = """
Tasks:
1. Briefly critique each answer.
2. Rank the answers from best to worst (accuracy, depth, clarity).
3. Propose your improved answer labelled FINAL.
"""

    return (base + instructions).strip()


def _build_chairman_prompt(
    kind: str,
    base_question: str,
    opinions: List[AdvisorCall],
    reviews: List[ReviewCall],
    *,
    baseline_summary: Optional[Dict[str, Any]],
    task_object: Optional[Dict[str, Any]],
    convergence_prep: Optional[Dict[str, Any]],
    is_final_turn: bool,
) -> str:
    opinions_text = "\n\n".join(f"{i+1}. [{o.provider}]\n{o.answer}" for i, o in enumerate(opinions))
    reviews_text = "\n\n".join(f"[Review by {r.provider}]\n{r.raw_review}" for r in reviews)

    if kind == "baseline":
        return f"""
You are the chair of an expert council.

Question:
{base_question}

Stage 1 answers:
{opinions_text}

Stage 2 reviews:
{reviews_text}

Produce:
- Consensus
- Disagreements
- Emerging themes
- Unexplored opportunities

Also provide a concise synthesis answer to the question.
Return a JSON block for the summary with keys: consensus, disagreements, themes, unexplored_opportunities.
""".strip()

    if kind == "divergence":
        return f"""
You are in task generation mode. Do NOT produce a final answer.

Question:
{base_question}

Turn 1 summary:
{json.dumps(baseline_summary or {}, indent=2)}

Stage 1 answers:
{opinions_text}

Stage 2 reviews:
{reviews_text}

Produce a structured task sheet:

SUBQUESTIONS:
- ...
- ...

STRATEGIC DIRECTIONS:
- ...
- ...

KEY TENSIONS / TRADEOFFS IDENTIFIED:
- ...

TASK LIST FOR NEXT TURN:
- ...

Return a JSON block with fields: subquestions, strategic_directions, tensions, tasks.
""".strip()

    if kind == "task_solve":
        return f"""
Prepare convergence. Do NOT give the final answer.

Question:
{base_question}

Tasks from previous turn:
{json.dumps(task_object or {}, indent=2)}

Stage 1 answers:
{opinions_text}

Stage 2 reviews:
{reviews_text}

Produce:
- UNRESOLVED ISSUES
- BEST OPTIONS IDENTIFIED
- RECOMMENDED FINAL DIRECTION

Return a JSON block with fields: unresolved_issues, best_options, recommended_direction.
""".strip()

    # convergence
    return f"""
Produce the FINAL ANSWER, integrating all insights.

Original question:
{base_question}

Turn 1 summary:
{json.dumps(baseline_summary or {}, indent=2)}

Task sheet:
{json.dumps(task_object or {}, indent=2)}

Convergence prep:
{json.dumps(convergence_prep or {}, indent=2)}

Stage 1 answers:
{opinions_text}

Stage 2 reviews:
{reviews_text}

Include:
- EXECUTIVE SUMMARY
- FULL SYNTHESIS
- TRADEOFFS
- RECOMMENDED NEXT STEPS
""".strip()


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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        if run.metadata:
            payload["metadata"] = run.metadata
        if run.error:
            payload["error"] = run.error
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_turn(self, turn: TurnRecord) -> None:
        if not self.enabled or self.conversation_dir is None:
            return
        turn_path = self.conversation_dir / f"turn-{turn.turn_index:02d}.json"
        payload = {
            "turn_index": turn.turn_index,
            "kind": turn.kind,
            "turn_prompt": turn.turn_prompt,
            "roles": turn.roles,
            "advisors": [
                {
                    "provider": adv.provider,
                    "prompt": adv.prompt,
                    "answer": adv.answer,
                    "meta": adv.meta,
                    "role": adv.role,
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
        if turn.summary_object is not None:
            payload["summary_object"] = turn.summary_object
        if turn.task_object is not None:
            payload["task_object"] = turn.task_object
        if turn.convergence_prep is not None:
            payload["convergence_prep"] = turn.convergence_prep
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
    progress_handler: Optional[Callable[[ProgressEvent], None]] = None,
    cancel_token: Optional[threading.Event] = None,
    conversation_id: Optional[str] = None,
) -> ConversationRun:
    conversation_id = conversation_id or generate_conversation_id()
    created_at = datetime.now(timezone.utc).astimezone().isoformat()
    limiter = ConcurrencyLimiter(cfg)
    effective_log_dir = log_dir or cfg.logging.base_dir
    effective_log_enabled = log_enabled and cfg.logging.enabled
    plan = _turn_plan(turns)
    effective_turns = len(plan)
    run = ConversationRun(
        conversation_id=conversation_id,
        created_at=created_at,
        question=question,
        members=members,
        chairman=chairman,
        turns_requested=effective_turns,
        config=_config_summary(cfg, effective_log_dir, effective_log_enabled),
    )

    logger = ConversationLogger(effective_log_dir, enabled=effective_log_enabled)
    logger.prepare(conversation_id)
    logger.write_meta(run, version=version)

    internal_cancel = asyncio.Event()
    run.metadata["turn_plan"] = plan
    baseline_summary: Optional[Dict[str, Any]] = None
    task_object: Optional[Dict[str, Any]] = None
    convergence_prep: Optional[Dict[str, Any]] = None

    async def _monitor_cancel() -> None:
        if cancel_token is None:
            return
        while not internal_cancel.is_set():
            if cancel_token.is_set():
                internal_cancel.set()
                break
            await asyncio.sleep(0.1)

    monitor_task: Optional[asyncio.Task[Any]] = None
    if cancel_token is not None:
        monitor_task = asyncio.create_task(_monitor_cancel())

    def emit(event: ProgressEvent) -> None:
        if progress_handler:
            progress_handler(event)

    try:
        for turn_index, turn_cfg in enumerate(plan, start=1):
            if internal_cancel.is_set():
                raise ProviderError("cancelled", "Conversation cancelled")

            kind = turn_cfg["kind"]
            is_final_turn = bool(turn_cfg.get("final", False))

            roles = _assign_roles(members) if kind in ("divergence", "task_solve") else {}

            advisor_prompts: Dict[str, str] = {}
            for member in members:
                advisor_prompts[member] = _build_advisor_prompt(
                    kind,
                    question,
                    role=roles.get(member),
                    baseline_summary=baseline_summary,
                    task_object=task_object,
                    convergence_prep=convergence_prep,
                )

            turn_prompt = f"Turn {turn_index} ({kind})"
            emit(
                ProgressEvent(
                    event="turn",
                    turn=turn_index,
                    stage="turn",
                    status="start",
                    message=json.dumps(roles) if roles else None,
                )
            )
            if show_progress:
                alog(f"[turn {turn_index}/{effective_turns}] Collecting first opinions from: {', '.join(members)} ...")
            opinions = await stage1_first_opinions_async(
                turn_prompt,
                members,
                cfg,
                limiter,
                log_progress=show_progress,
                cancel_event=internal_cancel,
                progress_handler=progress_handler,
                turn_index=turn_index,
                prompts_by_provider=advisor_prompts,
                roles=roles,
            )
            if show_progress:
                alog(f"[turn {turn_index}/{effective_turns}] Requesting peer reviews from: {', '.join(members)} ...")

            review_context = ""
            if baseline_summary:
                review_context += f"Baseline summary: {json.dumps(baseline_summary, indent=2)}\n"
            if task_object:
                review_context += f"Task sheet: {json.dumps(task_object, indent=2)}\n"
            if convergence_prep:
                review_context += f"Convergence prep: {json.dumps(convergence_prep, indent=2)}\n"
            review_prompt = _build_review_prompt(
                kind,
                question,
                opinions,
                context=review_context.strip() or None,
            )
            reviews = await stage2_reviews_async(
                turn_prompt,
                opinions,
                members,
                cfg,
                limiter,
                log_progress=show_progress,
                cancel_event=internal_cancel,
                progress_handler=progress_handler,
                turn_index=turn_index,
                review_prompt_override=review_prompt,
            )
            if show_progress:
                alog(f"[turn {turn_index}/{effective_turns}] Asking chairman '{chairman}' to synthesize final answer ...")

            chairman_prompt = _build_chairman_prompt(
                kind,
                question,
                opinions,
                reviews,
                baseline_summary=baseline_summary,
                task_object=task_object,
                convergence_prep=convergence_prep,
                is_final_turn=is_final_turn,
            )
            chairman_call = await stage3_chairman_async(
                turn_prompt,
                opinions,
                reviews,
                chairman,
                cfg,
                limiter,
                log_progress=show_progress,
                cancel_event=internal_cancel,
                progress_handler=progress_handler,
                turn_index=turn_index,
                chairman_prompt_override=chairman_prompt,
            )

            # Extract structured artefacts depending on turn kind
            summary_obj = None
            task_obj = None
            convergence_obj = None
            extracted = _extract_json_block(chairman_call.answer)
            if kind == "baseline":
                summary_obj = extracted
                baseline_summary = summary_obj or baseline_summary
            elif kind == "divergence":
                task_obj = extracted
                task_object = task_obj or task_object
            elif kind == "task_solve":
                convergence_obj = extracted
                convergence_prep = convergence_obj or convergence_prep

            turn_record = TurnRecord(
                turn_index=turn_index,
                opinions=opinions,
                reviews=reviews,
                chairman=chairman_call,
                turn_prompt=turn_prompt,
                kind=kind,
                roles=roles,
                summary_object=summary_obj,
                task_object=task_obj,
                convergence_prep=convergence_obj,
            )
            run.turns.append(turn_record)
            logger.write_turn(turn_record)
            emit(ProgressEvent(event="turn", turn=turn_index, stage="turn", status="done"))

    except Exception as exc:  # pragma: no cover - defensive
        run.error = str(exc)
        logger.write_error_log(run.error)
        logger.write_meta(run, version=version)
        raise
    finally:
        if monitor_task:
            monitor_task.cancel()

    logger.write_meta(run, version=version)
    emit(ProgressEvent(event="conversation", turn=turns, stage="done", status="done"))
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
    progress_handler: Optional[Callable[[ProgressEvent], None]] = None,
    cancel_token: Optional[threading.Event] = None,
    conversation_id: Optional[str] = None,
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
            progress_handler=progress_handler,
            cancel_token=cancel_token,
            conversation_id=conversation_id,
        )
    )
