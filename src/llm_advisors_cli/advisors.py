from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .config import AdvisorsConfig, ProviderParallelismConfig
from .exceptions import ProviderError
from .providers import ProviderResult, get_provider_functions


@dataclass
class AdvisorCall:
    provider: str
    prompt: str
    answer: str
    meta: Dict[str, Any]


@dataclass
class ReviewCall:
    provider: str
    prompt: str
    raw_review: str


@dataclass
class ChairmanCall:
    provider: str
    prompt: str
    answer: str


@dataclass
class ProgressEvent:
    event: str
    turn: int
    stage: str
    provider: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None


def alog(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_ollama_member(name: str) -> Optional[str]:
    """Return model name when member is like 'ollama/<model>'."""
    if not name.startswith("ollama/"):
        return None
    return name.split("/", 1)[1]


class ConcurrencyLimiter:
    """Simple concurrency limiter with global and per-provider semaphores."""

    def __init__(self, cfg: AdvisorsConfig):
        self.global_sem = asyncio.Semaphore(max(1, cfg.max_parallel))
        self.provider_sems: Dict[str, asyncio.Semaphore] = {}

        for name, pcfg in cfg.parallelism.items():
            limit = self._limit_for(pcfg)
            if limit is not None:
                self.provider_sems[name] = asyncio.Semaphore(max(1, limit))

    @staticmethod
    def _limit_for(pcfg: ProviderParallelismConfig) -> Optional[int]:
        mode = (pcfg.mode or "").lower()
        if mode == "sequential":
            return 1
        if mode == "limited":
            return pcfg.max_parallel
        return None

    @staticmethod
    def _provider_key(name: str) -> str:
        return "ollama" if name.startswith("ollama/") else name

    @asynccontextmanager
    async def limit(self, provider_name: str):
        key = self._provider_key(provider_name)
        await self.global_sem.acquire()
        provider_sem = self.provider_sems.get(key)
        if provider_sem is not None:
            await provider_sem.acquire()
        try:
            yield
        finally:
            self.global_sem.release()
            if provider_sem is not None:
                provider_sem.release()


async def _call_provider_async(
    name: str,
    prompt: str,
    cfg: AdvisorsConfig,
    limiter: ConcurrencyLimiter,
    *,
    cancel_event: Optional[asyncio.Event] = None,
    progress_handler: Optional[Callable[[ProgressEvent], None]] = None,
    stage: str,
    turn_index: int,
) -> ProviderResult:
    providers = get_provider_functions(cfg)
    base = "ollama" if name.startswith("ollama/") else name
    fn = providers.get(base)
    if not fn:
        raise ProviderError(name, f"Provider '{name}' is not configured or disabled.")

    async with limiter.limit(base):
        if cancel_event and cancel_event.is_set():
            raise ProviderError(name, "Cancelled by user")
        if progress_handler:
            progress_handler(
                ProgressEvent(
                    event="provider",
                    turn=turn_index,
                    stage=stage,
                    provider=name,
                    status="running",
                )
            )
        try:
            if base == "ollama":
                model_override = _parse_ollama_member(name)
                res = await fn(prompt, cfg, None, model_override, cancel_event)
                if model_override:
                    res.provider = name
            elif base == "codex":
                res = await fn(prompt, cfg, None, cancel_event)
            elif base == "claude":
                res = await fn(prompt, cfg, None, cancel_event)
            elif base == "gemini":
                res = await fn(prompt, cfg, None, cancel_event)
            else:
                raise ProviderError(name, f"Unknown provider '{name}'")
        except Exception as exc:
            if progress_handler:
                progress_handler(
                    ProgressEvent(
                        event="provider",
                        turn=turn_index,
                        stage=stage,
                        provider=name,
                        status="error",
                        message=str(exc),
                    )
                )
            raise

    if progress_handler:
        progress_handler(
            ProgressEvent(
                event="provider",
                turn=turn_index,
                stage=stage,
                provider=name,
                status="done",
                message=getattr(res, "answer", None),
            )
        )

    return res


def _build_review_prompt(question: str, opinions: List[AdvisorCall]) -> str:
    labelled = []
    for idx, res in enumerate(opinions):
        label = chr(ord("A") + idx)
        labelled.append(f"{label}) {res.answer}")

    opinions_block = "\n\n".join(labelled)

    return f"""
You are part of an expert panel answering a question.

Question:
{question}

Here are the panel's answers (labelled A, B, C, ...). You don't know who wrote which.

{opinions_block}

Tasks:
1. Briefly critique each answer (A, B, C, ...).
2. Rank the answers from best to worst in terms of:
   - factual accuracy
   - depth/insight
   - clarity
3. Then propose your own improved answer, labelled "FINAL".

In your response, prioritise brevity and clarity over feature count.

Respond in this format:

CRITIQUES:
A: ...
B: ...
...

RANKING (best to worst):
A > C > B > ...

FINAL:
<your best answer here>
""".strip()


def _build_chairman_prompt(
    question: str,
    opinions: List[AdvisorCall],
    reviews: List[ReviewCall],
) -> str:
    opinions_text = "\n\n".join(
        f"{i+1}. [{o.provider}]\n{o.answer}"
        for i, o in enumerate(opinions)
    )
    reviews_text = "\n\n".join(
        f"[Review by {r.provider}]\n{r.raw_review}"
        for r in reviews
    )

    return f"""
You are the chair of an expert council of language models.

Question:
{question}

Stage 1: First opinions
-----------------------
Here are the initial answers from each council member:

{opinions_text}

Stage 2: Peer reviews
---------------------
Here are the peer reviews and rankings from each member:

{reviews_text}

Task:
Synthesize a single best possible answer to the question, using the
strongest parts of the above. Resolve disagreements explicitly where relevant.
Strive for brevity and clarity in your response rather than extensiveness.

Structure your response as:

FINAL ANSWER:
<your answer>

NOTES:
- Key points of agreement
- Key points of disagreement and how you resolved them
""".strip()


def build_turn_prompt(
    original_question: str,
    previous_turns: List["TurnRecord"],
) -> str:
    if not previous_turns:
        return original_question

    context_chunks = []
    for turn in previous_turns:
        context_chunks.append(
            f"Turn {turn.turn_index} chairman answer:\n{turn.chairman.answer}"
        )
    context_block = "\n\n".join(context_chunks)

    return f"""
You are participating in a multi-turn council.

Original question:
{original_question}

Context from previous turns:
{context_block}

Now provide your updated answer for this turn, improving on previous attempts.
""".strip()


async def stage1_first_opinions_async(
    question: str,
    member_names: List[str],
    cfg: AdvisorsConfig,
    limiter: ConcurrencyLimiter,
    *,
    log_progress: bool = True,
    cancel_event: Optional[asyncio.Event] = None,
    progress_handler: Optional[Callable[[ProgressEvent], None]] = None,
    turn_index: int = 1,
) -> List[AdvisorCall]:
    tasks: List[tuple[int, asyncio.Task[ProviderResult]]] = []

    for idx, name in enumerate(member_names):
        if log_progress:
            alog(f"  - [stage1] asking {name} ...")
        task = asyncio.create_task(
            _call_provider_async(
                name,
                question,
                cfg,
                limiter,
                cancel_event=cancel_event,
                progress_handler=progress_handler,
                stage="stage1",
                turn_index=turn_index,
            )
        )
        tasks.append((idx, task))

    ordered_results: List[Optional[AdvisorCall]] = [None] * len(tasks)
    for idx, task in tasks:
        res = await task
        ordered_results[idx] = AdvisorCall(
            provider=res.provider,
            prompt=question,
            answer=res.answer,
            meta=res.meta,
        )
        if log_progress:
            alog(f"  - [stage1] {res.provider} done.")

    return [r for r in ordered_results if r is not None]


async def stage2_reviews_async(
    question: str,
    opinions: List[AdvisorCall],
    reviewer_names: List[str],
    cfg: AdvisorsConfig,
    limiter: ConcurrencyLimiter,
    *,
    log_progress: bool = True,
    cancel_event: Optional[asyncio.Event] = None,
    progress_handler: Optional[Callable[[ProgressEvent], None]] = None,
    turn_index: int = 1,
) -> List[ReviewCall]:
    review_prompt = _build_review_prompt(question, opinions)
    tasks: List[tuple[int, asyncio.Task[ProviderResult]]] = []

    for idx, name in enumerate(reviewer_names):
        if log_progress:
            alog(f"  - [stage2] asking {name} for review ...")
        task = asyncio.create_task(
            _call_provider_async(
                name,
                review_prompt,
                cfg,
                limiter,
                cancel_event=cancel_event,
                progress_handler=progress_handler,
                stage="stage2",
                turn_index=turn_index,
            )
        )
        tasks.append((idx, task))

    ordered_results: List[Optional[ReviewCall]] = [None] * len(tasks)
    for idx, task in tasks:
        res = await task
        ordered_results[idx] = ReviewCall(
            provider=res.provider,
            prompt=review_prompt,
            raw_review=res.answer,
        )
        if log_progress:
            alog(f"  - [stage2] {res.provider} review done.")

    return [r for r in ordered_results if r is not None]


async def stage3_chairman_async(
    question: str,
    opinions: List[AdvisorCall],
    reviews: List[ReviewCall],
    chairman: str,
    cfg: AdvisorsConfig,
    limiter: ConcurrencyLimiter,
    *,
    log_progress: bool = True,
    cancel_event: Optional[asyncio.Event] = None,
    progress_handler: Optional[Callable[[ProgressEvent], None]] = None,
    turn_index: int = 1,
) -> ChairmanCall:
    prompt = _build_chairman_prompt(question, opinions, reviews)
    if log_progress:
        alog(f"  - [stage3] asking chairman '{chairman}' ...")
    res = await _call_provider_async(
        chairman,
        prompt,
        cfg,
        limiter,
        cancel_event=cancel_event,
        progress_handler=progress_handler,
        stage="stage3",
        turn_index=turn_index,
    )
    if log_progress:
        alog("  - [stage3] chairman done.")
    return ChairmanCall(provider=res.provider, prompt=prompt, answer=res.answer)


# Typed forward reference for build_turn_prompt
@dataclass
class TurnRecord:
    turn_index: int
    opinions: List[AdvisorCall]
    reviews: List[ReviewCall]
    chairman: ChairmanCall
    turn_prompt: str
