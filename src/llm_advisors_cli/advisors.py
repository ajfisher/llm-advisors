from __future__ import annotations

import sys

from dataclasses import dataclass
from typing import List

from .config import AdvisorsConfig
from .providers import ProviderResult, ask_claude, ask_codex, ask_gemini, ask_ollama, get_provider_functions


@dataclass
class ReviewResult:
    provider: str
    raw_review: str

def alog(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def _parse_ollama_member(name: str) -> Optional[str]:
    """
    If name is like 'ollama/llama3.1:8b', return 'llama3.1:8b'.
    Otherwise return None.
    """
    if not name.startswith("ollama/"):
        return None
    return name.split("/", 1)[1]

def stage1_first_opinions(
    question: str,
    member_names: List[str],
    cfg: CouncilConfig,
) -> List[ProviderResult]:
    providers = get_provider_functions(cfg)
    results: List[ProviderResult] = []

    for name in member_names:
        fn = providers.get("ollama" if name.startswith("ollama/") else name)
        if not fn:
            continue

        alog(f"  - [stage1] asking {name} ...")

        if name == "ollama":
            res = ask_ollama(question, cfg)
        else:
            ollama_model = _parse_ollama_member(name)
            if ollama_model is not None:
                # e.g. name == 'ollama/llama3.1:8b'
                res = ask_ollama(question, cfg, model_override=ollama_model)
                # Preserve member name (so provider field is distinct)
                res.provider = name
            elif name == "codex":
                res = ask_codex(question, cfg)
            elif name == "claude":
                res = ask_claude(question, cfg)
            elif name == "gemini":
                res = ask_gemini(question, cfg)
            else:
                continue

        alog(f"  - [stage1] {name} done.")
        results.append(res)

    return results



def _build_review_prompt(question: str, opinions: List[ProviderResult]) -> str:
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


def stage2_reviews(
    question: str,
    opinions: List[ProviderResult],
    reviewer_names: List[str],
    cfg: CouncilConfig,
) -> List[ReviewResult]:
    providers = get_provider_functions(cfg)
    review_prompt = _build_review_prompt(question, opinions)
    reviews: List[ReviewResult] = []

    for name in reviewer_names:
        fn = providers.get("ollama" if name.startswith("ollama/") else name)
        if not fn:
            continue

        alog(f"  - [stage2] asking {name} for review ...")
        ollama_model = _parse_ollama_member(name)

        if name == "ollama":
            res = ask_ollama(review_prompt, cfg)
        elif ollama_model is not None:
            res = ask_ollama(review_prompt, cfg, model_override=ollama_model)
        elif name == "codex":
            res = ask_codex(review_prompt, cfg)
        elif name == "claude":
            res = ask_claude(review_prompt, cfg)
        elif name == "gemini":
            res = ask_gemini(review_prompt, cfg)
        else:
            continue

        alog(f"  - [stage2] {name} review done.")
        reviews.append(ReviewResult(provider=name, raw_review=res.answer))

    return reviews



def _build_chairman_prompt(
    question: str,
    opinions: List[ProviderResult],
    reviews: List[ReviewResult],
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

Structure your response as:

FINAL ANSWER:
<your answer>

NOTES:
- Key points of agreement
- Key points of disagreement and how you resolved them
""".strip()

def stage3_chairman(
    question: str,
    opinions: List[ProviderResult],
    reviews: List[ReviewResult],
    chairman: str,
    cfg: CouncilConfig,
) -> str:
    prompt = _build_chairman_prompt(question, opinions, reviews)

    ollama_model = _parse_ollama_member(chairman)

    if chairman == "ollama":
        res = ask_ollama(prompt, cfg)
    elif ollama_model is not None:
        res = ask_ollama(prompt, cfg, model_override=ollama_model)
    elif chairman == "codex":
        res = ask_codex(prompt, cfg)
    elif chairman == "claude":
        res = ask_claude(prompt, cfg)
    elif chairman == "gemini":
        res = ask_gemini(prompt, cfg)
    else:
        # Fallback: use the first opinion if chairman is unknown
        return opinions[0].answer

    return res.answer


