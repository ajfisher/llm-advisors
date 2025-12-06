from __future__ import annotations

import argparse
import sys
from typing import List

from .config import load_config
from .advisors import stage1_first_opinions, stage2_reviews, stage3_chairman
from .exceptions import ProviderError

def log(msg: str) -> None:
    """Log progress messages to stderr so stdout stays clean."""
    print(msg, file=sys.stderr, flush=True)

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        prog="llm-advisors",
        description="Multi-model advisors using Codex, Claude Code, Gemini CLI and Ollama",
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question / prompt to send to the council",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=cfg.members,
        help=(
            "Council members (default from config). "
            "Options: codex claude gemini ollama or ollama/<model> "
            "e.g. 'ollama/llama3.1:8b'"
        ),
    )
    parser.add_argument(
        "--chairman",
        default=cfg.chairman,
        help=(
            "Chairman provider for final synthesis (default from config). "
            "Can be codex, claude, gemini, ollama or ollama/<model>."
        ),
    )
    parser.add_argument(
        "--show-intermediate",
        action="store_true",
        help="Print stage 1 and stage 2 outputs before the final answer.",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    cfg = load_config()
    args = _parse_args(argv)

    question = " ".join(args.question)
    members = args.members
    chairman = args.chairman

    try:
        # Stage 1: first opinions
        log(f"[1/3] Collecting first opinions from: {', '.join(members)} ...")
        opinions = stage1_first_opinions(question, members, cfg)
        log(f"[1/3] Done. Received {len(opinions)} opinions.")

        if args.show_intermediate:
            print("=== STAGE 1: FIRST OPINIONS ===\n")
            for i, o in enumerate(opinions, start=1):
                print(f"[{i}] {o.provider}")
                print(o.answer)
                print()

        # Stage 2: peer reviews
        log(f"[2/3] Requesting peer reviews from: {', '.join(members)} ...")
        reviews = stage2_reviews(question, opinions, members, cfg)
        log(f"[2/3] Done. Received {len(reviews)} reviews.")

        if args.show_intermediate:
            print("=== STAGE 2: PEER REVIEWS ===\n")
            for r in reviews:
                print(f"[Review by {r.provider}]")
                print(r.raw_review)
                print()

        # Stage 3: chairman synthesis
        log(f"[3/3] Asking chairman '{chairman}' to synthesize final answer ...")
        final_answer = stage3_chairman(question, opinions, reviews, chairman, cfg)
        log("[3/3] Final answer ready.\n")

        print("=== FINAL ANSWER ===\n")
        print(final_answer)

    except ProviderError as e:
        print(f"Provider error:\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

