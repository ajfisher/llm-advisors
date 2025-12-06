from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .advisors import ProgressEvent
from .config import AdvisorsConfig, ProviderParallelismConfig, load_config
from .conversation import ConversationRun, run_conversation
from .providers import discover_ollama_models
from .exceptions import ProviderError


def log(msg: str) -> None:
    """Log progress messages to stderr so stdout stays clean."""
    print(msg, file=sys.stderr, flush=True)


class CLIProgressRenderer:
    """Render per-member statuses in-place for the CLI."""

    def __init__(self, members: List[str], chairman: str, turns: int):
        self.members = members
        self.chairman = chairman
        self.turns = turns
        self.turn = 1
        self.prev_lines = 0
        self.is_tty = sys.stderr.isatty()
        self.status: Dict[Tuple[int, str, str], str] = {}
        self._init_turn(1)

    def _init_turn(self, turn: int) -> None:
        self.turn = turn
        for stage in ("stage1", "stage2"):
            for member in self.members:
                self.status[(turn, stage, member)] = "pending"
        self.status[(turn, "stage3", self.chairman)] = "pending"

    def handle(self, event: ProgressEvent) -> None:
        if event.event == "turn" and event.status == "start":
            self._init_turn(event.turn)
            self._render()
            return

        if event.event == "provider" and event.provider:
            key = (event.turn, event.stage, event.provider)
            self.status[key] = event.status or "pending"
            self._render()
            return

        if event.event == "conversation" and event.status == "done":
            return

    def _render(self) -> None:
        lines: List[str] = []
        lines.append(f"Turn {self.turn}/{self.turns}")
        lines.append("Stage 1: Opinions")
        for member in self.members:
            lines.append(self._fmt_line(self.turn, "stage1", member))
        lines.append("Stage 2: Reviews")
        for member in self.members:
            lines.append(self._fmt_line(self.turn, "stage2", member))
        lines.append("Stage 3: Chairman")
        lines.append(self._fmt_line(self.turn, "stage3", self.chairman))

        block = "\n".join(lines) + "\n"

        if self.is_tty and self.prev_lines:
            # Move cursor up to rewrite previous block and clear it
            sys.stderr.write(f"\x1b[{self.prev_lines}F")
            sys.stderr.write("\x1b[0J")
        sys.stderr.write(block)
        sys.stderr.flush()
        self.prev_lines = len(lines)

    def _fmt_line(self, turn: int, stage: str, provider: str) -> str:
        status = self.status.get((turn, stage, provider), "pending")
        return f"  {provider:<20} {status}"


def _parse_args(cfg: AdvisorsConfig, argv: List[str] | None = None) -> argparse.Namespace:
    # Discover ollama models for better defaults
    ollama_models = discover_ollama_models(cfg)

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
        default=_default_members(cfg, ollama_models),
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
        "--turns",
        type=int,
        default=1,
        help="Number of council turns to run (default 1).",
    )
    parser.add_argument(
        "--show-turns",
        action="store_true",
        help="Print per-turn summaries after the final answer.",
    )
    parser.add_argument(
        "--show-intermediate",
        action="store_true",
        help="Print stage 1 and stage 2 outputs for each turn before the final answer.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=cfg.max_parallel,
        help="Global max parallel tasks (overrides config.general.max_parallel).",
    )
    parser.add_argument(
        "--ollama-parallel-mode",
        choices=["sequential", "limited", "parallel"],
        default=None,
        help="Override Ollama parallel mode (sequential | limited | parallel).",
    )
    parser.add_argument(
        "--ollama-max-parallel",
        type=int,
        default=None,
        help="Override Ollama max parallel when mode=limited.",
    )
    parser.add_argument(
        "--log-disabled",
        action="store_true",
        help="Disable writing conversation artefacts to disk.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to store conversation artefacts (overrides config).",
    )

    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: AdvisorsConfig, args: argparse.Namespace) -> AdvisorsConfig:
    cfg.max_parallel = max(1, args.max_parallel)

    ollama_parallel = cfg.parallelism.get("ollama", ProviderParallelismConfig())
    if args.ollama_parallel_mode:
        ollama_parallel.mode = args.ollama_parallel_mode
    if args.ollama_max_parallel is not None:
        ollama_parallel.max_parallel = max(1, args.ollama_max_parallel)
    cfg.parallelism["ollama"] = ollama_parallel

    if args.log_dir:
        cfg.logging.base_dir = args.log_dir
    if args.log_disabled:
        cfg.logging.enabled = False

    return cfg


def _default_members(cfg: AdvisorsConfig, ollama_models: List[str]) -> List[str]:
    members: List[str] = []
    for m in cfg.members:
        if m == "ollama":
            if ollama_models:
                members.extend([f"ollama/{model}" for model in ollama_models])
            else:
                members.append("ollama")
        else:
            members.append(m)
    return members


def _print_turn_summary(turn, total_turns: int) -> None:
    print(f"=== TURN {turn.turn_index}/{total_turns} ===\n")
    print("=== STAGE 1: FIRST OPINIONS ===\n")
    for i, opinion in enumerate(turn.opinions, start=1):
        print(f"[{i}] {opinion.provider}")
        print(opinion.answer)
        print()

    print("=== STAGE 2: PEER REVIEWS ===\n")
    for review in turn.reviews:
        print(f"[Review by {review.provider}]")
        print(review.raw_review)
        print()

    print("=== STAGE 3: CHAIRMAN ===\n")
    print(f"[{turn.chairman.provider}]")
    print(turn.chairman.answer)
    print()


def _print_turn_overview(run: ConversationRun) -> None:
    print("\n=== TURN OVERVIEW ===\n")
    for turn in run.turns:
        print(f"Turn {turn.turn_index}:")
        print(f"- Chairman ({turn.chairman.provider}) answer:\n{turn.chairman.answer}\n")


def main(argv: List[str] | None = None) -> None:
    cfg = load_config()
    args = _parse_args(cfg, argv)
    cfg = _apply_cli_overrides(cfg, args)

    question = " ".join(args.question)
    members = args.members
    chairman = args.chairman
    if args.turns > 4:
        log("Requested turns exceed 4; capping at 4 per protocol.")
    turns = max(1, min(args.turns, 4))

    renderer = CLIProgressRenderer(members, chairman, turns)

    try:
        run = run_conversation(
            question,
            members,
            chairman,
            turns,
            cfg,
            log_dir=args.log_dir,
            log_enabled=not args.log_disabled,
            show_progress=False,
            progress_handler=renderer.handle,
        )

        if args.show_intermediate:
            for turn in run.turns:
                _print_turn_summary(turn, turns)

        print("=== FINAL ANSWER ===\n")
        print(run.turns[-1].chairman.answer if run.turns else "")

        if args.show_turns:
            _print_turn_overview(run)

    except ProviderError as e:
        print(f"Provider error:\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
