from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from .advisors import ProgressEvent
from .config import AdvisorsConfig, load_config
from .conversation import generate_conversation_id, run_conversation
from .providers import discover_codex_models, discover_gemini_models, discover_ollama_models
import markdown

app = Flask(__name__)
app.secret_key = "llm-advisors"

NONE_ADVISOR_VALUE = "__none__"
BARE_WEB_PROVIDERS = {"codex", "claude", "gemini", "ollama"}
PREFERRED_ADVISOR_OPTIONS = [
    ("codex/gpt-5.2",),
    ("gemini/gemini-2.5-flash",),
    ("codex/gpt-5.4",),
    ("ollama/gemma4:latest", "ollama/gemma4:26b", "ollama/gemma4"),
]
PREFERRED_CHAIR_OPTIONS = ("codex/gpt-5.5",)
TURN_KIND_LABELS = {
    "baseline": "Baseline",
    "divergence": "Divergence",
    "task_solve": "Task Solve",
    "convergence": "Convergence",
}
STAGE_LABELS = {
    "advise": "Advise",
    "review": "Review",
    "chair": "Chair",
}
STAGE_TO_UI_STAGE = {
    "stage1": "advise",
    "stage2": "review",
    "stage3": "chair",
}


def _turn_kind_for_index(turn_index: int, turns: int) -> str:
    if turn_index == 1:
        return "baseline"
    if turn_index == turns:
        return "convergence"
    if turn_index == 2:
        return "divergence"
    return "task_solve"


class ProgressState:
    """Track per-stage, per-provider progress."""

    def __init__(self, members: List[str], chairman: str, turns: int):
        self.members = members
        self.chairman = chairman
        self.turns = turns
        self.turn = 1
        self.status = "running"
        self.messages: Dict[str, str] = {}
        self.stage_status: Dict[tuple[int, str, str], Dict[str, object]] = {}
        self.roles: Dict[str, str] = {}
        self._init_turn(self.turn)

    def _providers_for_stage(self, stage: str) -> List[str]:
        if stage in ("stage1", "stage2"):
            return self.members
        if stage == "stage3":
            return [self.chairman]
        return []

    def _init_turn(self, turn: int) -> None:
        for stage in ("stage1", "stage2", "stage3"):
            for provider in self._providers_for_stage(stage):
                self.stage_status.setdefault((turn, stage, provider), {"status": "pending"})

    def handle(self, event: ProgressEvent) -> None:
        if event.event == "turn" and event.status == "start":
            self.turn = event.turn
            self._init_turn(event.turn)
            self.roles = {}
            if event.message:
                try:
                    roles = json.loads(event.message)
                    if isinstance(roles, dict):
                        self.roles = {str(k): str(v) for k, v in roles.items()}
                except Exception:
                    pass
        elif event.event == "provider" and event.provider:
            key = (event.turn, event.stage, event.provider)
            current = self.stage_status.get(key, {})
            self.stage_status[key] = {
                "status": event.status or "pending",
                "message": event.message,
                "started_at": current.get("started_at") if event.status != "running" else time.monotonic(),
                "duration_seconds": event.duration_seconds,
            }
            if event.message:
                self.messages[f"{event.stage}:{event.provider}"] = event.message
        elif event.event == "conversation":
            self.status = "done" if event.status == "done" else event.status or "running"

    def _status_item(self, turn: int, stage: str, provider: str, now: float) -> dict:
        item = self.stage_status.get((turn, stage, provider), {"status": "pending"}).copy()
        if item.get("status") == "running" and item.get("started_at"):
            item["duration_seconds"] = now - float(item["started_at"])
        item.pop("started_at", None)
        return item

    def _live_calls(self, now: float) -> List[dict]:
        calls = []
        for turn in range(1, self.turns + 1):
            for stage in ("stage1", "stage2", "stage3"):
                for provider in self._providers_for_stage(stage):
                    item = self._status_item(turn, stage, provider, now)
                    seconds = _duration_seconds(item.get("duration_seconds"))
                    if seconds is None:
                        continue
                    calls.append(
                        {
                            "turn": turn,
                            "stage": STAGE_TO_UI_STAGE[stage],
                            "provider": provider,
                            "seconds": seconds,
                            "status": item.get("status", "pending"),
                        }
                    )
        return calls

    def _live_stats(self, now: float) -> dict:
        calls = self._live_calls(now)
        stage_totals: Dict[str, dict] = {
            stage: {"seconds": 0.0, "count": 0, "display": "n/a"}
            for stage in STAGE_LABELS
        }
        for call in calls:
            stage = str(call["stage"])
            stage_totals[stage]["seconds"] += float(call["seconds"])
            stage_totals[stage]["count"] += 1

        for stage in stage_totals:
            seconds = stage_totals[stage]["seconds"]
            stage_totals[stage]["display"] = _format_duration(
                seconds if stage_totals[stage]["count"] else None
            )

        total_seconds = sum(float(call["seconds"]) for call in calls)
        fastest = min(calls, key=lambda item: float(item["seconds"])) if calls else None
        slowest = max(calls, key=lambda item: float(item["seconds"])) if calls else None
        done_count = sum(1 for call in calls if call.get("status") == "done")
        active_count = sum(1 for call in calls if call.get("status") == "running")

        return {
            "call_count": len(calls),
            "done_count": done_count,
            "active_count": active_count,
            "total_model_seconds": total_seconds,
            "total_model_display": _format_duration(total_seconds if calls else None),
            "average_display": _format_duration(total_seconds / len(calls) if calls else None),
            "fastest_call": fastest,
            "fastest_display": _format_duration(float(fastest["seconds"]) if fastest else None),
            "slowest_call": slowest,
            "slowest_display": _format_duration(float(slowest["seconds"]) if slowest else None),
            "stage_totals": stage_totals,
        }

    def _live_timeline(self, now: float) -> List[dict]:
        timeline = []
        expected_per_turn = (len(self.members) * 2) + (1 if self.chairman else 0)

        for turn in range(1, self.turns + 1):
            items = []
            for stage in ("stage1", "stage2", "stage3"):
                for provider in self._providers_for_stage(stage):
                    items.append(self._status_item(turn, stage, provider, now))

            statuses = [str(item.get("status", "pending")) for item in items]
            completed_count = sum(
                1 for status in statuses if status in {"done", "error", "cancelled"}
            )
            running_count = sum(1 for status in statuses if status == "running")
            known_seconds = [
                seconds
                for item in items
                if (seconds := _duration_seconds(item.get("duration_seconds"))) is not None
            ]
            total_seconds = sum(known_seconds)

            if any(status == "error" for status in statuses):
                status = "error"
            elif any(status == "cancelled" for status in statuses):
                status = "cancelled"
            elif completed_count == expected_per_turn and expected_per_turn:
                status = "done"
            elif running_count or completed_count:
                status = "running"
            else:
                status = "pending"

            kind = _turn_kind_for_index(turn, self.turns)
            timeline.append(
                {
                    "index": turn,
                    "kind": kind,
                    "label": TURN_KIND_LABELS.get(kind, "Turn"),
                    "status": status,
                    "completed_count": completed_count,
                    "response_count": expected_per_turn,
                    "duration_display": _format_duration(total_seconds if known_seconds else None),
                }
            )

        return timeline

    def snapshot(self) -> Dict[str, object]:
        now = time.monotonic()

        def stage_map(stage: str) -> Dict[str, object]:
            result = {}
            for provider in self._providers_for_stage(stage):
                result[provider] = self._status_item(self.turn, stage, provider, now)
            return result

        return {
            "status": self.status,
            "turn": self.turn,
            "turns": self.turns,
            "stage1": stage_map("stage1"),
            "stage2": stage_map("stage2"),
            "stage3": stage_map("stage3"),
            "messages": self.messages,
            "roles": self.roles,
            "stats": self._live_stats(now),
            "timeline": self._live_timeline(now),
        }


class ConversationJob:
    def __init__(self, conversation_id: str, question: str, members: List[str], chairman: str, turns: int, cfg: AdvisorsConfig):
        self.conversation_id = conversation_id
        self.question = question
        self.members = members
        self.chairman = chairman
        self.turns = turns
        self.cfg = cfg
        self.state = ProgressState(members, chairman, turns)
        self.cancel = threading.Event()
        self.status = "running"
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.cancel.set()

    def _run(self) -> None:
        try:
            run_conversation(
                self.question,
                self.members,
                self.chairman,
                self.turns,
                self.cfg,
                log_enabled=True,
                show_progress=False,
                progress_handler=self.state.handle,
                cancel_token=self.cancel,
                conversation_id=self.conversation_id,
            )
            if self.cancel.is_set():
                self.status = "cancelled"
            else:
                self.status = "done"
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)
            self.status = "cancelled" if self.cancel.is_set() else "error"

    def snapshot(self) -> Dict[str, object]:
        snap = self.state.snapshot()
        snap["job_status"] = self.status
        snap["error"] = self.error
        return snap


jobs: Dict[str, ConversationJob] = {}


def _available_members(cfg: AdvisorsConfig) -> List[str]:
    options = set()
    options.update([m for m in cfg.members if m not in BARE_WEB_PROVIDERS])
    if cfg.chairman and cfg.chairman not in BARE_WEB_PROVIDERS:
        options.add(cfg.chairman)
    for candidates in PREFERRED_ADVISOR_OPTIONS:
        options.add(candidates[0])
    for candidate in PREFERRED_CHAIR_OPTIONS:
        options.add(candidate)

    for model in discover_codex_models(cfg):
        options.add(f"codex/{model}")

    for model in discover_gemini_models(cfg):
        options.add(f"gemini/{model}")

    # Dynamically discover Ollama models and add as ollama/<model>
    ollama_models = discover_ollama_models(cfg)
    for model in ollama_models:
        options.add(f"ollama/{model}")

    # Honour explicitly configured ollama/<model> entries
    for name in cfg.members:
        if name.startswith("ollama/"):
            options.add(name)
    if cfg.chairman.startswith("ollama/"):
        options.add(cfg.chairman)

    return sorted(options, key=_member_sort_key)


def _member_sort_key(member: str) -> tuple[int, str]:
    order = {"codex": 0, "claude": 1, "gemini": 2, "ollama": 3}
    base = member.split("/", 1)[0]
    return (order.get(base, 99), member)


def _resolve_member_option(member: str, options: List[str]) -> str | None:
    if member in options:
        return member
    if member in BARE_WEB_PROVIDERS:
        prefix = f"{member}/"
        return next((option for option in options if option.startswith(prefix)), None)
    return None


def _resolve_preferred_option(candidates: tuple[str, ...], options: List[str]) -> str | None:
    for candidate in candidates:
        if candidate in options:
            return candidate
    for candidate in candidates:
        if candidate.startswith("ollama/"):
            prefix = candidate.rsplit(":", 1)[0]
            match = next((option for option in options if option.startswith(prefix)), None)
            if match:
                return match
    return None


def _default_advisor_slots(cfg: AdvisorsConfig, options: List[str]) -> List[str | None]:
    selected: List[str] = []
    for candidates in PREFERRED_ADVISOR_OPTIONS:
        option = _resolve_preferred_option(candidates, options)
        if option and option not in selected:
            selected.append(option)

    for member in cfg.members:
        option = _resolve_member_option(member, options)
        if option and option not in selected:
            selected.append(option)
        if len(selected) == 4:
            break

    if not selected and options:
        selected.append(options[0])

    return (selected + [None] * 4)[:4]


def _default_chair(cfg: AdvisorsConfig, options: List[str], advisor_slots: List[str | None]) -> str:
    preferred = _resolve_preferred_option(PREFERRED_CHAIR_OPTIONS, options)
    if preferred:
        return preferred
    option = _resolve_member_option(cfg.chairman, options)
    if option:
        return option
    return next((slot for slot in advisor_slots if slot), options[0] if options else "")


def _advisor_members_from_form(options: List[str]) -> List[str]:
    valid_options = set(options)
    members: List[str] = []
    for idx in range(1, 5):
        value = (request.form.get(f"advisor_{idx}") or "").strip()
        if not value or value == NONE_ADVISOR_VALUE or value not in valid_options:
            continue
        if value not in members:
            members.append(value)
    return members[:4]


def _load_meta(conv_dir: Path) -> dict | None:
    meta_file = conv_dir / "meta.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_turns(conv_dir: Path) -> List[dict]:
    turns: List[dict] = []
    for path in sorted(conv_dir.glob("turn-*.json")):
        try:
            turns.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return turns


def _duration_seconds(value: object) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    rounded = int(round(seconds))
    if rounded < 60:
        return f"{rounded}s"
    minutes, remainder = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _provider_family(provider: str) -> str:
    return (provider or "").split("/", 1)[0] or "model"


def _format_created_at(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%d %b %Y, %H:%M")


def _response_calls(turns: List[dict]) -> List[dict]:
    calls: List[dict] = []
    for turn in turns:
        turn_index = turn.get("turn_index", "")
        for advisor in turn.get("advisors", []):
            seconds = _duration_seconds(advisor.get("duration_seconds"))
            if seconds is not None:
                calls.append(
                    {
                        "turn": turn_index,
                        "stage": "advise",
                        "provider": advisor.get("provider", ""),
                        "seconds": seconds,
                    }
                )
        for review in turn.get("reviews", []):
            seconds = _duration_seconds(review.get("duration_seconds"))
            if seconds is not None:
                calls.append(
                    {
                        "turn": turn_index,
                        "stage": "review",
                        "provider": review.get("provider", ""),
                        "seconds": seconds,
                    }
                )
        chairman = turn.get("chairman") or {}
        seconds = _duration_seconds(chairman.get("duration_seconds"))
        if seconds is not None:
            calls.append(
                {
                    "turn": turn_index,
                    "stage": "chair",
                    "provider": chairman.get("provider", ""),
                    "seconds": seconds,
                }
            )
    return calls


def _conversation_stats(turns: List[dict]) -> dict:
    calls = _response_calls(turns)
    stage_totals: Dict[str, dict] = {
        stage: {"seconds": 0.0, "count": 0, "display": "n/a"}
        for stage in STAGE_LABELS
    }
    families: Dict[str, int] = {}

    for call in calls:
        stage = str(call["stage"])
        stage_totals[stage]["seconds"] += float(call["seconds"])
        stage_totals[stage]["count"] += 1
        family = _provider_family(str(call["provider"]))
        families[family] = families.get(family, 0) + 1

    for stage in stage_totals:
        seconds = stage_totals[stage]["seconds"]
        stage_totals[stage]["display"] = _format_duration(
            seconds if stage_totals[stage]["count"] else None
        )

    total_seconds = sum(float(call["seconds"]) for call in calls)
    fastest = min(calls, key=lambda item: float(item["seconds"])) if calls else None
    slowest = max(calls, key=lambda item: float(item["seconds"])) if calls else None

    return {
        "call_count": len(calls),
        "total_model_seconds": total_seconds,
        "total_model_display": _format_duration(total_seconds if calls else None),
        "average_display": _format_duration(total_seconds / len(calls) if calls else None),
        "fastest_call": fastest,
        "fastest_display": _format_duration(float(fastest["seconds"]) if fastest else None),
        "slowest_call": slowest,
        "slowest_display": _format_duration(float(slowest["seconds"]) if slowest else None),
        "stage_totals": stage_totals,
        "families": families,
    }


def _turn_summaries(turns: List[dict]) -> List[dict]:
    summaries: List[dict] = []
    for turn in turns:
        calls = _response_calls([turn])
        total_seconds = sum(float(call["seconds"]) for call in calls)
        advisors = len(turn.get("advisors", []))
        reviews = len(turn.get("reviews", []))
        chair_count = 1 if turn.get("chairman") else 0
        kind = str(turn.get("kind") or "")
        summaries.append(
            {
                "index": turn.get("turn_index"),
                "kind": kind,
                "label": TURN_KIND_LABELS.get(kind, kind.replace("_", " ").title() or "Turn"),
                "duration_display": _format_duration(total_seconds if calls else None),
                "response_count": advisors + reviews + chair_count,
            }
        )
    return summaries


def _turn_views(turns: List[dict]) -> List[dict]:
    summaries = _turn_summaries(turns)
    return [
        {"turn": turn, "summary": summary}
        for turn, summary in zip(turns, summaries)
    ]


def _conversation_list_item(
    conversation_id: str,
    meta: dict,
    turns: List[dict] | None = None,
) -> dict:
    turns = turns or []
    stats = _conversation_stats(turns)
    status = "error" if meta.get("error") else meta.get("status", "done")
    return {
        "id": conversation_id,
        "meta": meta,
        "status": status,
        "created_at_display": _format_created_at(meta.get("created_at")),
        "question": meta.get("question", ""),
        "members": meta.get("members", []),
        "chair": meta.get("chair") or meta.get("chairman", ""),
        "turns_count": meta.get("turns", len(turns)),
        "stats": stats,
    }


@app.route("/", methods=["GET"])
def home():
    cfg = load_config()
    members = _available_members(cfg)
    advisor_slots = _default_advisor_slots(cfg, members)
    return render_template(
        "home.html",
        members=members,
        advisor_slots=advisor_slots,
        none_advisor_value=NONE_ADVISOR_VALUE,
        default_chair=_default_chair(cfg, members, advisor_slots),
        default_turns=1,
        thinking_enabled=cfg.thinking_enabled,
    )


@app.route("/conversations", methods=["GET"])
def list_conversations():
    cfg = load_config()
    base_dir = cfg.logging.base_dir
    conversations: List[dict] = []

    if base_dir.exists():
        for conv_dir in sorted(base_dir.iterdir(), reverse=True):
            if not conv_dir.is_dir():
                continue
            meta = _load_meta(conv_dir)
            if meta:
                conversations.append(
                    _conversation_list_item(conv_dir.name, meta, _load_turns(conv_dir))
                )

    # include running jobs without meta yet
    for conv_id, job in jobs.items():
        if not any(c["id"] == conv_id for c in conversations):
            conversations.append(
                _conversation_list_item(
                    conv_id,
                    {
                        "question": job.question,
                        "members": job.members,
                        "chair": job.chairman,
                        "chairman": job.chairman,
                        "turns": job.turns,
                        "created_at": "",
                        "status": job.status,
                    },
                )
            )

    return render_template("conversations.html", conversations=conversations)


@app.route("/conversations/<conversation_id>", methods=["GET"])
def conversation_detail(conversation_id: str):
    cfg = load_config()
    conv_dir = cfg.logging.base_dir / conversation_id
    job = jobs.get(conversation_id)
    meta = _load_meta(conv_dir) if conv_dir.exists() else None
    turns = _load_turns(conv_dir) if conv_dir.exists() else []

    if meta is None and job is None:
        abort(404)

    question = meta["question"] if meta else (job.question if job else "")
    members = meta["members"] if meta else (job.members if job else [])
    chairman = (meta.get("chair") or meta.get("chairman")) if meta else (job.chairman if job else "")
    turns_count = meta["turns"] if meta else (job.turns if job else 0)
    final_answer_html = None
    final_answer_text = ""
    if turns:
        final_answer_text = turns[-1].get("chairman", {}).get("answer", "")
        final_answer_html = markdown.markdown(final_answer_text, extensions=["extra", "sane_lists"])

    return render_template(
        "conversation_detail.html",
        conversation_id=conversation_id,
        meta=meta,
        turns=turns,
        job=job,
        question=question,
        members=members,
        chairman=chairman,
        turns_count=turns_count,
        final_answer_html=final_answer_html,
        final_answer_text=final_answer_text,
        stats=_conversation_stats(turns),
        turn_summaries=_turn_summaries(turns),
        turn_views=_turn_views(turns),
        turn_kind_labels=TURN_KIND_LABELS,
        stage_labels=STAGE_LABELS,
    )


@app.route("/conversations/<conversation_id>/status", methods=["GET"])
def conversation_status(conversation_id: str):
    cfg = load_config()
    conv_dir = cfg.logging.base_dir / conversation_id
    job = jobs.get(conversation_id)

    if job:
        return jsonify(job.snapshot())

    # Fallback for completed conversations without an active job
    meta = _load_meta(conv_dir)
    if meta:
        turns = _load_turns(conv_dir)
        members = meta.get("members", [])
        chairman = meta.get("chair") or meta.get("chairman", "")
        stage1 = {m: {"status": "done"} for m in members}
        stage2 = {m: {"status": "done"} for m in members}
        stage3 = {chairman: {"status": "done"}} if chairman else {}
        status_val = "error" if meta.get("error") else "done"
        timeline = [
            {
                **summary,
                "status": "done",
                "completed_count": summary["response_count"],
            }
            for summary in _turn_summaries(turns)
        ]
        return jsonify(
            {
                "status": status_val,
                "job_status": status_val,
                "turn": meta.get("turns", 1),
                "turns": meta.get("turns", 1),
                "stage1": stage1,
                "stage2": stage2,
                "stage3": stage3,
                "messages": {},
                "stats": _conversation_stats(turns),
                "timeline": timeline,
            }
        )

    abort(404)


@app.route("/markdown", methods=["POST"])
def render_markdown():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str):
        text = ""
    html = markdown.markdown(text, extensions=["extra", "sane_lists"])
    return jsonify({"html": html})


@app.route("/conversations/<conversation_id>/stop", methods=["POST"])
def stop_conversation(conversation_id: str):
    job = jobs.get(conversation_id)
    if job and job.status == "running":
        job.stop()
        return jsonify({"ok": True, "status": "stopping"})
    return jsonify({"ok": False}), 404


@app.route("/conversations/<conversation_id>/delete", methods=["POST"])
def delete_conversation(conversation_id: str):
    cfg = load_config()
    job = jobs.pop(conversation_id, None)
    if job and job.status == "running":
        job.stop()
    conv_dir = cfg.logging.base_dir / conversation_id
    if conv_dir.exists():
        shutil.rmtree(conv_dir, ignore_errors=True)
    return redirect(url_for("list_conversations"))


@app.route("/conversations", methods=["POST"])
def start_conversation():
    cfg = load_config()
    cfg.logging.enabled = True  # ensure artefacts are available for the web UI

    question = (request.form.get("question") or "").strip()
    available_members = _available_members(cfg)
    members = _advisor_members_from_form(available_members)
    advisor_slots = _default_advisor_slots(cfg, available_members)
    chairman = (request.form.get("chair") or request.form.get("chairman") or "").strip()
    if chairman not in available_members:
        chairman = _default_chair(cfg, available_members, advisor_slots)
    try:
        turns = int(request.form.get("turns", "1"))
    except ValueError:
        turns = 1
    turns = max(1, turns)
    cfg.thinking_enabled = request.form.get("thinking_enabled") == "on"

    if not question or not members:
        return redirect(url_for("home"))

    conversation_id = generate_conversation_id()
    job = ConversationJob(conversation_id, question, members, chairman, turns, cfg)
    jobs[conversation_id] = job
    job.start()

    return redirect(url_for("conversation_detail", conversation_id=conversation_id))


def main() -> None:
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    main()
