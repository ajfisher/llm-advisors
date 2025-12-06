from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Dict, List

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from .advisors import ProgressEvent
from .config import AdvisorsConfig, load_config
from .conversation import generate_conversation_id, run_conversation
from .providers import discover_ollama_models

app = Flask(__name__)
app.secret_key = "llm-advisors"


class ProgressState:
    """Track per-stage, per-provider progress."""

    def __init__(self, members: List[str], chairman: str, turns: int):
        self.members = members
        self.chairman = chairman
        self.turns = turns
        self.turn = 1
        self.status = "running"
        self.messages: Dict[str, str] = {}
        self.stage_status: Dict[tuple[int, str, str], str] = {}
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
                self.stage_status[(turn, stage, provider)] = "pending"

    def handle(self, event: ProgressEvent) -> None:
        if event.event == "turn" and event.status == "start":
            self.turn = event.turn
            self._init_turn(event.turn)
        elif event.event == "provider" and event.provider:
            key = (event.turn, event.stage, event.provider)
            self.stage_status[key] = event.status or "pending"
            if event.message:
                self.messages[f"{event.stage}:{event.provider}"] = event.message
        elif event.event == "conversation":
            self.status = "done" if event.status == "done" else event.status or "running"

    def snapshot(self) -> Dict[str, object]:
        def stage_map(stage: str) -> Dict[str, str]:
            return {
                provider: self.stage_status.get(
                    (self.turn, stage, provider), "pending"
                )
                for provider in self._providers_for_stage(stage)
            }

        return {
            "status": self.status,
            "turn": self.turn,
            "turns": self.turns,
            "stage1": stage_map("stage1"),
            "stage2": stage_map("stage2"),
            "stage3": stage_map("stage3"),
            "messages": self.messages,
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
    options = {"codex", "claude", "gemini"}
    options.update([m for m in cfg.members if not m.startswith("ollama/") and m != "ollama"])
    options.update([cfg.chairman] if cfg.chairman not in ("ollama",) else [])

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

    return sorted(options)


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


@app.route("/", methods=["GET"])
def home():
    cfg = load_config()
    members = _available_members(cfg)
    return render_template(
        "home.html",
        members=members,
        default_members=cfg.members,
        default_chairman=cfg.chairman if cfg.chairman in members else members[0],
        default_turns=1,
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
                conversations.append({"id": conv_dir.name, "meta": meta})

    # include running jobs without meta yet
    for conv_id, job in jobs.items():
        if not any(c["id"] == conv_id for c in conversations):
            conversations.append(
                {
                    "id": conv_id,
                    "meta": {
                        "question": job.question,
                        "members": job.members,
                        "chairman": job.chairman,
                        "turns": job.turns,
                        "created_at": "",
                        "status": job.status,
                    },
                }
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
    chairman = meta["chairman"] if meta else (job.chairman if job else "")
    turns_count = meta["turns"] if meta else (job.turns if job else 0)

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
        members = meta.get("members", [])
        chairman = meta.get("chairman", "")
        stage1 = {m: "done" for m in members}
        stage2 = {m: "done" for m in members}
        stage3 = {chairman: "done"} if chairman else {}
        status_val = "error" if meta.get("error") else "done"
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
            }
        )

    abort(404)


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
    members = request.form.getlist("members") or cfg.members
    chairman = (request.form.get("chairman") or cfg.chairman).strip()
    try:
        turns = int(request.form.get("turns", "1"))
    except ValueError:
        turns = 1
    turns = max(1, turns)

    if not question:
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
