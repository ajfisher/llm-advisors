from __future__ import annotations

import json
from pathlib import Path
from typing import List

from flask import Flask, abort, redirect, render_template, request, url_for

from .config import AdvisorsConfig, load_config
from .conversation import run_conversation

app = Flask(__name__)
app.secret_key = "llm-advisors"


def _available_members(cfg: AdvisorsConfig) -> List[str]:
    options = {"codex", "claude", "gemini", "ollama"}
    options.update(cfg.members)
    options.add(cfg.chairman)
    for name, pcfg in cfg.providers.items():
        if name == "ollama" and pcfg.model:
            options.add(f"ollama/{pcfg.model}")
        else:
            options.add(name)
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

    return render_template("conversations.html", conversations=conversations)


@app.route("/conversations/<conversation_id>", methods=["GET"])
def conversation_detail(conversation_id: str):
    cfg = load_config()
    conv_dir = cfg.logging.base_dir / conversation_id
    if not conv_dir.exists():
        abort(404)

    meta = _load_meta(conv_dir)
    turns = _load_turns(conv_dir)
    if meta is None:
        abort(404)

    return render_template(
        "conversation_detail.html",
        conversation_id=conversation_id,
        meta=meta,
        turns=turns,
    )


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

    run = run_conversation(
        question,
        members,
        chairman,
        turns,
        cfg,
        log_enabled=True,
        show_progress=False,
    )
    return redirect(url_for("conversation_detail", conversation_id=run.conversation_id))


def main() -> None:
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    main()
