# llm-advisors

A small CLI that lets you run a "council of LLMs as advisors" using:

- Codex (OpenAI CLI)
- Claude Code
- Gemini CLI
- Ollama (local models)

It **does not** use any extra API keys – it shells out to the official CLIs, which
authenticate using your existing subscriptions/accounts, and to Ollama for local models.

Vibe coded in an afternoon with codex.

The behaviour is inspired by Karpathy’s [llm-council](https://github.com/karpathy/llm-council), but instead of going via OpenRouter/API keys, it coordinates multiple CLIs you already use.

The tool **does not manage API keys**. It shells out to the official CLIs, which use your existing accounts/subscriptions, and to Ollama for local models.

## 1. What it does

Given a question/prompt, `llm-council`:

1. **Stage 1 – first opinions**
   Sends the prompt to each configured provider and collects all answers.

2. **Stage 2 – peer review**
   Builds a review prompt containing the original question and all answers (labelled A/B/C/…).
   Each provider gets this prompt and returns a review/ranking and an improved answer.

3. **Stage 3 – chairman synthesis**
   A selected “chairman” provider gets:
   - the original question
   - all first-round answers
   - all reviews
   and produces a single final answer plus some notes.

All the coordination logic lives in this repo. The actual inference is done by the CLIs and Ollama.

---

## 2. Requirements

You’ll need:

- **Python**: 3.11 or newer
- A working install of the following CLIs (pick the ones you care about):
  - `codex` (OpenAI / ChatGPT CLI)
  - `claude` (Claude Code CLI)
  - `gemini` (Gemini CLI)
  - `ollama` (local LLM runtime)

Each CLI must already be:

- Installed on your `$PATH`
- Logged in / configured to use whatever account or plan you have
- Working from the shell by itself (e.g. `codex "hello"`, `claude -p "hello"`, `gemini -p "hello"`, `ollama run llama3.2 "hello"`)

For Ollama, you’ll also need at least one model pulled, for example:

```bash
ollama pull llama3.2
```


