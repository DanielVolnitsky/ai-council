# AI Council

A multimodel deliberation tool that sends your question to N language models from different providers in parallel, then synthesizes their responses into a structured analysis showing consensus, disagreements, and unique insights.

## How It Works

1. User submits a question (via CLI or web UI)
2. The question is sent to all configured models in parallel (async)
3. Each model's response streams back in real-time
4. A configurable synthesizer model analyzes all responses and produces a structured document

## Synthesis Output Structure

- **Summary** — overall synthesis of all responses
- **Consensus** — points all models agree on
- **Disagreements** — where models differ, with attribution
- **Strongest / Weakest** — verdict on best and worst response with justification
- **Unique Insights** — novel points raised by only one model
- **Blind Spots** — gaps no model addressed
- **Actionable Takeaways** — concrete next steps (when applicable)

## Architecture

Monorepo with a LangGraph backend sharing a common core:

```
core/              # Shared: models, config, synthesis prompt
backends/
  langgraph/       # Provider fan-out; LangGraph orchestration planned
frontend/          # React (Vite) SPA — not built yet
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Vite |
| Backend | FastAPI (Python) |
| Provider abstraction | LangChain (`init_chat_model`) |
| Orchestration | LangGraph _(planned)_ |
| Streaming | SSE (Server-Sent Events) |
| Config | YAML |
| Tracing | Langfuse (self-hosted, Docker) |
| Package manager | uv |

### API

Client-agnostic REST API designed for reuse by future clients (Telegram bot, mobile app, etc.):

- `POST /api/council/ask` — synchronous, returns full JSON result
- `POST /api/council/ask/stream` — SSE, streams council member tokens live

Both take `{"question": "..."}`. The stream emits four event types:

| Event | Payload |
|-------|---------|
| `model_token` | `{"model_id", "token"}` — one per chunk, all members interleaved by arrival |
| `model_done` | `{"model_id", "response", "error"}` — one per member, `error` set if it failed |
| `synth_done` | `{"synthesis"}` — the complete `CouncilSynthesis` |
| `error` | `{"message"}` — the synthesizer failed, or every member did |

A client can rebuild the whole sync `CouncilResult` from the `model_done` events
plus `synth_done`; the `model_token` events exist only so a UI can fill panels live.
There is deliberately no `synth_token` event: synthesis uses structured output, so
the synthesizer streams raw JSON argument fragments rather than prose, and forwarding
those gives a client nothing it can render. The synthesis therefore arrives whole,
after a pause.

Neither endpoint takes anything but the question — the synthesizer is whatever
`default_synthesizer` in `config.yaml` names, with no per-request override, so a
round's composition is a config decision rather than a caller's. On `ask`, a
synthesizer that fails — or a round where every council member failed — returns 502;
on `ask/stream` the response headers are already sent by then, so the same failure
arrives as a final `error` event over a 200.

### Supported Providers

- OpenAI (GPT-4o+)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (local models)

## Design Decisions

- **Parallel async execution** — all models queried concurrently; partial failures are tolerated (continue with available responses)
- **Configurable synthesizer** — any model can be the synthesizer, user picks via config
- **LangChain for provider abstraction** — `init_chat_model` resolves a `"<provider>:<model>"` string into the right chat model class, so `config.yaml` ids need no translation table
- **LangGraph orchestration** _(planned)_ — provider fan-out and synthesis will be modelled as graph nodes; today `fanout.py` is a plain `asyncio.gather` and no graph exists yet
- **No persistence** — a council round returns its result and keeps nothing; history and the session endpoints the original spec described are dropped until something actually needs to re-read a past round
- **Single-user, local-only** for MVP
- **Question-only input** for MVP (no file attachments or system prompts)

## Planned Features

- [ ] **Second debate round** — each model sees the other models' first-round answers and revises its own before synthesis runs
- [ ] **Web search tool** — council models can call a search tool to ground their answers in current data instead of training-cutoff knowledge

## Getting Started

All commands are run from the repository root. `uv` manages a single `.venv` at
the root shared by every workspace member (`core`, `backends/langgraph`).

### 1. Provide the API keys

```bash
cp .env.example .env      # then paste your real keys into .env
```

`.env` is gitignored. `config.yaml` refers to keys only by env var *name*
(`api_key_env`); the backend reads `os.environ[<that name>]` at call time.

### 2. Install dependencies

```bash
uv sync --all-packages --frozen
```

Resolves and installs every workspace member's dependencies into the root
`.venv`. `--all-packages` is required: a plain `uv sync` at the root treats only
the root package as installable and *uninstalls* `core` and `council-langgraph`,
after which every import of them fails. `--frozen` installs exactly what `uv.lock` pins and fails instead of
re-resolving, so the environment matches the committed lockfile — use it
whenever you have not intentionally changed a dependency. Run this after
pulling, or whenever imports fail for a package that is already listed in a
`pyproject.toml`.

### 3. Verify the environment (optional)

```bash
uv run --directory backends/langgraph \
  python -c "import langchain_anthropic, langchain_openai, council_langgraph.fanout"
```

Cheap import-only smoke test that spends no API credits. `--directory` runs the
command with that member as the active project, which is what makes the
`council_langgraph` package importable. Silence means everything resolved.

### 4. Ask the council a question

```bash
uv run --package council-langgraph uvicorn council_langgraph.api:app --reload
```

Serves `POST /api/council/ask` and `POST /api/council/ask/stream` on
http://127.0.0.1:8000, with the interactive schema at `/docs`.

`--package` selects the workspace member to run *without* changing the working
directory — unlike `--directory` above. That matters because `load_config()`
resolves `config.yaml` relative to the current directory, so the server must stay
at the repository root. Plain `uv run uvicorn ...` does not work, because the root
workspace package does not depend on `council-langgraph`.

The API is the only way to run a council round — there is no CLI harness. Ask it
for a whole result at once:

```bash
curl -X POST http://127.0.0.1:8000/api/council/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Spider-Man vs Punisher, who wins?"}'
```

Or watch the answers arrive token by token:

```bash
curl -N -X POST http://127.0.0.1:8000/api/council/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "Spider-Man vs Punisher, who wins?"}'
```

`-N` disables curl's output buffering, without which the whole stream appears at
once. Both calls make real, billable provider calls — one per council member plus
one for the synthesizer.

To debug a round, open the **api** run configuration in `.run/` and start it with
the debugger rather than this command: PyCharm launches `.venv/bin/python`
directly, so breakpoints bind, and `curl` then blocks on your breakpoints. Omit
`--reload` when debugging — reload runs the app in a forked child whose
breakpoints never bind.

### 5. Run the tests

```bash
uv run --group dev pytest
```

The `dev` group (pytest, pytest-asyncio) lives in the root `pyproject.toml` so a
single invocation covers all members. Every test runs in-process — there is no
database or other service to start first.

## License

_TODO_