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
core/              # Shared: models, DB, config, synthesis prompt
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
| Persistence | PostgreSQL (asyncpg, Docker) |
| Config | YAML |
| Tracing | Langfuse (self-hosted, Docker) |
| Package manager | uv |

### API

Client-agnostic REST API designed for reuse by future clients (Telegram bot, mobile app, etc.):

- `POST /api/council/ask` — synchronous, returns full JSON result
- `POST /api/council/ask/stream` — SSE, streams token events per model + synthesis

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
- **Single-user, local-only** for MVP
- **Question-only input** for MVP (no file attachments or system prompts)

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
uv sync --frozen
```

Resolves and installs every workspace member's dependencies into the root
`.venv`. `--frozen` installs exactly what `uv.lock` pins and fails instead of
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

### 4. Ask the council a real question

```bash
uv run --package council-langgraph \
  python -m council_langgraph.demo "is it worth it?"
```

Runs the demo harness in `backends/langgraph/council_langgraph/demo.py`:
it fans the question out to every enabled model in `config.yaml` concurrently and
prints each raw answer. This makes real, billable provider calls.

`--package` selects the workspace member to run *without* changing the working
directory — unlike `--directory` above. That matters here because
`load_config()` resolves `config.yaml` relative to the current directory, so the
command must stay at the repository root. Plain `uv run python -m ...` does not
work, because the root workspace package does not depend on `council-langgraph`.

### 5. Run the tests

```bash
docker compose up -d db     # core/tests/test_db.py talks to a real PostgreSQL
uv run --group dev pytest
```

The `dev` group (pytest, pytest-asyncio) lives in the root `pyproject.toml` so a
single invocation covers all members. Without the database running, the config
and fan-out tests still pass but every `test_db.py` case errors out with
`Connect call failed ('127.0.0.1', 5432)`.

## License

_TODO_