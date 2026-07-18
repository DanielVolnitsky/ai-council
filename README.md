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

Monorepo with two backend implementations sharing a common core:

```
core/              # Shared: models, DB, config, synthesis prompt
backends/
  litellm/         # Implementation A: LiteLLM + custom async logic
  langgraph/       # Implementation B: LangGraph orchestration
frontend/          # React (Vite) SPA
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Vite |
| Backend | FastAPI (Python) |
| Provider abstraction | LiteLLM / LangGraph |
| Streaming | SSE (Server-Sent Events) |
| Persistence | SQLite |
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
- **Two implementations** — LiteLLM (minimal, DIY) vs LangGraph (framework-based) for comparison and learning goals
- **Single-user, local-only** for MVP
- **Question-only input** for MVP (no file attachments or system prompts)

## Getting Started

_TODO: setup instructions after implementation_

## License

_TODO_