from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import TypeAdapter

from core.config import CouncilConfig, load_config
from core.types import CouncilResult, CouncilStreamEvent

from council_langgraph.council import ask_council, ask_council_streaming
from council_langgraph.synthesis import SynthesisError


@lru_cache(maxsize=1)
def get_config() -> CouncilConfig:
    load_dotenv()
    return load_config()


app: FastAPI = FastAPI(title="AI Council")


@app.exception_handler(SynthesisError)
async def _synthesis_error_handler(request: Request, exc: SynthesisError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.post("/api/council/ask", response_model=CouncilResult)
async def ask(
    question: Annotated[str, Body(embed=True, min_length=1, pattern=r"\S")],
    config: CouncilConfig = Depends(get_config),
) -> CouncilResult:
    return await ask_council(config, question)


_STREAM_EVENT_ADAPTER: TypeAdapter[CouncilStreamEvent] = TypeAdapter(CouncilStreamEvent)


def _sse_message(event: CouncilStreamEvent) -> str:
    data: str = _STREAM_EVENT_ADAPTER.dump_json(event).decode()
    return f"event: {event.event}\ndata: {data}\n\n"


@app.post("/api/council/ask/stream")
async def ask_streaming(
    question: Annotated[str, Body(embed=True, min_length=1, pattern=r"\S")],
    config: CouncilConfig = Depends(get_config),
) -> StreamingResponse:
    async def messages() -> AsyncIterator[str]:
        async for event in ask_council_streaming(config, question):
            yield _sse_message(event)

    return StreamingResponse(messages(), media_type="text/event-stream")
