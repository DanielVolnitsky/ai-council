import asyncio
import json
from collections.abc import AsyncIterator
from typing import Callable, TypedDict
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig

from council_langgraph import council, fanout
from council_langgraph.api import app, get_config
from core.config import CouncilConfig, ModelConfig
from core.events import (
    CouncilStreamEvent,
    ErrorEvent,
    ModelDoneEvent,
    ModelTokenEvent,
    SessionStartEvent,
    SynthDoneEvent,
)
from core.domain import CouncilSynthesis, Disagreement, ModelInsights, Verdict
from core.synthesis import SynthesisError

CONFIG: CouncilConfig = CouncilConfig(
    default_synthesizer="openai:gpt-4o",
    models=[
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"),
        ModelConfig(id="anthropic:claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY"),
        ModelConfig(id="ollama:llama3", base_url="http://localhost:11434", enabled=False),
    ],
)

SYNTHESIS: CouncilSynthesis = CouncilSynthesis(
    summary="the council is split",
    consensus=["the question is worth asking"],
    disagreements=[
        Disagreement(
            point="whether it is worth it",
            models_for=["openai:gpt-4o"],
            models_against=["anthropic:claude-sonnet-5"],
        )
    ],
    verdict=Verdict(
        strongest="openai:gpt-4o",
        weakest="anthropic:claude-sonnet-5",
        justification="gpt gave reasons",
    ),
    unique_insights=[ModelInsights(model_id="openai:gpt-4o", insights=["cost matters"])],
    blind_spots=["nobody mentioned time"],
    takeaways=["decide by friday"],
)


class StubStreamingChatModel:
    def __init__(self, tokens: list[str] | None = None, error: Exception | None = None):
        self._tokens: list[str] = tokens or []
        self._error: Exception | None = error

    async def astream(
        self, question: str, config: RunnableConfig
    ) -> AsyncIterator[AIMessageChunk]:
        if self._error is not None:
            raise self._error
        for token in self._tokens:
            await asyncio.sleep(0)
            yield AIMessageChunk(content=token)


def stub_builder(
    models: dict[str, StubStreamingChatModel],
) -> Callable[[ModelConfig], StubStreamingChatModel]:
    return lambda model_config: models[model_config.id]


async def collect(events: AsyncIterator[CouncilStreamEvent]) -> list[CouncilStreamEvent]:
    return [event async for event in events]


async def test_each_model_streams_tokens_then_one_done_event(monkeypatch):
    monkeypatch.setattr(fanout, "build_chat_model", stub_builder({
        "openai:gpt-4o": StubStreamingChatModel(["gpt ", "says ", "yes"]),
        "anthropic:claude-sonnet-5": StubStreamingChatModel(["claude ", "says no"]),
    }))

    events: list[CouncilStreamEvent] = await collect(
        fanout.stream_question(CONFIG, "is it worth it?")
    )

    assert events == [
        ModelTokenEvent(model_id="openai:gpt-4o", token="gpt "),
        ModelTokenEvent(model_id="anthropic:claude-sonnet-5", token="claude "),
        ModelTokenEvent(model_id="openai:gpt-4o", token="says "),
        ModelTokenEvent(model_id="anthropic:claude-sonnet-5", token="says no"),
        ModelDoneEvent(
            model_id="anthropic:claude-sonnet-5", response="claude says no", error=None
        ),
        ModelTokenEvent(model_id="openai:gpt-4o", token="yes"),
        ModelDoneEvent(model_id="openai:gpt-4o", response="gpt says yes", error=None),
    ]


async def test_failing_model_reports_its_error_without_stopping_the_stream(monkeypatch):
    monkeypatch.setattr(fanout, "build_chat_model", stub_builder({
        "openai:gpt-4o": StubStreamingChatModel(error=TimeoutError("request timed out")),
        "anthropic:claude-sonnet-5": StubStreamingChatModel(["claude says no"]),
    }))

    events: list[CouncilStreamEvent] = await collect(
        fanout.stream_question(CONFIG, "is it worth it?")
    )

    assert events == [
        ModelDoneEvent(
            model_id="openai:gpt-4o", response="", error="TimeoutError: request timed out"
        ),
        ModelTokenEvent(model_id="anthropic:claude-sonnet-5", token="claude says no"),
        ModelDoneEvent(
            model_id="anthropic:claude-sonnet-5", response="claude says no", error=None
        ),
    ]


def stub_stream_question(events: list[CouncilStreamEvent]):
    async def _stream(config: CouncilConfig, question: str) -> AsyncIterator[CouncilStreamEvent]:
        for event in events:
            yield event

    return _stream


def stub_synthesis(synthesis: CouncilSynthesis | None = None, error: Exception | None = None):
    async def _synthesise(config, question, responses):
        if error is not None:
            raise error
        return synthesis

    return _synthesise


FANOUT_EVENTS: list[CouncilStreamEvent] = [
    ModelTokenEvent(model_id="openai:gpt-4o", token="gpt says yes"),
    ModelDoneEvent(model_id="openai:gpt-4o", response="gpt says yes", error=None),
    ModelDoneEvent(model_id="anthropic:claude-sonnet-5", response="", error="TimeoutError: nope"),
]


async def test_stream_council_ends_with_the_synthesis(monkeypatch):
    monkeypatch.setattr(council, "stream_question", stub_stream_question(FANOUT_EVENTS))
    monkeypatch.setattr(council, "council_synthesised_answer", stub_synthesis(SYNTHESIS))

    events: list[CouncilStreamEvent] = await collect(
        council.ask_council_streaming(CONFIG, "is it worth it?")
    )

    assert isinstance(events[0], SessionStartEvent)
    assert UUID(events[0].session_id)
    assert events[1:] == FANOUT_EVENTS + [SynthDoneEvent(synthesis=SYNTHESIS)]


async def test_stream_council_ends_with_an_error_event_when_synthesis_fails(monkeypatch):
    monkeypatch.setattr(council, "stream_question", stub_stream_question(FANOUT_EVENTS))
    monkeypatch.setattr(
        council,
        "council_synthesised_answer",
        stub_synthesis(error=SynthesisError("synthesizer 'openai:gpt-4o' failed")),
    )

    events: list[CouncilStreamEvent] = await collect(
        council.ask_council_streaming(CONFIG, "is it worth it?")
    )

    assert isinstance(events[0], SessionStartEvent)
    assert events[1:] == FANOUT_EVENTS + [
        ErrorEvent(message="synthesizer 'openai:gpt-4o' failed")
    ]


class SseMessage(TypedDict):
    event: str
    data: dict


def parse_sse(body: str) -> list[SseMessage]:
    messages: list[SseMessage] = []
    for block in body.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        messages.append(
            {
                "event": event_line.removeprefix("event: "),
                "data": json.loads(data_line.removeprefix("data: ")),
            }
        )
    return messages


@pytest.fixture
def client():
    app.dependency_overrides[get_config] = lambda: CONFIG
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_stream_endpoint_emits_one_sse_message_per_event(client, monkeypatch):
    monkeypatch.setattr(council, "stream_question", stub_stream_question(FANOUT_EVENTS))
    monkeypatch.setattr(council, "council_synthesised_answer", stub_synthesis(SYNTHESIS))

    response = client.post("/api/council/ask/stream", json={"question": "is it worth it?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    messages: list[SseMessage] = parse_sse(response.text)
    assert messages[0]["event"] == "session_start"
    assert UUID(messages[0]["data"]["session_id"])
    assert messages[1:] == [
        {"event": "model_token", "data": {"model_id": "openai:gpt-4o", "token": "gpt says yes"}},
        {
            "event": "model_done",
            "data": {"model_id": "openai:gpt-4o", "response": "gpt says yes", "error": None},
        },
        {
            "event": "model_done",
            "data": {
                "model_id": "anthropic:claude-sonnet-5",
                "response": "",
                "error": "TimeoutError: nope",
            },
        },
        {"event": "synth_done", "data": {"synthesis": SYNTHESIS.model_dump()}},
    ]


def test_stream_endpoint_rejects_an_empty_question(client):
    response = client.post("/api/council/ask/stream", json={"question": "  "})

    assert response.status_code == 422
