from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from council_langgraph import council
from council_langgraph.api import app, get_config
from council_langgraph.synthesis import SynthesisError
from core.config import CouncilConfig, ModelConfig
from core.domain import (
    CouncilSynthesis,
    Disagreement,
    ModelInsights,
    ModelResponse,
    Verdict,
)

CONFIG: CouncilConfig = CouncilConfig(
    default_synthesizer="openai:gpt-4o",
    models=[
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"),
        ModelConfig(id="anthropic:claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY"),
    ],
)

RESPONSES: list[ModelResponse] = [
    ModelResponse(model_id="openai:gpt-4o", response="gpt says yes"),
    ModelResponse(model_id="anthropic:claude-sonnet-5", response="claude says no"),
]

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
    unique_insights=[
        ModelInsights(model_id="openai:gpt-4o", insights=["cost matters"]),
    ],
    blind_spots=["nobody mentioned time"],
    takeaways=["decide by friday"],
)


@pytest.fixture
def client():
    app.dependency_overrides[get_config] = lambda: CONFIG
    yield TestClient(app)
    app.dependency_overrides.clear()


def stub_fanout(responses: list[ModelResponse]):
    async def _fanout(config: CouncilConfig, question: str) -> list[ModelResponse]:
        return responses

    return _fanout


def stub_synthesis(synthesis: CouncilSynthesis | None = None, error: Exception | None = None):
    async def _synthesise(
        config: CouncilConfig,
        question: str,
        responses: list[ModelResponse],
    ) -> CouncilSynthesis:
        if error is not None:
            raise error
        return synthesis

    return _synthesise


def test_ask_returns_responses_and_synthesis(client, monkeypatch):
    monkeypatch.setattr(council, "fanout_question", stub_fanout(RESPONSES))
    monkeypatch.setattr(council, "council_synthesised_answer", stub_synthesis(SYNTHESIS))

    response = client.post("/api/council/ask", json={"question": "is it worth it?"})

    assert response.status_code == 200
    body: dict = response.json()
    assert UUID(body["session_id"])
    assert body["question"] == "is it worth it?"
    assert body["model_responses"] == [
        {"model_id": "openai:gpt-4o", "response": "gpt says yes", "error": None},
        {"model_id": "anthropic:claude-sonnet-5", "response": "claude says no", "error": None},
    ]
    assert body["synthesis"] == SYNTHESIS.model_dump()


def test_failed_synthesis_is_an_upstream_error(client, monkeypatch):
    monkeypatch.setattr(council, "fanout_question", stub_fanout(RESPONSES))
    monkeypatch.setattr(
        council,
        "council_synthesised_answer",
        stub_synthesis(error=SynthesisError("synthesizer 'openai:gpt-4o' failed")),
    )

    response = client.post("/api/council/ask", json={"question": "is it worth it?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "synthesizer 'openai:gpt-4o' failed"}


def test_empty_question_is_rejected(client):
    response = client.post("/api/council/ask", json={"question": "  "})

    assert response.status_code == 422
