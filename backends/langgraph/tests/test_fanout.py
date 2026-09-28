from typing import Callable, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from council_langgraph import fanout
from core.config import CouncilConfig, ModelConfig
from core.domain import ModelResponse

CONFIG: CouncilConfig = CouncilConfig(
    default_synthesizer="openai:gpt-4o",
    models=[
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"),
        ModelConfig(id="anthropic:claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY"),
        ModelConfig(id="ollama:llama3", base_url="http://localhost:11434", enabled=False),
    ],
)


class StubChatModel:
    def __init__(self, reply: AIMessage | None = None, error: Exception | None = None):
        self._reply: AIMessage | None = reply
        self._error: Exception | None = error

    async def ainvoke(self, question: str, config: RunnableConfig) -> AIMessage:
        if self._error is not None:
            raise self._error
        return self._reply


def stub_builder(
    replies: dict[str, StubChatModel],
) -> Callable[[ModelConfig], StubChatModel]:
    return lambda model_config: replies[model_config.id]


class CapturedInitArgs(TypedDict, total=False):
    model_id: str
    api_key: str
    base_url: str


async def test_fanout_returns_one_response_per_enabled_model(monkeypatch):
    monkeypatch.setattr(fanout, "build_chat_model", stub_builder({
        "openai:gpt-4o": StubChatModel(AIMessage(content="gpt says yes")),
        "anthropic:claude-sonnet-5": StubChatModel(AIMessage(content="claude says no")),
    }))

    responses: list[ModelResponse] = await fanout.fanout_question(CONFIG, "is it worth it?")

    assert responses == [
        ModelResponse(model_id="openai:gpt-4o", response="gpt says yes", error=None),
        ModelResponse(model_id="anthropic:claude-sonnet-5", response="claude says no", error=None),
    ]


async def test_failing_model_does_not_abort_the_others(monkeypatch):
    monkeypatch.setattr(fanout, "build_chat_model", stub_builder({
        "openai:gpt-4o": StubChatModel(error=TimeoutError("request timed out")),
        "anthropic:claude-sonnet-5": StubChatModel(AIMessage(content="claude says no")),
    }))

    responses: list[ModelResponse] = await fanout.fanout_question(CONFIG, "is it worth it?")

    assert responses == [
        ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: request timed out"),
        ModelResponse(model_id="anthropic:claude-sonnet-5", response="claude says no", error=None),
    ]


def test_build_chat_model_reads_api_key_from_env(monkeypatch):
    captured: CapturedInitArgs = {}

    def fake_init_chat_model(model_id: str, **kwargs: str) -> StubChatModel:
        captured.update({"model_id": model_id, **kwargs})
        return StubChatModel(AIMessage(content=""))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setattr(fanout, "init_chat_model", fake_init_chat_model)

    fanout.build_chat_model(ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"))

    assert captured == {"model_id": "openai:gpt-4o", "api_key": "sk-test-123"}


def test_build_chat_model_passes_base_url_without_api_key(monkeypatch):
    captured: CapturedInitArgs = {}

    def fake_init_chat_model(model_id: str, **kwargs: str) -> StubChatModel:
        captured.update({"model_id": model_id, **kwargs})
        return StubChatModel(AIMessage(content=""))

    monkeypatch.setattr(fanout, "init_chat_model", fake_init_chat_model)

    fanout.build_chat_model(
        ModelConfig(id="ollama:llama3", base_url="http://localhost:11434")
    )

    assert captured == {"model_id": "ollama:llama3", "base_url": "http://localhost:11434"}


def test_missing_api_key_env_var_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        fanout.build_chat_model(
            ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY")
        )
