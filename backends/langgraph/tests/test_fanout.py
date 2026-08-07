# backends/langgraph/tests/test_fanout.py
#
# Unit tests for the model fan-out.  No network calls: build_chat_model is
# monkeypatched with a stub factory, so every test runs offline and instantly.
#
# monkeypatch is pytest's built-in fixture for temporarily replacing an
# attribute; the original is restored when the test ends.  It is the closest
# equivalent to Mockito's static mocking, but scoped by the test runner rather
# than a try/finally block.

from typing import Callable, TypedDict

import pytest
from langchain_core.messages import AIMessage

from council_langgraph import fanout
from core.config import CouncilConfig, ModelConfig
from core.types import ModelResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG: CouncilConfig = CouncilConfig(
    default_synthesizer="openai:gpt-4o",
    models=[
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"),
        ModelConfig(id="anthropic:claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY"),
        ModelConfig(id="ollama:llama3", base_url="http://localhost:11434", enabled=False),
    ],
)


class StubChatModel:
    """
    Stands in for a LangChain chat model.

    Either replies with `reply` or raises `error` from ainvoke(), which is the
    only method run_fanout() calls on a chat model.
    """

    def __init__(self, reply: AIMessage | None = None, error: Exception | None = None):
        self._reply: AIMessage | None = reply
        self._error: Exception | None = error

    async def ainvoke(self, question: str) -> AIMessage:
        if self._error is not None:
            raise self._error
        return self._reply


def stub_builder(
    replies: dict[str, StubChatModel],
) -> Callable[[ModelConfig], StubChatModel]:
    """Build a build_chat_model replacement that dispatches on model id."""
    return lambda model_config: replies[model_config.id]


class CapturedInitArgs(TypedDict, total=False):
    """
    What the fake init_chat_model recorded about the call it received.

    `total=False` because api_key and base_url are each present only when the
    ModelConfig under test declares them.
    """
    model_id: str
    api_key: str
    base_url: str


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_fanout_returns_one_response_per_enabled_model(monkeypatch):
    """
    Golden path: every enabled model answers, and the disabled Ollama model is
    absent from the result.  Responses come back in config order.
    """
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
    """
    A dead provider must degrade to an error marker, not kill the round —
    the surviving model's answer is still returned.
    """
    monkeypatch.setattr(fanout, "build_chat_model", stub_builder({
        "openai:gpt-4o": StubChatModel(error=TimeoutError("request timed out")),
        "anthropic:claude-sonnet-5": StubChatModel(AIMessage(content="claude says no")),
    }))

    responses: list[ModelResponse] = await fanout.fanout_question(CONFIG, "is it worth it?")

    assert responses == [
        ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: request timed out"),
        ModelResponse(model_id="anthropic:claude-sonnet-5", response="claude says no", error=None),
    ]


async def test_block_style_content_is_flattened_to_text(monkeypatch):
    """
    Providers may reply with typed content blocks instead of a plain string.
    Only the text blocks survive; non-text blocks are dropped.
    """
    monkeypatch.setattr(fanout, "build_chat_model", stub_builder({
        "openai:gpt-4o": StubChatModel(AIMessage(content=[
            {"type": "text", "text": "first part "},
            {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
            {"type": "text", "text": "second part"},
        ])),
        "anthropic:claude-sonnet-5": StubChatModel(AIMessage(content="")),
    }))

    responses: list[ModelResponse] = await fanout.fanout_question(CONFIG, "is it worth it?")

    assert responses == [
        ModelResponse(model_id="openai:gpt-4o", response="first part second part", error=None),
        ModelResponse(model_id="anthropic:claude-sonnet-5", response="", error=None),
    ]


def test_build_chat_model_reads_api_key_from_env(monkeypatch):
    """
    The key is looked up from the env var *named* in config.yaml and handed to
    the provider class — config never holds the secret itself.
    """
    captured: CapturedInitArgs = {}

    def fake_init_chat_model(model_id: str, **kwargs: str) -> StubChatModel:
        captured.update({"model_id": model_id, **kwargs})
        return StubChatModel(AIMessage(content=""))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setattr(fanout, "init_chat_model", fake_init_chat_model)

    fanout.build_chat_model(ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"))

    assert captured == {"model_id": "openai:gpt-4o", "api_key": "sk-test-123"}


def test_build_chat_model_passes_base_url_without_api_key(monkeypatch):
    """A local provider (Ollama) has no key — only base_url is forwarded."""
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
    """
    An unset API key env var is a deployment error.  build_chat_model raises
    KeyError; run_fanout is what converts it into a per-model error marker.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        fanout.build_chat_model(
            ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY")
        )
