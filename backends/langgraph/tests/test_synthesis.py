import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from council_langgraph import synthesis
from core.config import CouncilConfig, ModelConfig
from core.domain import CouncilSynthesis, Disagreement, ModelInsights, ModelResponse, Verdict
from core.synthesis import (
    NoResponsesToSynthesizeError,
    SynthesizerCallError,
    UnknownSynthesizerError,
)

CONFIG: CouncilConfig = CouncilConfig(
    default_synthesizer="openai:gpt-4o",
    models=[
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"),
        ModelConfig(id="anthropic:claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY"),
        ModelConfig(id="ollama:llama3", base_url="http://localhost:11434", enabled=False),
    ],
)

RESPONSES: list[ModelResponse] = [
    ModelResponse(model_id="openai:gpt-4o", response="Yes, because of X."),
    ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y."),
]

SYNTHESIS: CouncilSynthesis = CouncilSynthesis(
    summary="The council split on the question.",
    consensus=["The question is worth asking."],
    disagreements=[
        Disagreement(
            point="Whether it is worth it",
            models_for=["openai:gpt-4o"],
            models_against=["anthropic:claude-haiku-4-5"],
        )
    ],
    verdict=Verdict(
        strongest="openai:gpt-4o",
        weakest="anthropic:claude-haiku-4-5",
        justification="gpt-4o gave reasons; claude asserted.",
    ),
    unique_insights=[
        ModelInsights(model_id="openai:gpt-4o", insights=["X matters more than Y."]),
    ],
    blind_spots=["Neither model considered cost."],
    takeaways=["Estimate the cost before deciding."],
)


class StubStructuredModel:
    def __init__(self, result: CouncilSynthesis | None = None, error: Exception | None = None):
        self._result: CouncilSynthesis | None = result
        self._error: Exception | None = error
        self.received_messages: list = []

    async def ainvoke(self, messages: list, config: RunnableConfig) -> CouncilSynthesis:
        self.received_messages = messages
        if self._error is not None:
            raise self._error
        return self._result


class StubChatModel:
    def __init__(self, structured: StubStructuredModel):
        self._structured: StubStructuredModel = structured
        self.structured_output_schema: type | None = None
        self.structured_output_method: str | None = None

    def with_structured_output(self, schema: type, *, method: str) -> StubStructuredModel:
        self.structured_output_schema = schema
        self.structured_output_method = method
        return self._structured


def install_stub(
    monkeypatch,
    structured: StubStructuredModel,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    chat_model: StubChatModel = StubChatModel(structured)

    def fake_build_chat_model(model_config: ModelConfig) -> StubChatModel:
        captured["model_config"] = model_config
        return chat_model

    def fake_synthesis_input(question: str, responses: list[ModelResponse]) -> str:
        captured["question"] = question
        captured["responses"] = responses
        return "<rendered prompt>"

    monkeypatch.setattr(synthesis, "build_chat_model", fake_build_chat_model)
    monkeypatch.setattr(synthesis, "synthesis_input", fake_synthesis_input)

    captured["chat_model"] = chat_model
    return captured


async def test_synthesize_returns_the_structured_document(monkeypatch):
    install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    result: CouncilSynthesis = await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert result == SYNTHESIS


async def test_default_synthesizer_is_used_when_none_is_requested(monkeypatch):
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["model_config"] == ModelConfig(
        id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"
    )


async def test_explicit_synthesizer_overrides_the_default(monkeypatch):
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(
        CONFIG, "is it worth it?", RESPONSES,
        synthesizer_model_override="anthropic:claude-haiku-4-5",
    )

    assert captured["model_config"] == ModelConfig(
        id="anthropic:claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY"
    )


async def test_output_is_structured_against_the_synthesis_schema(monkeypatch):
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["chat_model"].structured_output_schema is CouncilSynthesis


async def test_output_is_structured_via_tool_calling(monkeypatch):
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["chat_model"].structured_output_method == "function_calling"


async def test_system_prompt_precedes_the_rendered_responses(monkeypatch):
    structured: StubStructuredModel = StubStructuredModel(result=SYNTHESIS)
    install_stub(monkeypatch, structured)

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert [(type(m).__name__, m.content) for m in structured.received_messages] == [
        ("SystemMessage", synthesis.SYNTHESIS_SYSTEM_PROMPT),
        ("HumanMessage", "<rendered prompt>"),
    ]


async def test_failed_members_are_excluded_from_the_prompt(monkeypatch):
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", [
        ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: timed out"),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y."),
    ])

    assert captured["responses"] == [
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y.")
    ]


async def test_question_is_passed_through_verbatim(monkeypatch):
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["question"] == "is it worth it?"


async def test_all_members_failed_raises_before_calling_the_synthesizer(monkeypatch):
    structured: StubStructuredModel = StubStructuredModel(result=SYNTHESIS)
    install_stub(monkeypatch, structured)

    with pytest.raises(NoResponsesToSynthesizeError):
        await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", [
            ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: timed out"),
            ModelResponse(model_id="anthropic:claude-haiku-4-5", response="", error="AuthError: bad key"),
        ])

    assert structured.received_messages == []


async def test_unknown_synthesizer_id_raises(monkeypatch):
    install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    with pytest.raises(UnknownSynthesizerError):
        await synthesis.council_synthesised_answer(
            CONFIG, "is it worth it?", RESPONSES,
            synthesizer_model_override="google:gemini-2.5-pro",
        )


async def test_provider_failure_is_wrapped_with_its_cause(monkeypatch):
    error: TimeoutError = TimeoutError("request timed out")
    install_stub(monkeypatch, StubStructuredModel(error=error))

    with pytest.raises(SynthesizerCallError, match="TimeoutError: request timed out") as exc_info:
        await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert exc_info.value.__cause__ is error


async def test_schema_violation_is_wrapped_too(monkeypatch):
    error: ValidationError = ValidationError.from_exception_data("CouncilSynthesis", [])
    install_stub(monkeypatch, StubStructuredModel(error=error))

    with pytest.raises(SynthesizerCallError):
        await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)
