# backends/langgraph/tests/test_synthesis.py
#
# Unit tests for the synthesis step.  No network calls and no real prompt: both
# build_chat_model and format_council_responses are monkeypatched, so these
# tests stay green while format_council_responses is still unimplemented and do
# not re-test what core/tests/test_prompts.py already covers.
#
# What is actually under test here is the wiring and the failure policy:
# which model gets picked, which responses reach the synthesizer, and which
# exception type each failure mode produces.

import pytest
from pydantic import ValidationError

from council_langgraph import synthesis
from core.config import CouncilConfig, ModelConfig
from core.types import CouncilSynthesis, Disagreement, ModelInsights, ModelResponse, Verdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    """
    Stands in for the Runnable returned by with_structured_output().

    Either returns `result` from ainvoke() or raises `error` — the only two
    outcomes synthesize() distinguishes.
    """

    def __init__(self, result: CouncilSynthesis | None = None, error: Exception | None = None):
        self._result: CouncilSynthesis | None = result
        self._error: Exception | None = error
        self.received_messages: list = []

    async def ainvoke(self, messages: list) -> CouncilSynthesis:
        self.received_messages = messages
        if self._error is not None:
            raise self._error
        return self._result


class StubChatModel:
    """Records which schema and method it was asked to structure output against."""

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
    """
    Patch out both the provider factory and the prompt renderer.

    Returns a dict the test can inspect afterwards: which ModelConfig
    build_chat_model was handed, and which responses reached the prompt.
    """
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


# ---------------------------------------------------------------------------
# Golden path
# ---------------------------------------------------------------------------

async def test_synthesize_returns_the_structured_document(monkeypatch):
    """The document the provider produced is returned unchanged."""
    install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    result: CouncilSynthesis = await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert result == SYNTHESIS


async def test_default_synthesizer_is_used_when_none_is_requested(monkeypatch):
    """Omitting the id falls back to config.default_synthesizer."""
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["model_config"] == ModelConfig(
        id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"
    )


async def test_explicit_synthesizer_overrides_the_default(monkeypatch):
    """A caller may synthesize with any enabled model."""
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(
        CONFIG, "is it worth it?", RESPONSES,
        synthesizer_model_override="anthropic:claude-haiku-4-5",
    )

    assert captured["model_config"] == ModelConfig(
        id="anthropic:claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY"
    )


async def test_output_is_structured_against_the_synthesis_schema(monkeypatch):
    """
    The schema handed to the provider is CouncilSynthesis itself — that is what
    makes the reply validated rather than hand-parsed.
    """
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["chat_model"].structured_output_schema is CouncilSynthesis


async def test_output_is_structured_via_tool_calling(monkeypatch):
    """
    Tool calling, not the provider default: OpenAI's json_schema mode is strict
    and rejects the open-ended `unique_insights` object.
    """
    captured: dict[str, object] = install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert captured["chat_model"].structured_output_method == "function_calling"


async def test_system_prompt_precedes_the_rendered_responses(monkeypatch):
    """Two messages reach the model: standing instructions, then this round's payload."""
    structured: StubStructuredModel = StubStructuredModel(result=SYNTHESIS)
    install_stub(monkeypatch, structured)

    await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert [(type(m).__name__, m.content) for m in structured.received_messages] == [
        ("SystemMessage", synthesis.SYNTHESIS_SYSTEM_PROMPT),
        ("HumanMessage", "<rendered prompt>"),
    ]


# ---------------------------------------------------------------------------
# Which responses reach the synthesizer
# ---------------------------------------------------------------------------

async def test_failed_members_are_excluded_from_the_prompt(monkeypatch):
    """
    An error marker is a fact about the infrastructure, not a position in the
    debate — the synthesizer must not weigh "TimeoutError" as an answer.
    """
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


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------

async def test_all_members_failed_raises_before_calling_the_synthesizer(monkeypatch):
    """
    Nothing to analyse means the synthesizer is never called — no credits spent
    asking a model to summarise an empty council.
    """
    structured: StubStructuredModel = StubStructuredModel(result=SYNTHESIS)
    install_stub(monkeypatch, structured)

    with pytest.raises(synthesis.NoResponsesToSynthesizeError, match="All 2 council members failed"):
        await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", [
            ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: timed out"),
            ModelResponse(model_id="anthropic:claude-haiku-4-5", response="", error="AuthError: bad key"),
        ])

    assert structured.received_messages == []


async def test_unknown_synthesizer_id_raises(monkeypatch):
    """Client-supplied id that is not an enabled model — a 400, not a 500."""
    install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    with pytest.raises(synthesis.UnknownSynthesizerError, match="google:gemini-2.5-pro"):
        await synthesis.council_synthesised_answer(
            CONFIG, "is it worth it?", RESPONSES,
            synthesizer_model_override="google:gemini-2.5-pro",
        )


async def test_disabled_model_cannot_be_the_synthesizer(monkeypatch):
    """Present in config.yaml but enabled=false is still not selectable."""
    install_stub(monkeypatch, StubStructuredModel(result=SYNTHESIS))

    with pytest.raises(synthesis.UnknownSynthesizerError, match="ollama:llama3"):
        await synthesis.council_synthesised_answer(
            CONFIG, "is it worth it?", RESPONSES,
            synthesizer_model_override="ollama:llama3",
        )


async def test_provider_failure_is_wrapped_with_its_cause(monkeypatch):
    """
    Unlike a council member's failure, a synthesizer failure aborts the round —
    but the original exception stays reachable through __cause__.
    """
    error: TimeoutError = TimeoutError("request timed out")
    install_stub(monkeypatch, StubStructuredModel(error=error))

    with pytest.raises(synthesis.SynthesizerCallError, match="TimeoutError: request timed out") as exc_info:
        await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)

    assert exc_info.value.__cause__ is error


async def test_schema_violation_is_wrapped_too(monkeypatch):
    """
    Output that does not fit CouncilSynthesis surfaces as a ValidationError from
    LangChain's own parsing; it must reach the caller as a SynthesisError like
    any other synthesizer failure, not as a bare pydantic error.
    """
    error: ValidationError = ValidationError.from_exception_data("CouncilSynthesis", [])
    install_stub(monkeypatch, StubStructuredModel(error=error))

    with pytest.raises(synthesis.SynthesizerCallError):
        await synthesis.council_synthesised_answer(CONFIG, "is it worth it?", RESPONSES)


@pytest.mark.parametrize("error_type", [
    synthesis.UnknownSynthesizerError,
    synthesis.NoResponsesToSynthesizeError,
    synthesis.SynthesizerCallError,
])
def test_every_failure_mode_is_a_synthesis_error(error_type: type):
    """
    A caller that only needs "did it work?" catches one type; the API layer
    catches the subclasses to choose a status code.
    """
    assert issubclass(error_type, synthesis.SynthesisError)
