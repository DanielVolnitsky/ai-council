import pytest

from core.config import CouncilConfig, ModelConfig
from core.domain import ModelResponse
from core.synthesis import (
    NoResponsesToSynthesizeError,
    SynthesisError,
    SynthesizerCallError,
    UnknownSynthesizerError,
    responses_to_synthesize,
    synthesizer_effective_config,
)

CONFIG: CouncilConfig = CouncilConfig(
    default_synthesizer="openai:gpt-4o",
    models=[
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"),
        ModelConfig(id="anthropic:claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY"),
        ModelConfig(id="ollama:llama3", base_url="http://localhost:11434", enabled=False),
    ],
)


def test_default_synthesizer_is_used_without_an_override():
    assert synthesizer_effective_config(CONFIG, None) == ModelConfig(
        id="openai:gpt-4o", api_key_env="OPENAI_API_KEY"
    )


def test_override_replaces_the_default_synthesizer():
    assert synthesizer_effective_config(CONFIG, "anthropic:claude-haiku-4-5") == ModelConfig(
        id="anthropic:claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY"
    )


def test_unknown_synthesizer_id_raises():
    with pytest.raises(UnknownSynthesizerError, match="google:gemini-2.5-pro"):
        synthesizer_effective_config(CONFIG, "google:gemini-2.5-pro")


def test_disabled_model_cannot_be_the_synthesizer():
    with pytest.raises(UnknownSynthesizerError, match="ollama:llama3"):
        synthesizer_effective_config(CONFIG, "ollama:llama3")


def test_failed_members_are_not_synthesized():
    responses: list[ModelResponse] = [
        ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: timed out"),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y."),
    ]

    assert responses_to_synthesize(responses) == [
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y."),
    ]


def test_all_members_failed_raises():
    responses: list[ModelResponse] = [
        ModelResponse(model_id="openai:gpt-4o", response="", error="TimeoutError: timed out"),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="", error="AuthError: bad key"),
    ]

    with pytest.raises(NoResponsesToSynthesizeError, match="All 2 council members failed"):
        responses_to_synthesize(responses)


@pytest.mark.parametrize("error_type", [
    UnknownSynthesizerError,
    NoResponsesToSynthesizeError,
    SynthesizerCallError,
])
def test_every_failure_mode_is_a_synthesis_error(error_type: type):
    assert issubclass(error_type, SynthesisError)
