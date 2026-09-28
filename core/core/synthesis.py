from __future__ import annotations

from core.config import CouncilConfig, ModelConfig
from core.domain import ModelResponse


class SynthesisError(Exception):
    pass


class UnknownSynthesizerError(SynthesisError):
    pass


class NoResponsesToSynthesizeError(SynthesisError):
    pass


class SynthesizerCallError(SynthesisError):
    pass


def synthesizer_effective_config(
    config: CouncilConfig,
    synthesizer_override: str | None,
) -> ModelConfig:
    synthesizer_id: str = synthesizer_override or config.default_synthesizer

    for model_config in config.enabled_models:
        if model_config.id == synthesizer_id:
            return model_config

    enabled_ids: list[str] = sorted(m.id for m in config.enabled_models)
    raise UnknownSynthesizerError(
        f"'{synthesizer_id}' is not an enabled model. Enabled model IDs: {enabled_ids}"
    )


def responses_to_synthesize(responses: list[ModelResponse]) -> list[ModelResponse]:
    successful_responses: list[ModelResponse] = [
        response for response in responses if response.error is None
    ]
    if not successful_responses:
        raise NoResponsesToSynthesizeError(
            f"All {len(responses)} council members failed; nothing to synthesize."
        )

    return successful_responses
