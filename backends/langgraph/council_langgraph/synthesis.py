# backends/langgraph/council_langgraph/synthesis.py
#
# Synthesis: turn the council's raw answers into the structured seven-section
# document (core.types.CouncilSynthesis).
#
# This is the second half of one council round.  fanout.py produces N answers;
# this module reduces them to one analysis.
#
# Structured output:
#   The synthesizer must return JSON matching CouncilSynthesis exactly.  Rather
#   than asking for JSON in the prompt and parsing the reply by hand,
#   `with_structured_output(CouncilSynthesis)` derives a JSON schema from the
#   Pydantic model and wires it into whatever mechanism the provider offers —
#   OpenAI's response_format, Anthropic's tool use.  LangChain then validates
#   the reply against the model and hands back a CouncilSynthesis instance.
#
#   Java analogy: the difference between calling objectMapper.readValue() on a
#   String you hope is JSON, and having the transport layer deserialise into
#   the target type for you and throw if it does not fit.
#
# Failure policy — deliberately the opposite of fanout.py:
#   A failed council *member* degrades the round (N-1 answers still synthesize
#   fine), so fanout.py converts those into error markers.  A failed
#   *synthesizer* leaves nothing to return, so every failure here raises.  The
#   API layer maps SynthesisError to a 500 response (sync) or an `error` SSE
#   event (stream), per the design spec.

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import CouncilConfig, ModelConfig
from core.prompts import SYNTHESIS_SYSTEM_PROMPT, synthesis_input
from core.types import CouncilSynthesis, ModelResponse

from council_langgraph.fanout import build_chat_model


class SynthesisError(Exception):
    """
    Base class for every reason a synthesis could not be produced.

    Callers that only need "did it work?" catch this; the API layer catches the
    subclasses to pick a status code (400 for an unknown synthesizer, 500 for
    the rest).
    """


class UnknownSynthesizerError(SynthesisError):
    """
    The requested synthesizer model id is not an enabled model in config.yaml.

    Caused by client input (the `synthesizer_model` request field), so the API
    layer maps this one to 400 rather than 500.
    """


class NoResponsesToSynthesizeError(SynthesisError):
    """
    Every council member failed, so there is nothing to analyse.

    Distinct from a synthesizer failure: the synthesizer was never called.
    Usually means a local problem — no API keys, no network — rather than a
    simultaneous outage at every provider.
    """


class SynthesizerCallError(SynthesisError):
    """
    The synthesizer model was called and did not return a usable document.

    Covers both transport failures (auth, rate limit, timeout) and output that
    did not validate against the CouncilSynthesis schema.  The original
    exception is attached as __cause__.
    """


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


def _council_successful_responses(responses: list[ModelResponse]) -> list[ModelResponse]:
    return [response for response in responses if response.error is None]


async def council_synthesised_answer(
    config: CouncilConfig,
    question: str,
    responses: list[ModelResponse],
    synthesizer_model_override: str | None = None,
) -> CouncilSynthesis:
    synthesizer_config: ModelConfig = synthesizer_effective_config(config, synthesizer_model_override)

    council_responses: list[ModelResponse] = _council_successful_responses(responses)
    if not council_responses:
        raise NoResponsesToSynthesizeError(
            f"All {len(responses)} council members failed; nothing to synthesize."
        )

    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_input(question, council_responses)),
    ]

    try:
        chat_model: BaseChatModel = build_chat_model(synthesizer_config)
        # method="function_calling" rather than the provider default: OpenAI's
        # default (response_format / json_schema) runs the schema through its
        # *strict* validator, which rejects several shapes pydantic emits
        # routinely.  Tool calling carries the same schema without those
        # restrictions, and is what Anthropic uses by default anyway, so both
        # providers take one path.
        #
        # with_structured_output returns a Runnable whose output type the type
        # checker widens to dict | BaseModel, because the schema argument may
        # also be a plain dict.  Passing a Pydantic class pins it to that class
        # at runtime; the cast restates that guarantee.
        structured = chat_model.with_structured_output(
            CouncilSynthesis, method="function_calling"
        )
        synthesis = await structured.ainvoke(messages)
    except Exception as exc:
        # Bare `except Exception` for the same reason as in fanout.py: provider
        # SDKs raise a wide, undocumented range of errors, and pydantic adds
        # ValidationError on top when the reply does not fit the schema.  All of
        # them mean the same thing to the caller — no document — so they are
        # normalised into one type.  Nothing is swallowed: `from exc` keeps the
        # original traceback reachable via __cause__.
        raise SynthesizerCallError(
            f"synthesizer '{synthesizer_config.id}' failed: {type(exc).__name__}: {exc}"
        ) from exc

    return cast(CouncilSynthesis, synthesis)
