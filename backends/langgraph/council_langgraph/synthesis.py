from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import CouncilConfig, ModelConfig
from core.prompts import SYNTHESIS_SYSTEM_PROMPT, synthesis_input
from core.domain import CouncilSynthesis, ModelResponse
from core.synthesis import (
    SynthesizerCallError,
    responses_to_synthesize,
    synthesizer_effective_config,
)

from council_langgraph.fanout import build_chat_model
from council_langgraph.tracing import traced_run_config


async def council_synthesised_answer(
    config: CouncilConfig,
    question: str,
    responses: list[ModelResponse],
    synthesizer_model_override: str | None = None,
) -> CouncilSynthesis:
    synthesizer_config: ModelConfig = synthesizer_effective_config(config, synthesizer_model_override)
    council_responses: list[ModelResponse] = responses_to_synthesize(responses)

    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_input(question, council_responses)),
    ]

    try:
        chat_model: BaseChatModel = build_chat_model(synthesizer_config)
        structured = chat_model.with_structured_output(
            CouncilSynthesis, method="function_calling"
        )
        synthesis = await structured.ainvoke(
            messages, config=traced_run_config(synthesizer_config.id)
        )
    except Exception as exc:
        raise SynthesizerCallError(
            f"synthesizer '{synthesizer_config.id}' failed: {type(exc).__name__}: {exc}"
        ) from exc

    return cast(CouncilSynthesis, synthesis)
