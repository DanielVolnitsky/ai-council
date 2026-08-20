from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

from core.config import CouncilConfig
from core.types import (
    CouncilResult,
    CouncilStreamEvent,
    CouncilSynthesis,
    ErrorEvent,
    ModelDoneEvent,
    ModelResponse,
    SynthDoneEvent,
)

from council_langgraph.fanout import fanout_question, stream_question
from council_langgraph.synthesis import SynthesisError, council_synthesised_answer


async def ask_council(config: CouncilConfig, question: str) -> CouncilResult:
    responses: list[ModelResponse] = await fanout_question(config, question)
    synthesis: CouncilSynthesis = await council_synthesised_answer(config, question, responses)

    return CouncilResult(
        question=question,
        created_at=datetime.now(timezone.utc),
        model_responses=responses,
        synthesis=synthesis,
    )


async def ask_council_streaming(
    config: CouncilConfig,
    question: str,
) -> AsyncIterator[CouncilStreamEvent]:
    responses: list[ModelResponse] = []

    async for event in stream_question(config, question):
        yield event
        if isinstance(event, ModelDoneEvent):
            responses.append(
                ModelResponse(
                    model_id=event.model_id,
                    response=event.response,
                    error=event.error,
                )
            )

    try:
        synthesis: CouncilSynthesis = await council_synthesised_answer(config, question, responses)
    except SynthesisError as exc:
        yield ErrorEvent(message=str(exc))
        return

    yield SynthDoneEvent(synthesis=synthesis)
