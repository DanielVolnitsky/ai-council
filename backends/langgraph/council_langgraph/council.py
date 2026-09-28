from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TypedDict

from core.config import CouncilConfig
from core.types import (
    CouncilResult,
    CouncilStreamEvent,
    CouncilSynthesis,
    ErrorEvent,
    ModelDoneEvent,
    ModelResponse,
    SessionStartEvent,
    SynthDoneEvent,
)

from council_langgraph.fanout import fanout_question, stream_question
from council_langgraph.synthesis import SynthesisError, council_synthesised_answer
from council_langgraph.tracing import council_session, council_trace, new_session_id


class SynthesisTraceInput(TypedDict):
    question: str
    responses: list[ModelResponse]


ROUND_TRACE: str = "round_1"
SYNTHESIS_SPAN: str = "synthesis"


async def ask_council(config: CouncilConfig, question: str) -> CouncilResult:
    session_id: str = new_session_id()

    with council_session(session_id), council_trace(ROUND_TRACE, input=question) as round_span:
        responses: list[ModelResponse] = await fanout_question(config, question)
        synthesis: CouncilSynthesis = await _traced_synthesis(config, question, responses)
        round_span.update(output=synthesis)

    return CouncilResult(
        session_id=session_id,
        question=question,
        created_at=datetime.now(timezone.utc),
        model_responses=responses,
        synthesis=synthesis,
    )


async def ask_council_streaming(
    config: CouncilConfig,
    question: str,
) -> AsyncIterator[CouncilStreamEvent]:
    session_id: str = new_session_id()
    yield SessionStartEvent(session_id=session_id)

    responses: list[ModelResponse] = []

    with council_session(session_id), council_trace(ROUND_TRACE, input=question) as round_span:
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
            synthesis: CouncilSynthesis = await _traced_synthesis(config, question, responses)
        except SynthesisError as exc:
            yield ErrorEvent(message=str(exc))
            return

        round_span.update(output=synthesis)

    yield SynthDoneEvent(synthesis=synthesis)


async def _traced_synthesis(
    config: CouncilConfig,
    question: str,
    responses: list[ModelResponse],
) -> CouncilSynthesis:
    synthesis_input: SynthesisTraceInput = {
        "question": question,
        "responses": responses,
    }
    with council_trace(SYNTHESIS_SPAN, input=synthesis_input) as synthesis_span:
        synthesis: CouncilSynthesis = await council_synthesised_answer(config, question, responses)
        synthesis_span.update(output=synthesis)

    return synthesis
