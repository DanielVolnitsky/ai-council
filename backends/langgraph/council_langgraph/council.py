from __future__ import annotations

from datetime import datetime, timezone

from core.config import CouncilConfig
from core.types import CouncilResult, CouncilSynthesis, ModelResponse

from council_langgraph.fanout import fanout_question
from council_langgraph.synthesis import council_synthesised_answer


async def ask_council(config: CouncilConfig, question: str) -> CouncilResult:
    responses: list[ModelResponse] = await fanout_question(config, question)
    synthesis: CouncilSynthesis = await council_synthesised_answer(config, question, responses)

    return CouncilResult(
        question=question,
        created_at=datetime.now(timezone.utc),
        model_responses=responses,
        synthesis=synthesis,
    )
