from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from core.domain import ModelResponse

ROUND_TRACE: str = "round_1"
SYNTHESIS_SPAN: str = "synthesis"


class SynthesisTraceInput(TypedDict):
    question: str
    responses: list[ModelResponse]


def new_session_id() -> str:
    return str(uuid4())
