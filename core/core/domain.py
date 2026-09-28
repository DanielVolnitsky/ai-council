from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

Question = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class Disagreement(BaseModel):
    point: str
    models_for: list[str]
    models_against: list[str]


class ModelInsights(BaseModel):
    model_id: str
    insights: list[str]


class Verdict(BaseModel):
    strongest: str
    weakest: str
    justification: str


class CouncilSynthesis(BaseModel):
    summary: str
    consensus: list[str]
    disagreements: list[Disagreement]
    verdict: Verdict
    unique_insights: list[ModelInsights]
    blind_spots: list[str]
    takeaways: list[str]


class ModelResponse(BaseModel):
    model_id: str
    response: str
    error: str | None = None


class CouncilResult(BaseModel):
    session_id: str
    question: str
    created_at: datetime
    model_responses: list[ModelResponse]
    synthesis: CouncilSynthesis
