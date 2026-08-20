from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel


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
    question: str
    created_at: datetime
    model_responses: list[ModelResponse]
    synthesis: CouncilSynthesis


@dataclass
class ModelTokenEvent:
    model_id: str
    token: str
    event: ClassVar[str] = "model_token"


@dataclass
class ModelDoneEvent:
    model_id: str
    response: str
    error: str | None = None
    event: ClassVar[str] = "model_done"


@dataclass
class SynthDoneEvent:
    synthesis: CouncilSynthesis
    event: ClassVar[str] = "synth_done"


@dataclass
class ErrorEvent:
    message: str
    event: ClassVar[str] = "error"


CouncilStreamEvent = ModelTokenEvent | ModelDoneEvent | SynthDoneEvent | ErrorEvent
