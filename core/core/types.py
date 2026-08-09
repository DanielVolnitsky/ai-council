# Shared data shapes used by both backends, the DB layer, and the API contract.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, TypedDict

from pydantic import BaseModel


class Disagreement(BaseModel):
    point: str
    models_for: list[str]
    models_against: list[str]


class ModelInsights(BaseModel):
    # A list of these rather than a model_id -> insights mapping: a mapping with
    # free-form keys compiles to a JSON schema whose value type providers do not
    # reliably honour, and which OpenAI's strict mode rejects outright.
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
    # one entry per model that contributed something no other model did
    unique_insights: list[ModelInsights]
    blind_spots: list[str]
    takeaways: list[str]


class ModelResponse(BaseModel):
    # A failed model is included with response="" and error set, rather than
    # omitted: callers must check `error is not None` to tell failure from an
    # empty answer.
    model_id: str
    response: str
    error: str | None = None


class CouncilResult(BaseModel):
    session_id: str
    question: str
    created_at: datetime
    model_responses: list[ModelResponse]
    synthesis: CouncilSynthesis


class SessionSummary(TypedDict):
    id: str
    question: str
    created_at: datetime


# Each dataclass below is one SSE frame emitted by POST /api/council/ask/stream.
# `event` is a ClassVar so dataclasses.asdict() yields only the JSON payload
# while `instance.event` still names the frame.

@dataclass
class SessionStartEvent:
    session_id: str
    event: ClassVar[str] = "session_start"


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
class SynthTokenEvent:
    token: str
    event: ClassVar[str] = "synth_token"


@dataclass
class SynthDoneEvent:
    synthesis: CouncilSynthesis
    event: ClassVar[str] = "synth_done"


@dataclass
class ErrorEvent:
    # The stream is closed immediately after this event.
    message: str
    event: ClassVar[str] = "error"
