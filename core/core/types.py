# core/core/types.py
#
# Shared data shapes used by both backends, the DB layer, and the API contract.
#
# Two categories:
#   1. Pydantic models  — CouncilSynthesis, ModelResponse, CouncilResult.
#      Pydantic is used here because these types cross the DB and HTTP boundaries:
#      they are serialized to/from JSON (model_dump_json / model_validate_json).
#      Think of it as a combination of Java's record + Jackson's ObjectMapper.
#
#   2. Dataclasses      — SSE event types.
#      These are outbound-only (never deserialize from external input), so
#      Pydantic's validation overhead is unnecessary. Plain dataclasses give
#      us typed fields and dataclasses.asdict() for free.
#
#   3. TypedDict        — SessionSummary.
#      Used as the return type of db.list_sessions() to avoid a bare dict[].
#      TypedDict adds type safety without the overhead of a full class.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, TypedDict

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Synthesis sub-shapes
# ---------------------------------------------------------------------------

class Disagreement(BaseModel):
    """One point of contention, with which models landed on each side."""
    point: str
    models_for: list[str]      # model IDs that agreed with the point
    models_against: list[str]  # model IDs that disagreed


class Verdict(BaseModel):
    """
    The synthesizer's quality verdict across all council responses.

    `strongest` / `weakest` are model IDs (e.g. "openai/gpt-4o").
    `justification` is free text explaining the ranking.
    """
    strongest: str
    weakest: str
    justification: str


# ---------------------------------------------------------------------------
# Core synthesis document
# ---------------------------------------------------------------------------

class CouncilSynthesis(BaseModel):
    """
    The structured seven-section document produced by the synthesizer model.

    The synthesizer is instructed to return JSON matching this schema exactly.
    The backend parses it with model_validate_json(), which raises a
    ValidationError on mismatch — resulting in a 500 (sync) or error SSE event
    (stream).

    Java analogy: a Pydantic BaseModel with strict field validation is similar
    to a Java record with @JsonProperty annotations, but validation runs at
    object construction time rather than at deserialization time.
    """
    summary: str
    consensus: list[str]
    disagreements: list[Disagreement]
    verdict: Verdict
    # model_id -> list of insights unique to that model
    unique_insights: dict[str, list[str]]
    blind_spots: list[str]
    takeaways: list[str]


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

class ModelResponse(BaseModel):
    """
    One model's response to the council question.

    A failed model is *included* in the result
    with response="" and error set, rather than omitted.  Callers must check
    `error is not None` to distinguish failure from an empty response.
    """
    model_id: str
    response: str            # empty string on failure
    error: str | None = None # None on success


class CouncilResult(BaseModel):
    """
    The complete output of one council session.

    This is:
    - the JSON body returned by POST /api/council/ask  (sync endpoint)
    - the payload of the `synth_done` SSE event        (streaming endpoint)
    - the shape reconstructed from db.get_session()
    """
    session_id: str
    question: str
    created_at: datetime
    model_responses: list[ModelResponse]
    synthesis: CouncilSynthesis


# ---------------------------------------------------------------------------
# Session summary (for GET /api/sessions list)
# ---------------------------------------------------------------------------

class SessionSummary(TypedDict):
    """
    Lightweight session row returned by db.list_sessions().

    TypedDict (not Pydantic) because it's an internal DB-to-API bridge type
    that never needs validation — it is constructed directly from DB rows.

    Java analogy: a DTO record with no business logic.
    """
    id: str
    question: str
    created_at: datetime


# ---------------------------------------------------------------------------
# SSE event dataclasses
# ---------------------------------------------------------------------------
#
# Each class maps to one SSE frame emitted by POST /api/council/ask/stream.
# An SSE frame has two parts:
#   event: <type>           ← the `event:` field, e.g. "model_token"
#   data:  <json-payload>   ← everything else, serialized to JSON
#
# `event` is declared as ClassVar so it is a *class-level constant*, not an
# instance field.  This means dataclasses.asdict() omits it from the dict
# (giving us the JSON payload), while `instance.event` still works via class
# lookup (giving us the SSE event name).
#
# Java analogy: a sealed interface with one impl per event type, where the
# interface declares String EVENT_TYPE and each record holds its payload fields.

@dataclass
class SessionStartEvent:
    """First event in every stream. Gives the client the session ID for history."""
    session_id: str
    event: ClassVar[str] = "session_start"


@dataclass
class ModelTokenEvent:
    """One streaming token from a council model, delivered as it arrives."""
    model_id: str
    token: str
    event: ClassVar[str] = "model_token"


@dataclass
class ModelDoneEvent:
    """
    Signals that one model has finished streaming.

    On success: response is the full concatenated text, error is None.
    On failure: response is empty, error holds the exception message.
    """
    model_id: str
    response: str
    error: str | None = None
    event: ClassVar[str] = "model_done"


@dataclass
class SynthTokenEvent:
    """One streaming token from the synthesizer model."""
    token: str
    event: ClassVar[str] = "synth_token"


@dataclass
class SynthDoneEvent:
    """
    Final event. Contains the complete structured synthesis document.

    After this event the SSE stream is closed by the server.
    """
    synthesis: CouncilSynthesis
    event: ClassVar[str] = "synth_done"


@dataclass
class ErrorEvent:
    """
    Emitted when an unrecoverable server error occurs mid-stream.

    The stream is closed immediately after this event.
    """
    message: str
    event: ClassVar[str] = "error"
