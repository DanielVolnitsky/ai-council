from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from core.domain import CouncilSynthesis


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
class SynthDoneEvent:
    synthesis: CouncilSynthesis
    event: ClassVar[str] = "synth_done"


@dataclass
class ErrorEvent:
    message: str
    event: ClassVar[str] = "error"


CouncilStreamEvent = (
    SessionStartEvent | ModelTokenEvent | ModelDoneEvent | SynthDoneEvent | ErrorEvent
)
