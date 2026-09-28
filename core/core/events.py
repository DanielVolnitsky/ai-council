from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import TypeAdapter

from core.domain import CouncilSynthesis, ModelResponse


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

    def to_model_response(self) -> ModelResponse:
        return ModelResponse(model_id=self.model_id, response=self.response, error=self.error)


@dataclass
class SynthDoneEvent:
    synthesis: CouncilSynthesis
    event: ClassVar[str] = "synth_done"


@dataclass
class ErrorEvent:
    message: str
    event: ClassVar[str] = "error"


ModelStreamEvent = ModelTokenEvent | ModelDoneEvent

CouncilStreamEvent = (
    SessionStartEvent | ModelTokenEvent | ModelDoneEvent | SynthDoneEvent | ErrorEvent
)

_STREAM_EVENT_ADAPTER: TypeAdapter[CouncilStreamEvent] = TypeAdapter(CouncilStreamEvent)


def sse_message(event: CouncilStreamEvent) -> str:
    data: str = _STREAM_EVENT_ADAPTER.dump_json(event).decode()
    return f"event: {event.event}\ndata: {data}\n\n"
