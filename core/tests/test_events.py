from core.domain import ModelResponse
from core.events import ModelDoneEvent, ModelTokenEvent, sse_message


def test_done_event_converts_to_its_model_response():
    event: ModelDoneEvent = ModelDoneEvent(
        model_id="openai:gpt-4o", response="", error="TimeoutError: nope"
    )

    assert event.to_model_response() == ModelResponse(
        model_id="openai:gpt-4o", response="", error="TimeoutError: nope"
    )


def test_sse_message_names_the_event_and_carries_its_fields_as_json():
    message: str = sse_message(ModelTokenEvent(model_id="openai:gpt-4o", token="yes"))

    assert message == 'event: model_token\ndata: {"model_id":"openai:gpt-4o","token":"yes"}\n\n'
