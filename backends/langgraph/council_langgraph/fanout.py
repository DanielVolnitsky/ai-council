from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from core.config import CouncilConfig, ModelConfig
from core.events import ModelDoneEvent, ModelStreamEvent, ModelTokenEvent
from core.domain import ModelResponse

from council_langgraph.tracing import traced_run_config


class ChatModelKwargs(TypedDict, total=False):
    api_key: str
    base_url: str


def build_chat_model(model_config: ModelConfig) -> BaseChatModel:
    kwargs: ChatModelKwargs = {}
    api_key: str | None = model_config.api_key
    if api_key is not None:
        kwargs["api_key"] = api_key
    if model_config.base_url is not None:
        kwargs["base_url"] = model_config.base_url

    return cast(BaseChatModel, init_chat_model(model_config.id, **kwargs))


async def _ask_model(model_config: ModelConfig, question: str) -> ModelResponse:
    try:
        chat_model: BaseChatModel = build_chat_model(model_config)
        message: AIMessage = await chat_model.ainvoke(
            question, config=traced_run_config(model_config.id)
        )
    except Exception as exc:
        return ModelResponse(
            model_id=model_config.id,
            response="",
            error=f"{type(exc).__name__}: {exc}",
        )

    return ModelResponse(model_id=model_config.id, response=cast(str, message.content))


async def fanout_question(config: CouncilConfig, question: str) -> list[ModelResponse]:
    return await asyncio.gather(
        *(_ask_model(model_config, question) for model_config in config.enabled_models)
    )


async def _stream_model(
    model_config: ModelConfig,
    question: str,
    events: asyncio.Queue[ModelStreamEvent],
) -> None:
    chunks: list[str] = []
    error: str | None = None
    try:
        chat_model: BaseChatModel = build_chat_model(model_config)
        async for chunk in chat_model.astream(question, config=traced_run_config(model_config.id)):
            token: str = cast(str, chunk.content)
            if not token:
                continue
            chunks.append(token)
            await events.put(ModelTokenEvent(model_id=model_config.id, token=token))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    await events.put(
        ModelDoneEvent(
            model_id=model_config.id,
            response="" if error else "".join(chunks),
            error=error,
        )
    )


async def stream_question(
    config: CouncilConfig,
    question: str,
) -> AsyncIterator[ModelStreamEvent]:
    events: asyncio.Queue[ModelStreamEvent] = asyncio.Queue()
    tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(_stream_model(model_config, question, events))
        for model_config in config.enabled_models
    ]

    unfinished: int = len(tasks)
    try:
        while unfinished:
            event = await events.get()
            yield event
            if isinstance(event, ModelDoneEvent):
                unfinished -= 1
    finally:
        for task in tasks:
            task.cancel()
