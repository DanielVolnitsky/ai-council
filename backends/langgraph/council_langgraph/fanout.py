# Fan-out: ask every enabled council model the same question, concurrently.
#
# Failure isolation: a council of N models is useful with N-1 answers, so a
# failure becomes an error-carrying ModelResponse rather than an exception that
# aborts the round.  Nothing is swallowed — the error text is persisted and
# surfaced to the client.

from __future__ import annotations

import asyncio
import os
from typing import TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from core.config import CouncilConfig, ModelConfig
from core.types import ModelResponse


class ChatModelKwargs(TypedDict, total=False):
    # total=False: both keys are optional, and a provider may need neither.
    api_key: str
    base_url: str


def build_chat_model(model_config: ModelConfig) -> BaseChatModel:
    # Raises KeyError when api_key_env names an unset variable.  That is a
    # deployment error, which _ask_model turns into a per-model error marker
    # rather than letting it abort the fan-out.
    kwargs: ChatModelKwargs = {}
    if model_config.api_key_env is not None:
        kwargs["api_key"] = os.environ[model_config.api_key_env]
    if model_config.base_url is not None:
        kwargs["base_url"] = model_config.base_url

    # init_chat_model's overloads narrow the return to BaseChatModel whenever
    # `model` is a str and `configurable_fields` is unset — both true here — but
    # a type checker cannot pick an overload through a **kwargs unpack.  The cast
    # restates that guarantee; it is not a runtime check.
    return cast(BaseChatModel, init_chat_model(model_config.id, **kwargs))


def _response_text(message: BaseMessage) -> str:
    # Providers may return typed content blocks instead of a string.  Only text
    # blocks mean anything to the council, so the rest are dropped.
    if isinstance(message.content, str):
        return message.content

    return "".join(
        block["text"]
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def _ask_model(model_config: ModelConfig, question: str) -> ModelResponse:
    try:
        chat_model: BaseChatModel = build_chat_model(model_config)
        message: AIMessage = await chat_model.ainvoke(question)
    except Exception as exc:
        # Bare `except Exception` is deliberate: provider SDKs raise a wide and
        # undocumented range of errors, and any of them must degrade this one
        # model rather than the whole council.  The type is kept in the message
        # so the cause stays identifiable downstream.
        return ModelResponse(
            model_id=model_config.id,
            response="",
            error=f"{type(exc).__name__}: {exc}",
        )

    return ModelResponse(model_id=model_config.id, response=_response_text(message))


async def fanout_question(config: CouncilConfig, question: str) -> list[ModelResponse]:
    return await asyncio.gather(
        *(_ask_model(model_config, question) for model_config in config.enabled_models)
    )
