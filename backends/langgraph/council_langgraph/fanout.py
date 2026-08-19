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
    api_key: str
    base_url: str


def build_chat_model(model_config: ModelConfig) -> BaseChatModel:
    kwargs: ChatModelKwargs = {}
    if model_config.api_key_env is not None:
        kwargs["api_key"] = os.environ[model_config.api_key_env]
    if model_config.base_url is not None:
        kwargs["base_url"] = model_config.base_url

    return cast(BaseChatModel, init_chat_model(model_config.id, **kwargs))


def _response_text(message: BaseMessage) -> str:
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
