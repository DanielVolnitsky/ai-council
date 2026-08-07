# backends/langgraph/council_langgraph/fanout.py
#
# Fan-out: ask every enabled council model the same question, concurrently.
#
# Provider abstraction:
#   LangChain's init_chat_model() resolves a "<provider>:<model>" string into
#   the right chat model class (ChatOpenAI, ChatAnthropic, ChatOllama, ...).
#   ModelConfig.id is already in that exact format, so the config id is passed
#   straight through — no provider lookup table needed here.
#
#   Java analogy: init_chat_model is a factory method keyed by a provider
#   string, similar to DriverManager.getConnection() picking a JDBC driver
#   from the URL scheme.
#
# Failure isolation:
#   A council of N models is useful with N-1 answers.  Each model is asked
#   inside its own try/except, and a failure becomes a ModelResponse with
#   error set rather than an exception that aborts the whole round.  Nothing
#   is swallowed: the error text is carried in the result, persisted by
#   core.db.save_model_response(), and surfaced to the client.

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
    """
    Optional constructor arguments forwarded to the provider's chat model class.

    `total=False` marks every key as optional — the dict is built up key by key
    depending on what the ModelConfig declares, and both keys are absent for a
    provider that needs neither.
    """
    api_key: str
    base_url: str


def build_chat_model(model_config: ModelConfig) -> BaseChatModel:
    """
    Resolve one ModelConfig into a ready-to-call LangChain chat model.

    The API key is read from the environment at build time — config.yaml holds
    only the *name* of the env var (see core.config.ModelConfig), so secrets
    never live in the config file.

    Raises:
        KeyError — if api_key_env names an environment variable that is not
                   set.  This is a deployment error; _ask_model() converts it
                   into a per-model error marker rather than letting it abort
                   the whole fan-out.
    """
    kwargs: ChatModelKwargs = {}
    if model_config.api_key_env is not None:
        kwargs["api_key"] = os.environ[model_config.api_key_env]
    if model_config.base_url is not None:
        kwargs["base_url"] = model_config.base_url

    # init_chat_model is declared as returning BaseChatModel | _ConfigurableModel.
    # Its overloads narrow that to BaseChatModel whenever `model` is a str and
    # `configurable_fields` is left unset — both true here — but a type checker
    # cannot pick an overload through a **kwargs unpack, so it falls back to the
    # union.  The cast restates the guarantee the overload already makes; it is
    # not a runtime check.
    return cast(BaseChatModel, init_chat_model(model_config.id, **kwargs))


def _response_text(message: BaseMessage) -> str:
    """
    Extract plain text from a chat model reply.

    `content` is a str for ordinary text replies, but providers may return a
    list of typed content blocks (text, images, tool calls) instead.  Only the
    text blocks are meaningful to the council, so the rest are dropped.
    """
    if isinstance(message.content, str):
        return message.content

    return "".join(
        block["text"]
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def _ask_model(model_config: ModelConfig, question: str) -> ModelResponse:
    """
    Ask one model the question, converting any failure into an error marker.

    The bare `except Exception` is deliberate: provider SDKs raise a wide and
    undocumented range of errors (auth, rate limit, timeout, malformed
    response), and any of them must degrade this one model rather than the
    whole council.  The exception type is kept in the message so the cause is
    still identifiable downstream.
    """
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
    """
    Ask every enabled model the question concurrently.
    """
    return await asyncio.gather(
        *(_ask_model(model_config, question) for model_config in config.enabled_models)
    )
