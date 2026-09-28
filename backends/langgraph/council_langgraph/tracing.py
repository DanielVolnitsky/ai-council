from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, LangfuseSpan, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler


def new_session_id() -> str:
    return str(uuid4())


@contextmanager
def council_session(session_id: str) -> Iterator[None]:
    with propagate_attributes(session_id=session_id):
        yield


@contextmanager
def council_trace(name: str, input: Any) -> Iterator[LangfuseSpan]:
    langfuse: Langfuse = get_client()
    with langfuse.start_as_current_observation(name=name, as_type="span", input=input) as span:
        yield span


def traced_run_config(run_name: str) -> RunnableConfig:
    return RunnableConfig(callbacks=[CallbackHandler()], run_name=run_name)


def shutdown_tracing() -> None:
    get_client().shutdown()
