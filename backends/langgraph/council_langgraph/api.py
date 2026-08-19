from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from core.config import CouncilConfig, load_config
from core.types import CouncilResult

from council_langgraph.council import ask_council
from council_langgraph.synthesis import SynthesisError


@lru_cache(maxsize=1)
def get_config() -> CouncilConfig:
    load_dotenv()
    return load_config()


app: FastAPI = FastAPI(title="AI Council")


@app.exception_handler(SynthesisError)
async def _synthesis_error_handler(request: Request, exc: SynthesisError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.post("/api/council/ask", response_model=CouncilResult)
async def ask(
    question: Annotated[str, Body(embed=True, min_length=1, pattern=r"\S")],
    config: CouncilConfig = Depends(get_config),
) -> CouncilResult:
    return await ask_council(config, question)
