from __future__ import annotations

from importlib.resources import files

from core.types import ModelResponse


def _load_prompt(filename: str) -> str:
    return files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SYNTHESIS_SYSTEM_PROMPT: str = _load_prompt("synthesizer.md")


def synthesis_input(question: str, successful_responses: list[ModelResponse]) -> str:
    result: str = f"=== QUESTION: {question}\n"
    for sr in successful_responses:
        result += f"\n=== MODEL {sr.model_id} RESPONSE:\n{sr.response}\n"
    return result
