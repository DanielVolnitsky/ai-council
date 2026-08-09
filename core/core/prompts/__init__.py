# core/core/prompts/__init__.py
#
#   This is a package (a directory with __init__.py) rather than a single
#   prompts.py precisely so the .md files can sit next to the code that loads
#   them.  Import paths are unaffected: `from core.prompts import ...` resolves
#   to this file either way.
#
# Why the output schema is not described in the Markdown:
#   synthesis.py calls LangChain's `with_structured_output(CouncilSynthesis)`,
#   which derives a JSON schema from the Pydantic model and hands it to the
#   provider's native structured-output mode (OpenAI response_format, Anthropic
#   tool use).  The field list therefore reaches the model as a machine-readable
#   schema, not as prose.  Restating it in the .md would be a second source of
#   truth that could fall out of sync with core.types.CouncilSynthesis.  What
#   the Markdown *does* carry is the part a schema cannot express: what each
#   section is *for*, and the attribution rules.

from __future__ import annotations

from importlib.resources import files

from core.types import ModelResponse


def _load_prompt(filename: str) -> str:
    return files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SYNTHESIS_SYSTEM_PROMPT: str = _load_prompt("synthesizer.md")


def synthesis_input(question: str, successful_responses: list[ModelResponse]) -> str:
    result : str = f"=== QUESTION: {question}\n"
    for sr in successful_responses:
        result += f"\n=== MODEL {sr.model_id} RESPONSE:\n{sr.response}\n"
    return result
