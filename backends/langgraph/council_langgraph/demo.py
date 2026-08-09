# backends/langgraph/council_langgraph/demo.py
#
# Throwaway development harness: ask the council one question, print every
# model's raw answer, then print the synthesis of them.  Deliberately NOT the
# CLI from the project plan — that one talks to the FastAPI backend over HTTP
# and has history commands.  This one calls fanout_question() and synthesize()
# in-process so a real provider round trip can be observed before any of the
# HTTP layer exists.  Delete it (or keep it as a debug tool) once the real CLI
# lands.
#
# Run it from the repository root, because load_config() resolves "config.yaml"
# relative to the current working directory.  --package selects this workspace
# member without changing the working directory (unlike --directory), and is
# required because the root workspace package does not depend on this one:
#
#     uv run --package council-langgraph python -m council_langgraph.demo "is it worth it?"

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from core.config import CouncilConfig, load_config
from core.types import CouncilSynthesis, ModelResponse

from council_langgraph.fanout import fanout_question
from council_langgraph.synthesis import SynthesisError, council_synthesised_answer


def _format_failure(response: ModelResponse) -> str:
    return f"=== {response.model_id} ===\nERROR: {response.error}"


def _format_successful_answer(response: ModelResponse) -> str:
    return f"=== {response.model_id} ===\n{response.response}"


def _format_response(response: ModelResponse) -> str:
    if response.error is not None:
        return _format_failure(response)
    return _format_successful_answer(response)


def _format_synthesis(synthesis: CouncilSynthesis) -> str:
    return f"=== SYNTHESIS ===\n{synthesis.model_dump_json(indent=2)}"


async def _ask_council(question: str) -> int:
    load_dotenv()

    config: CouncilConfig = load_config()
    responses: list[ModelResponse] = await fanout_question(config, question)

    for response in responses:
        print(_format_response(response))
        print()

    try:
        synthesis: CouncilSynthesis = await council_synthesised_answer(config, question, responses)
    except SynthesisError as exc:
        # stdout is block-buffered when redirected to a file or pipe, while
        # stderr is unbuffered — so without this flush the message below would
        # overtake the answers it refers to.
        sys.stdout.flush()
        print(f"Synthesis failed: {exc}", file=sys.stderr)
        return 1

    print(_format_synthesis(synthesis))
    return 0


def main() -> int:
    question: str = " ".join(sys.argv[1:]).strip()
    if not question:
        print(
            "usage: uv run --package council-langgraph "
            'python -m council_langgraph.demo "your question"',
            file=sys.stderr,
        )
        return 2
    return asyncio.run(_ask_council(question))


if __name__ == "__main__":
    # sys.exit() sets the process exit status from main()'s return value.
    sys.exit(main())
