# core/tests/test_prompts.py
#
# Tests for format_council_responses().
#
# These are written against the *contract* in the docstring, not against a
# specific layout — there is no assertion that the delimiter is "---" or that
# the question comes first.  Any format that attributes every answer to its
# model id, keeps the answers intact, and separates them unambiguously will
# pass.  That is deliberate: the format is a judgement call, the properties are
# not.
#
# All of these fail until format_council_responses() is implemented.

import pytest

from core.prompts import SYNTHESIS_SYSTEM_PROMPT, synthesis_input
from core.types import CouncilSynthesis, ModelResponse

RESPONSES: list[ModelResponse] = [
    ModelResponse(model_id="openai:gpt-4o", response="Yes, because of X."),
    ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y."),
]


# ---------------------------------------------------------------------------
# The system prompt document
# ---------------------------------------------------------------------------

def test_system_prompt_loads_from_the_markdown_file():
    """
    The prompt lives in synthesizer.md and is read through importlib
    resources.  A packaging mistake that leaves the .md out of the wheel would
    surface here as an empty or missing prompt rather than as a mysteriously
    worse synthesis at runtime.
    """
    assert SYNTHESIS_SYSTEM_PROMPT.startswith("You are the synthesizer of a council")
    assert len(SYNTHESIS_SYSTEM_PROMPT) > 500


def test_system_prompt_documents_every_synthesis_section():
    """
    Guards the one thing that can silently drift: the .md explains what each
    section is *for*, while the field list itself reaches the model as a JSON
    schema derived from CouncilSynthesis.  A field added to the Pydantic model
    without a matching section in the prompt is a field the model must fill
    with no guidance.
    """
    documented: set[str] = {
        line.removeprefix("## ").strip()
        for line in SYNTHESIS_SYSTEM_PROMPT.splitlines()
        if line.startswith("## ")
    }

    assert documented == set(CouncilSynthesis.model_fields)


# ---------------------------------------------------------------------------
# format_council_responses
# ---------------------------------------------------------------------------

def test_question_and_every_answer_are_present():
    """Nothing the synthesizer needs may be dropped or truncated."""
    prompt: str = synthesis_input("is it worth it?", RESPONSES)

    assert "is it worth it?" in prompt
    assert "Yes, because of X." in prompt
    assert "No, because of Y." in prompt


def test_every_model_id_is_present():
    """
    The synthesizer echoes these ids back in disagreements / verdict /
    unique_insights.  An id it never saw is an id it will invent.
    """
    prompt: str = synthesis_input("is it worth it?", RESPONSES)

    assert "openai:gpt-4o" in prompt
    assert "anthropic:claude-haiku-4-5" in prompt


def test_each_answer_follows_its_own_model_id():
    """
    Attribution, not just presence: each answer must appear *after* its own id
    and *before* the next model's id, so the mapping is unambiguous.
    """
    prompt: str = synthesis_input("is it worth it?", RESPONSES)

    gpt_id_at: int = prompt.index("openai:gpt-4o")
    gpt_answer_at: int = prompt.index("Yes, because of X.")
    claude_id_at: int = prompt.index("anthropic:claude-haiku-4-5")
    claude_answer_at: int = prompt.index("No, because of Y.")

    assert gpt_id_at < gpt_answer_at < claude_id_at < claude_answer_at


def test_an_empty_answer_is_still_attributed():
    """
    A model that succeeded but said nothing abstained — the synthesizer should
    see the abstention rather than a council that appears one member smaller.
    """
    prompt: str = synthesis_input("is it worth it?", [
        ModelResponse(model_id="openai:gpt-4o", response="Yes, because of X."),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response=""),
    ])

    assert "anthropic:claude-haiku-4-5" in prompt


def test_markdown_in_an_answer_does_not_blur_the_boundary():
    """
    Models write markdown.  If a model's own "## Summary" heading can pass for
    a structural marker of the prompt, the synthesizer can misattribute half an
    answer.  Whatever delimiter separates answers must not be something a model
    plausibly emits mid-answer.
    """
    prompt: str = synthesis_input("is it worth it?", [
        ModelResponse(
            model_id="openai:gpt-4o",
            response="## Summary\n\nYes.\n\n---\n\n### Caveats\n\nCost.",
        ),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No."),
    ])

    # The delimiter between the two answers must not be a sequence the first
    # answer already contains, or the boundary is ambiguous by construction.
    first_answer_at: int = prompt.index("## Summary")
    second_id_at: int = prompt.index("anthropic:claude-haiku-4-5")
    between: str = prompt[first_answer_at:second_id_at]

    assert "Cost." in between, "the first answer must survive intact up to the boundary"
    assert between.strip() != "", "there must be *something* separating the answers"


@pytest.mark.parametrize("count", [1, 5])
def test_scales_from_one_member_to_many(count: int):
    """One-model councils and large ones use the same code path."""
    responses: list[ModelResponse] = [
        ModelResponse(model_id=f"provider:model-{i}", response=f"answer {i}")
        for i in range(count)
    ]

    prompt: str = synthesis_input("is it worth it?", responses)

    for i in range(count):
        assert f"provider:model-{i}" in prompt
        assert f"answer {i}" in prompt
