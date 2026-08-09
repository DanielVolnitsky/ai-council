# Written against the contract, not a layout: nothing asserts a particular
# delimiter or ordering.  Any format that attributes every answer to its model
# id, keeps the answers intact, and separates them unambiguously passes.

import pytest

from core.prompts import SYNTHESIS_SYSTEM_PROMPT, synthesis_input
from core.types import CouncilSynthesis, ModelResponse

RESPONSES: list[ModelResponse] = [
    ModelResponse(model_id="openai:gpt-4o", response="Yes, because of X."),
    ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No, because of Y."),
]


def test_system_prompt_loads_from_the_markdown_file():
    """A packaging mistake that leaves synthesizer.md out of the wheel surfaces
    here, rather than as a mysteriously worse synthesis at runtime."""
    assert SYNTHESIS_SYSTEM_PROMPT.startswith("You are the synthesizer of a council")
    assert len(SYNTHESIS_SYSTEM_PROMPT) > 500


def test_system_prompt_documents_every_synthesis_section():
    """The one thing that can silently drift: a field added to CouncilSynthesis
    without a matching section in the prompt is a field the model must fill with
    no guidance."""
    documented: set[str] = {
        line.removeprefix("## ").strip()
        for line in SYNTHESIS_SYSTEM_PROMPT.splitlines()
        if line.startswith("## ")
    }

    assert documented == set(CouncilSynthesis.model_fields)


def test_question_and_every_answer_are_present():
    prompt: str = synthesis_input("is it worth it?", RESPONSES)

    assert "is it worth it?" in prompt
    assert "Yes, because of X." in prompt
    assert "No, because of Y." in prompt


def test_every_model_id_is_present():
    """The synthesizer echoes these ids back in disagreements / verdict /
    unique_insights.  An id it never saw is an id it will invent."""
    prompt: str = synthesis_input("is it worth it?", RESPONSES)

    assert "openai:gpt-4o" in prompt
    assert "anthropic:claude-haiku-4-5" in prompt


def test_each_answer_follows_its_own_model_id():
    """Attribution, not just presence: each answer must appear after its own id
    and before the next model's, so the mapping is unambiguous."""
    prompt: str = synthesis_input("is it worth it?", RESPONSES)

    gpt_id_at: int = prompt.index("openai:gpt-4o")
    gpt_answer_at: int = prompt.index("Yes, because of X.")
    claude_id_at: int = prompt.index("anthropic:claude-haiku-4-5")
    claude_answer_at: int = prompt.index("No, because of Y.")

    assert gpt_id_at < gpt_answer_at < claude_id_at < claude_answer_at


def test_an_empty_answer_is_still_attributed():
    """A model that succeeded but said nothing abstained — the synthesizer should
    see the abstention rather than a council that appears one member smaller."""
    prompt: str = synthesis_input("is it worth it?", [
        ModelResponse(model_id="openai:gpt-4o", response="Yes, because of X."),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response=""),
    ])

    assert "anthropic:claude-haiku-4-5" in prompt


def test_markdown_in_an_answer_does_not_blur_the_boundary():
    """Models write markdown.  If a model's own "## Summary" heading can pass for
    a structural marker of the prompt, the synthesizer can misattribute half an
    answer."""
    prompt: str = synthesis_input("is it worth it?", [
        ModelResponse(
            model_id="openai:gpt-4o",
            response="## Summary\n\nYes.\n\n---\n\n### Caveats\n\nCost.",
        ),
        ModelResponse(model_id="anthropic:claude-haiku-4-5", response="No."),
    ])

    first_answer_at: int = prompt.index("## Summary")
    second_id_at: int = prompt.index("anthropic:claude-haiku-4-5")
    between: str = prompt[first_answer_at:second_id_at]

    assert "Cost." in between, "the first answer must survive intact up to the boundary"
    assert between.strip() != "", "there must be *something* separating the answers"


@pytest.mark.parametrize("count", [1, 5])
def test_scales_from_one_member_to_many(count: int):
    responses: list[ModelResponse] = [
        ModelResponse(model_id=f"provider:model-{i}", response=f"answer {i}")
        for i in range(count)
    ]

    prompt: str = synthesis_input("is it worth it?", responses)

    for i in range(count):
        assert f"provider:model-{i}" in prompt
        assert f"answer {i}" in prompt
