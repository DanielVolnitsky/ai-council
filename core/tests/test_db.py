import os

import asyncpg
import pytest_asyncio

from core.db import (
    create_session,
    get_session,
    init_db,
    list_sessions,
    save_model_response,
    save_synthesis,
)
from core.types import CouncilSynthesis, Disagreement, ModelInsights, ModelResponse, Verdict

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://council:council@localhost:5432/council_test",
)


@pytest_asyncio.fixture
async def db_pool():
    pool = await init_db(TEST_DSN)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def conn(db_pool: asyncpg.Pool):
    async with db_pool.acquire() as c:
        tr = c.transaction()
        await tr.start()
        yield c
        await tr.rollback()


def _make_synthesis() -> CouncilSynthesis:
    return CouncilSynthesis(
        summary="Both models agreed on the core answer.",
        consensus=["The answer is 42.", "Reasoning was clear."],
        disagreements=[
            Disagreement(
                point="Tone of response",
                models_for=["openai:gpt-4o"],
                models_against=["anthropic:claude-sonnet-5"],
            )
        ],
        verdict=Verdict(
            strongest="openai:gpt-4o",
            weakest="anthropic:claude-sonnet-5",
            justification="GPT provided a more structured answer.",
        ),
        unique_insights=[
            ModelInsights(
                model_id="openai:gpt-4o",
                insights=["Mentioned the philosophical angle."],
            ),
        ],
        blind_spots=["Neither model addressed edge cases."],
        takeaways=["Trust GPT for structured answers."],
    )


async def test_create_session_returns_uuid(conn):
    session_id = await create_session(conn, "What is the meaning of life?")

    assert len(session_id) == 36
    assert session_id.count("-") == 4


async def test_get_session_returns_none_for_missing_id(conn):
    result = await get_session(conn, "00000000-0000-0000-0000-000000000000")

    assert result is None


async def test_get_session_returns_none_when_synthesis_absent(conn):
    session_id = await create_session(conn, "Partial session")
    await save_model_response(conn, session_id, "openai:gpt-4o", "42", None)

    result = await get_session(conn, session_id)

    assert result is None


async def test_get_session_returns_full_result(conn):
    session_id = await create_session(conn, "What is the meaning of life?")

    await save_model_response(conn, session_id, "openai:gpt-4o", "42", None)
    await save_model_response(
        conn, session_id, "anthropic:claude-sonnet-5", "", "API error"
    )

    synthesis = _make_synthesis()
    await save_synthesis(conn, session_id, synthesis)

    result = await get_session(conn, session_id)

    assert result.session_id == session_id
    assert result.question == "What is the meaning of life?"
    assert result.model_responses == [
        ModelResponse(model_id="openai:gpt-4o", response="42", error=None),
        ModelResponse(
            model_id="anthropic:claude-sonnet-5",
            response="",
            error="API error",
        ),
    ]
    assert result.synthesis == synthesis


async def test_list_sessions_is_empty_initially(conn):
    sessions = await list_sessions(conn)

    assert sessions == []


async def test_list_sessions_returns_newest_first(conn):
    id1 = await create_session(conn, "First question")
    id2 = await create_session(conn, "Second question")

    sessions = await list_sessions(conn)

    assert sessions == [
        {"id": id2, "question": "Second question", "created_at": sessions[0]["created_at"]},
        {"id": id1, "question": "First question",  "created_at": sessions[1]["created_at"]},
    ]
