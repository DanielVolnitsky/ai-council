# core/tests/test_db.py
#
# Unit tests for the PostgreSQL persistence layer.
#
# These tests require a running PostgreSQL instance.  Start one with:
#   docker compose up db -d
#
# The connection string defaults to the local Docker Compose database.
# Override by setting TEST_DATABASE_URL in your environment.
#
# Test isolation — transaction rollback:
#   Each test runs inside a transaction that is rolled back when the test
#   finishes, leaving the database in exactly the state it was before the
#   test started.  No TRUNCATE or DROP needed between tests.
#
#   This is the standard "transactional test case" pattern — identical to
#   Spring's @Transactional on test classes where the transaction is rolled
#   back rather than committed.
#
# Event loop scope:
#   asyncpg connections and pools are bound to the asyncio event loop they
#   were created in.  pytest-asyncio creates a *new* event loop per test
#   function by default.  To avoid "Future attached to a different loop"
#   errors, both the pool fixture and the conn fixture must be function-scoped
#   so they are created and destroyed within the same loop as the test.
#
#   Creating a pool per test adds ~1 connection round-trip of overhead, which
#   is negligible for a local test suite.

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
from core.types import CouncilSynthesis, Disagreement, ModelResponse, Verdict

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://council:council@localhost:5432/council_test",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_pool():
    """
    Create a fresh connection pool for each test function.

    Function scope (the default) ensures the pool is created in the same
    event loop as the test itself.  asyncpg binds connections to the event
    loop they were created in; mixing loops causes RuntimeError.

    Teardown closes all connections in the pool, releasing them back to PG.
    """
    pool = await init_db(TEST_DSN)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def conn(db_pool: asyncpg.Pool):
    """
    Provide an open transaction for one test, then roll it back.

    `scope` defaults to "function" — each test gets its own transaction.

    How it works:
    1. Acquire a connection from the pool.
    2. Start an explicit transaction (`tr.start()`).
    3. Yield the connection to the test — the test sees a clean DB.
    4. After the test returns (pass or fail), roll back the transaction.
       Every INSERT / UPDATE the test made is undone; the DB is unchanged.

    asyncpg.Connection.transaction() returns a Transaction object.  It must
    be started with await tr.start() — unlike aiosqlite where BEGIN is
    implicit.
    """
    async with db_pool.acquire() as c:
        tr = c.transaction()
        await tr.start()
        yield c
        await tr.rollback()


# ---------------------------------------------------------------------------
# Helper: a complete CouncilSynthesis for use across tests
# ---------------------------------------------------------------------------

def _make_synthesis() -> CouncilSynthesis:
    return CouncilSynthesis(
        summary="Both models agreed on the core answer.",
        consensus=["The answer is 42.", "Reasoning was clear."],
        disagreements=[
            Disagreement(
                point="Tone of response",
                models_for=["openai/gpt-4o"],
                models_against=["anthropic/claude-3-5-sonnet-20241022"],
            )
        ],
        verdict=Verdict(
            strongest="openai/gpt-4o",
            weakest="anthropic/claude-3-5-sonnet-20241022",
            justification="GPT provided a more structured answer.",
        ),
        unique_insights={
            "openai/gpt-4o": ["Mentioned the philosophical angle."],
            "anthropic/claude-3-5-sonnet-20241022": [],
        },
        blind_spots=["Neither model addressed edge cases."],
        takeaways=["Trust GPT for structured answers."],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_create_session_returns_uuid(conn):
    session_id = await create_session(conn, "What is the meaning of life?")

    assert len(session_id) == 36   # UUID v4: 8-4-4-4-12 hex digits + 4 dashes
    assert session_id.count("-") == 4


async def test_get_session_returns_none_for_missing_id(conn):
    result = await get_session(conn, "00000000-0000-0000-0000-000000000000")

    assert result is None


async def test_get_session_returns_none_when_synthesis_absent(conn):
    """
    A session with model responses but no synthesis is still in-progress;
    get_session must return None rather than a partial CouncilResult.
    """
    session_id = await create_session(conn, "Partial session")
    await save_model_response(conn, session_id, "openai/gpt-4o", "42", None)

    result = await get_session(conn, session_id)

    assert result is None


async def test_get_session_returns_full_result(conn):
    """
    After saving two model responses (one success, one failure) and a synthesis,
    get_session returns a CouncilResult matching exactly what was written.
    """
    session_id = await create_session(conn, "What is the meaning of life?")

    await save_model_response(conn, session_id, "openai/gpt-4o", "42", None)
    await save_model_response(
        conn, session_id, "anthropic/claude-3-5-sonnet-20241022", "", "API error"
    )

    synthesis = _make_synthesis()
    await save_synthesis(conn, session_id, synthesis)

    result = await get_session(conn, session_id)

    assert result.session_id == session_id
    assert result.question == "What is the meaning of life?"
    assert result.model_responses == [
        ModelResponse(model_id="openai/gpt-4o", response="42", error=None),
        ModelResponse(
            model_id="anthropic/claude-3-5-sonnet-20241022",
            response="",
            error="API error",
        ),
    ]
    assert result.synthesis == synthesis


async def test_list_sessions_is_empty_initially(conn):
    sessions = await list_sessions(conn)

    assert sessions == []


async def test_list_sessions_returns_newest_first(conn):
    """
    list_sessions orders by created_at DESC.  We capture the actual
    created_at values from the result to build the expected structure —
    this verifies all keys are present, the order is correct, and the
    questions are paired with the right IDs.
    """
    id1 = await create_session(conn, "First question")
    id2 = await create_session(conn, "Second question")

    sessions = await list_sessions(conn)

    assert sessions == [
        {"id": id2, "question": "Second question", "created_at": sessions[0]["created_at"]},
        {"id": id1, "question": "First question",  "created_at": sessions[1]["created_at"]},
    ]
