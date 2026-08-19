from __future__ import annotations

import uuid
from datetime import datetime, timezone

import asyncpg

from core.types import CouncilResult, CouncilSynthesis, ModelResponse, SessionSummary


_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        question   TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_responses (
        id         BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id),
        model_id   TEXT NOT NULL,
        response   TEXT NOT NULL,
        error      TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS syntheses (
        session_id     TEXT PRIMARY KEY REFERENCES sessions(id),
        synthesis_json TEXT NOT NULL
    )
    """,
]


async def init_db(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)

    async with pool.acquire() as conn:
        for statement in _DDL_STATEMENTS:
            await conn.execute(statement)

    return pool


async def create_session(conn: asyncpg.Connection, question: str) -> str:
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    await conn.execute(
        "INSERT INTO sessions (id, question, created_at) VALUES ($1, $2, $3)",
        session_id, question, created_at,
    )
    return session_id


async def save_model_response(
    conn: asyncpg.Connection,
    session_id: str,
    model_id: str,
    response: str,
    error: str | None,
) -> None:
    await conn.execute(
        "INSERT INTO model_responses (session_id, model_id, response, error) "
        "VALUES ($1, $2, $3, $4)",
        session_id, model_id, response, error,
    )


async def save_synthesis(
    conn: asyncpg.Connection,
    session_id: str,
    synthesis: CouncilSynthesis,
) -> None:
    await conn.execute(
        """
        INSERT INTO syntheses (session_id, synthesis_json)
        VALUES ($1, $2)
        ON CONFLICT (session_id) DO UPDATE SET synthesis_json = EXCLUDED.synthesis_json
        """,
        session_id, synthesis.model_dump_json(),
    )


async def get_session(
    conn: asyncpg.Connection,
    session_id: str,
) -> CouncilResult | None:
    session_row = await conn.fetchrow(
        "SELECT id, question, created_at FROM sessions WHERE id = $1",
        session_id,
    )
    if session_row is None:
        return None

    response_rows = await conn.fetch(
        "SELECT model_id, response, error FROM model_responses WHERE session_id = $1",
        session_id,
    )

    synthesis_row = await conn.fetchrow(
        "SELECT synthesis_json FROM syntheses WHERE session_id = $1",
        session_id,
    )
    if synthesis_row is None:
        return None

    model_responses = [
        ModelResponse(
            model_id=row["model_id"],
            response=row["response"],
            error=row["error"],
        )
        for row in response_rows
    ]

    synthesis = CouncilSynthesis.model_validate_json(synthesis_row["synthesis_json"])

    return CouncilResult(
        session_id=session_row["id"],
        question=session_row["question"],
        created_at=session_row["created_at"],
        model_responses=model_responses,
        synthesis=synthesis,
    )


async def list_sessions(conn: asyncpg.Connection) -> list[SessionSummary]:
    rows = await conn.fetch(
        "SELECT id, question, created_at FROM sessions ORDER BY created_at DESC",
    )
    return [
        SessionSummary(id=row["id"], question=row["question"], created_at=row["created_at"])
        for row in rows
    ]
