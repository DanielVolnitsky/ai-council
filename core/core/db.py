# core/core/db.py
#
# PostgreSQL persistence layer for council sessions, backed by asyncpg.
#
# Why asyncpg instead of aiosqlite?
#   PostgreSQL runs as a separate Docker service, which means:
#   - The database survives process restarts and can be inspected independently.
#   - Multiple backend processes (if ever needed) share the same DB safely.
#   asyncpg is a pure-Python async PostgreSQL driver.  It does NOT wrap the
#   synchronous libpq C library — it speaks the PostgreSQL wire protocol
#   directly, so it integrates cleanly with asyncio without hidden thread
#   hand-offs.
#
#   Java analogy: asyncpg is to PostgreSQL what R2DBC is to a relational DB —
#   a non-blocking driver built for reactive/async runtimes.
#
# Connection pooling:
#   asyncpg uses a connection pool (asyncpg.Pool) as the top-level handle.
#   The pool manages N concurrent connections to PostgreSQL.  Each request
#   borrows a connection for the duration of the operation and returns it.
#   This is the same pattern as HikariCP in Java: you never hold a connection
#   longer than you need it.
#
#   `init_db()` creates and returns the pool.  Callers (FastAPI startup,
#   tests) pass a Connection (acquired from the pool) to each helper.
#
# Placeholder syntax:
#   PostgreSQL uses $1, $2, … instead of SQLite's ?.  asyncpg passes
#   parameters as positional arguments after the SQL string:
#     await conn.execute("INSERT INTO t VALUES ($1, $2)", val1, val2)
#   This mirrors JDBC's PreparedStatement but with positional references.
#
# Transactions:
#   asyncpg auto-commits every `execute` / `fetch` call that is NOT inside
#   an explicit transaction block.  For single-statement helpers this is fine.
#   Tests use explicit transaction rollback for isolation (see test_db.py).
#
# DATABASE_URL environment variable:
#   The connection DSN is read from DATABASE_URL at the call site (backends)
#   or TEST_DATABASE_URL (tests).  Format:
#     postgresql://<user>:<password>@<host>:<port>/<database>
#   Example: postgresql://council:council@localhost:5432/council
#   This keeps credentials out of config.yaml and follows the Twelve-Factor
#   App convention for external service URLs.

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import asyncpg

from core.types import CouncilResult, CouncilSynthesis, ModelResponse, SessionSummary


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Each statement is a separate string so they can be executed one at a time.
# asyncpg.Connection.execute() accepts a single SQL statement; running a
# semicolon-delimited script requires either a loop or conn.executemany().
#
# PostgreSQL differences from SQLite schema:
#   - BIGSERIAL = auto-incrementing BIGINT (equivalent to SQLite's AUTOINCREMENT)
#   - No `NOT NULL` needed on PRIMARY KEY — PG enforces that automatically.
#   - FK constraints are enforced by default in PostgreSQL (unlike SQLite where
#     they are opt-in via PRAGMA foreign_keys = ON).

_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        question   TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL  -- asyncpg maps this to/from Python datetime automatically
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_responses (
        id         BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id),
        model_id   TEXT NOT NULL,
        response   TEXT NOT NULL,  -- empty string on failure
        error      TEXT            -- NULL on success
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS syntheses (
        session_id     TEXT PRIMARY KEY REFERENCES sessions(id),
        synthesis_json TEXT NOT NULL   -- CouncilSynthesis serialised by model_dump_json()
    )
    """,
]


# ---------------------------------------------------------------------------
# Connection initialisation
# ---------------------------------------------------------------------------

async def init_db(dsn: str) -> asyncpg.Pool:
    """
    Create a connection pool to PostgreSQL and apply the schema.

    `dsn` is the full connection string, e.g.:
        postgresql://council:council@localhost:5432/council

    Returns an asyncpg.Pool that callers use as a context-managed borrow point:
        async with pool.acquire() as conn:
            await create_session(conn, "...")

    `CREATE TABLE IF NOT EXISTS` makes this idempotent — safe to call every
    time the backend starts up.  No separate migrations step needed for MVP.

    `min_size=1` keeps one warm connection alive so the first request does
    not pay the TCP + TLS handshake cost.  For a local single-user tool,
    `max_size=5` is plenty.
    """
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)

    # Apply the schema inside a single borrowed connection.
    # `async with pool.acquire() as conn` checks out a connection, runs the
    # block, and returns the connection to the pool — identical in spirit to
    # try-with-resources around a JDBC Connection.
    async with pool.acquire() as conn:
        for statement in _DDL_STATEMENTS:
            await conn.execute(statement)

    return pool


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

async def create_session(conn: asyncpg.Connection, question: str) -> str:
    """
    Insert a new session row and return its generated UUID.

    asyncpg.Connection.execute() takes the SQL then positional args —
    no tuple wrapping needed.
    """
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
    """Persist one model's response (or failure marker) for a session."""
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
    """
    Persist the synthesizer's structured output for a session.

    `ON CONFLICT ... DO UPDATE` is PostgreSQL's upsert syntax — equivalent
    to SQLite's `INSERT OR REPLACE`.  If a synthesis row already exists for
    this session, it is overwritten rather than raising a unique-key error.
    """
    await conn.execute(
        """
        INSERT INTO syntheses (session_id, synthesis_json)
        VALUES ($1, $2)
        ON CONFLICT (session_id) DO UPDATE SET synthesis_json = EXCLUDED.synthesis_json
        """,
        session_id, synthesis.model_dump_json(),
    )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

async def get_session(
    conn: asyncpg.Connection,
    session_id: str,
) -> CouncilResult | None:
    """
    Load a complete council session by ID.

    Returns None if the session does not exist or synthesis has not been
    saved yet (session still in progress).

    asyncpg.Connection.fetchrow() returns an asyncpg.Record (column-name
    accessible, like a dict) or None if no row matched.
    asyncpg.Connection.fetch() returns a list of Records.
    """
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
        # Session exists but synthesis is not complete yet.
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
    """
    Return all sessions as summaries, newest first.

    TIMESTAMPTZ sorts chronologically, so `ORDER BY created_at DESC` is correct.
    """
    rows = await conn.fetch(
        "SELECT id, question, created_at FROM sessions ORDER BY created_at DESC",
    )
    return [
        SessionSummary(id=row["id"], question=row["question"], created_at=row["created_at"])
        for row in rows
    ]
