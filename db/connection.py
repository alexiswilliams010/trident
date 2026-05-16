"""Async Postgres connection helpers + migration runner."""

from __future__ import annotations

import getpass
import os
import zlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def default_dsn() -> str:
    """Default DSN for local Homebrew Postgres.

    Homebrew Postgres allows the running OS user to connect over the local socket / TCP
    """
    user = os.environ.get("PGUSER") or getpass.getuser()
    db = os.environ.get("PGDATABASE")
    if not db:
        raise RuntimeError(
            "DATABASE_URL or PGDATABASE must be set, there is no default DB. "
            "Pass DB=<name> to the Makefile, or export DATABASE_URL."
        )
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    return f"postgresql://{user}@{host}:{port}/{db}"


def get_dsn() -> str:
    return os.environ.get("DATABASE_URL") or default_dsn()


async def create_pool(dsn: str | None = None, **kwargs) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn or get_dsn(), **kwargs)


@asynccontextmanager
async def pool_ctx(dsn: str | None = None) -> AsyncIterator[asyncpg.Pool]:
    pool = await create_pool(dsn)
    try:
        yield pool
    finally:
        await pool.close()


async def apply_migrations(
    conn: asyncpg.Connection,
    migrations_dir: Path | None = None,
) -> list[str]:
    """Apply unapplied .up.sql migrations in order.

    Mirrors the `make db-migrate` target: tracks applied migrations in
    `schema_migrations`. Returns the filenames applied this run.
    """
    if migrations_dir is None:
        migrations_dir = MIGRATIONS_DIR

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename   TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    files = sorted(migrations_dir.glob("*.up.sql"))
    applied_now: list[str] = []
    for path in files:
        already = await conn.fetchval(
            "SELECT 1 FROM schema_migrations WHERE filename = $1",
            path.name,
        )
        if already:
            continue
        sql = path.read_text()
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES ($1)",
                path.name,
            )
        applied_now.append(path.name)
    return applied_now


async def reserve_node_ids(conn: asyncpg.Connection, count: int) -> int:
    """Reserve a contiguous block of `count` ids from nodes_id_seq.

    Returns the first id; the block is [first, first + count).
    """
    return await _reserve_sequence_ids(conn, "nodes_id_seq", count)


async def reserve_definition_ids(conn: asyncpg.Connection, count: int) -> int:
    """Reserve a contiguous block of `count` ids from definitions_id_seq.

    Returns the first id; the block is [first, first + count). Used by Tier 2
    to insert all of a file's definitions in a single batch instead of
    `INSERT ... RETURNING id` once per row.
    """
    return await _reserve_sequence_ids(conn, "definitions_id_seq", count)


async def _reserve_sequence_ids(
    conn: asyncpg.Connection, sequence: str, count: int,
) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    # nextval+setval is racy under concurrent reservers: two backends can
    # each call nextval (getting non-contiguous values) then setval over
    # each other's range, handing out overlapping IDs. A DB-wide advisory
    # lock serializes only the reservation of this specific sequence.
    # crc32 over the sequence name gives a stable lock key across processes.
    lock_key = zlib.crc32(sequence.encode())
    await conn.fetchval("SELECT pg_advisory_lock($1)", lock_key)
    try:
        first = await conn.fetchval(f"SELECT nextval('{sequence}')")
        if count > 1:
            await conn.fetchval(f"SELECT setval('{sequence}', $1)", first + count - 1)
        return first
    finally:
        await conn.fetchval("SELECT pg_advisory_unlock($1)", lock_key)
