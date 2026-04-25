"""Async Postgres connection helpers + migration runner."""

from __future__ import annotations

import getpass
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def default_dsn() -> str:
    """Default DSN for local Homebrew Postgres.

    Homebrew Postgres allows the running OS user to connect over the local
    socket / TCP without a password. The database name defaults to `tsgrep`.
    Override with the DATABASE_URL env var.
    """
    user = os.environ.get("PGUSER") or getpass.getuser()
    db = os.environ.get("PGDATABASE", "tsgrep")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    return f"postgresql://{user}@{host}:{port}/{db}"


def get_dsn() -> str:
    return os.environ.get("DATABASE_URL", default_dsn())


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
    if count <= 0:
        raise ValueError("count must be positive")
    first = await conn.fetchval("SELECT nextval('nodes_id_seq')")
    if count > 1:
        await conn.fetchval("SELECT setval('nodes_id_seq', $1)", first + count - 1)
    return first
