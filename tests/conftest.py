"""Shared pytest fixtures.

Tests that need Postgres use the `pg_pool` fixture. If Postgres is not
reachable, those tests are skipped with a clear message so the suite can
still run partially in CI without a DB.
"""

from __future__ import annotations

import os
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

from db.connection import apply_migrations, create_pool, get_dsn

FIXTURES = Path(__file__).parent / "fixtures"
PYTHON_FIXTURE = FIXTURES / "python_fixture"
SOLIDITY_FIXTURE = FIXTURES / "solidity_foundry_fixture"


async def _can_connect(dsn: str) -> bool:
    try:
        conn = await asyncpg.connect(dsn, timeout=2)
    except Exception:
        return False
    await conn.close()
    return True


@pytest_asyncio.fixture(scope="session")
async def pg_pool():
    dsn = get_dsn()
    if not await _can_connect(dsn):
        pytest.skip(f"Postgres not reachable at {dsn}; start docker-compose first.")
    pool = await create_pool(dsn, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        await apply_migrations(conn)
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def clean_repo(pg_pool):
    """Yield (pool, repo_id). Removes all rows for that repo_id afterwards."""
    # Use process pid + a counter via attribute on the fixture function.
    repo_id = int.from_bytes(os.urandom(4), "big") % 1_000_000_000
    yield pg_pool, repo_id
    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM files WHERE repo_id=$1", repo_id)


@pytest.fixture
def python_fixture_root() -> Path:
    return PYTHON_FIXTURE


@pytest.fixture
def solidity_fixture_root() -> Path:
    return SOLIDITY_FIXTURE
