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
GO_FIXTURE = FIXTURES / "go_fixture"
NODE_FIXTURE = FIXTURES / "node_fixture"
RUST_FIXTURE = FIXTURES / "rust_fixture"


async def _can_connect(dsn: str) -> bool:
    try:
        conn = await asyncpg.connect(dsn, timeout=2)
    except Exception:
        return False
    await conn.close()
    return True


@pytest_asyncio.fixture(scope="session", loop_scope="session")
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


@pytest_asyncio.fixture(loop_scope="session")
async def clean_repo(pg_pool):
    """Yield (pool, repo_id). Inserts a `repos` row first so the FK on
    `files.repo_id → repos.id` is satisfied; deleting the repos row at
    teardown cascades through files / nodes / definitions / chunks."""
    name = f"test-{os.urandom(8).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name,
        )
    yield pg_pool, repo_id
    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)


@pytest_asyncio.fixture(loop_scope="session")
async def two_repos(pg_pool):
    """Yield (pool, repo_id_a, repo_id_b). Used by cross-repo / MMR tests
    that need two distinct repos coexisting in the same DB."""
    name_a = f"test-a-{os.urandom(8).hex()}"
    name_b = f"test-b-{os.urandom(8).hex()}"
    async with pg_pool.acquire() as conn:
        a = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name_a,
        )
        b = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name_b,
        )
    yield pg_pool, a, b
    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM repos WHERE id = ANY($1::bigint[])", [a, b])


@pytest.fixture
def python_fixture_root() -> Path:
    return PYTHON_FIXTURE


@pytest.fixture
def solidity_fixture_root() -> Path:
    return SOLIDITY_FIXTURE


@pytest.fixture
def go_fixture_root() -> Path:
    return GO_FIXTURE


@pytest.fixture
def node_fixture_root() -> Path:
    return NODE_FIXTURE


@pytest.fixture
def rust_fixture_root() -> Path:
    return RUST_FIXTURE
