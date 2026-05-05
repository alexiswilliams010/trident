"""Shared pytest fixtures.

Tests that need Postgres use the `pg_pool` fixture. If Postgres is not
reachable, those tests are skipped with a clear message so the suite can
still run partially in CI without a DB.

Branch model: every `clean_repo` / `two_repos` fixture also creates a
default branch (named `main`) and yields its branch_id. Existing tests
that destructured `pool, repo_id = clean_repo` will need updating to
`pool, repo_id, branch_id = clean_repo` (and pass branch_id to indexer,
resolver, chunk-assembler, embedder, retrieval, and graph functions).
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
RUST_WORKSPACE_FIXTURE = FIXTURES / "rust_workspace_fixture"


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
    """Yield (pool, repo_id, branch_id). Inserts a `repos` row + a default
    `main` branch so the FK chain is satisfied. Deleting the repos row at
    teardown cascades through branches → branch_files → everything below."""
    name = f"test-{os.urandom(8).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name,
        )
        branch_id = await conn.fetchval(
            "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', TRUE) RETURNING id",
            repo_id,
        )
    yield pg_pool, repo_id, branch_id
    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)


@pytest_asyncio.fixture(loop_scope="session")
async def two_repos(pg_pool):
    """Yield (pool, (repo_a_id, branch_a_id), (repo_b_id, branch_b_id))."""
    name_a = f"test-a-{os.urandom(8).hex()}"
    name_b = f"test-b-{os.urandom(8).hex()}"
    async with pg_pool.acquire() as conn:
        a = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name_a,
        )
        ba = await conn.fetchval(
            "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', TRUE) RETURNING id",
            a,
        )
        b = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name_b,
        )
        bb = await conn.fetchval(
            "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', TRUE) RETURNING id",
            b,
        )
    yield pg_pool, (a, ba), (b, bb)
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


@pytest.fixture
def rust_workspace_fixture_root() -> Path:
    return RUST_WORKSPACE_FIXTURE
