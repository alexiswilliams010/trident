"""Phase 4a acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.chunk_assembler import assemble_chunks, count_tokens
from core.extractor import index_repo
from core.heuristic_resolver import resolve_repo_imports
from core.semantic_resolver import resolve_repo


async def _full_pipeline(pool, repo_id: int, root: Path):
    await index_repo(pool, repo_id, root)
    await resolve_repo(pool, repo_id)
    await resolve_repo_imports(pool, repo_id)
    return await assemble_chunks(pool, repo_id)


# ────────────────────────────────────────────────────────────────────
# Token counting smoke
# ────────────────────────────────────────────────────────────────────


def test_count_tokens_basic():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


# ────────────────────────────────────────────────────────────────────
# Python fixture
# ────────────────────────────────────────────────────────────────────


async def test_assemble_python_chunks(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    stats = await _full_pipeline(pool, repo_id, python_fixture_root)

    # Each function gets a function-level chunk; each module gets a module chunk.
    assert stats.n_function > 0
    assert stats.n_module >= 4  # 4 Python files including __init__.py

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.granularity, c.token_count, c.metadata, c.content,
                   d.qualified_name
            FROM chunks c JOIN files f ON f.id=c.file_id
            LEFT JOIN definitions d ON d.id=c.anchor_def_id
            WHERE f.repo_id=$1
            ORDER BY c.id
            """,
            repo_id,
        )
        by_anchor: dict[tuple[str, str], dict] = {}
        for r in rows:
            by_anchor[(r["qualified_name"], r["granularity"])] = r

        # Calculator.add (function) chunk should mention helper as a callee.
        add_fn = by_anchor[("main.Calculator.add", "function")]
        assert "helper" in add_fn["content"]
        # Metadata is well-formed JSON inside the chunk.
        assert "tsgrep-meta" in add_fn["content"]
        meta = json.loads(add_fn["metadata"])
        assert meta["anchor"] == "main.Calculator.add"
        assert meta["language"] == "python"
        assert meta["granularity"] == "function"

        # Cross-module chunk for run() should reference Calculator (its callee).
        run_xm = by_anchor.get(("main.run", "cross-module"))
        assert run_xm is not None
        assert "Calculator" in run_xm["content"]
        assert run_xm["token_count"] <= 2048

        # Module chunk for main.py should list its imports + dependencies.
        main_module = by_anchor[("main", "module")]
        main_meta = json.loads(main_module["metadata"])
        assert "main.Calculator" in main_meta["dependencies"]
        assert "imports" in main_module["content"]


async def test_chunk_metadata_lists_external_deps(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, python_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT metadata FROM chunks c "
            "JOIN files f ON f.id=c.file_id "
            "JOIN definitions d ON d.id=c.anchor_def_id "
            "WHERE f.repo_id=$1 AND d.qualified_name='main' AND c.granularity='module'",
            repo_id,
        )
        assert row is not None
        meta = json.loads(row["metadata"])
        # `import requests` is the external import in main.py.
        assert "requests" in meta["external_deps"]


# ────────────────────────────────────────────────────────────────────
# Solidity fixture
# ────────────────────────────────────────────────────────────────────


async def test_assemble_solidity_chunks(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    stats = await _full_pipeline(pool, repo_id, solidity_fixture_root)
    assert stats.n_function > 0

    async with pool.acquire() as conn:
        # Vault.deposit (function) — mentions balances + Token.transfer.
        row = await conn.fetchrow(
            """
            SELECT c.content, c.metadata
            FROM chunks c JOIN definitions d ON d.id=c.anchor_def_id
            JOIN files f ON f.id=c.file_id
            WHERE f.repo_id=$1 AND d.qualified_name='Vault.Vault.deposit'
              AND c.granularity='function'
            """,
            repo_id,
        )
        assert row is not None
        content = row["content"]
        assert "balances" in content              # state-var dependency
        assert "transfer" in content              # callee signature/body
        meta = json.loads(row["metadata"])
        assert "Vault.Vault.balances" in meta["dependencies"]


async def test_chunk_idempotency(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    first = await _full_pipeline(pool, repo_id, python_fixture_root)
    second = await assemble_chunks(pool, repo_id)
    # Re-running must not insert duplicates and must mostly hit the unchanged path.
    assert first.total == second.total
    assert second.n_inserted == 0
    assert second.n_unchanged == first.total
