"""Inheritance + overrides: schema, intra-file resolution, cross-file linking,
override edge generation, retrieval expansion, and chunk-content enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.chunk_assembler import assemble_chunks
from core.embedder import embed_repo_chunks, make_fake_embedder
from core.extractor import index_repo
from core.heuristic_resolver import resolve_repo_imports
from core.retrieval import hybrid_query, semantic_query
from core.semantic_resolver import resolve_repo


async def _seed(pool, repo_id: int, root: Path):
    await index_repo(pool, repo_id, root)
    await resolve_repo(pool, repo_id)
    await resolve_repo_imports(pool, repo_id)
    await assemble_chunks(pool, repo_id)
    embed_fn, model = make_fake_embedder()
    await embed_repo_chunks(pool, repo_id, embed_fn, model)
    return embed_fn


# ────────────────────────────────────────────────────────────────────
# Solidity
# ────────────────────────────────────────────────────────────────────


async def test_solidity_intra_file_inheritance(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        # SingleExecutorPolicy is Policy → one inherits_edges row.
        # Both defs live in the same file, so resolution is intra-file (certain).
        rows = await conn.fetch(
            """
            SELECT child.qualified_name AS child_name,
                   base.qualified_name  AS base_name,
                   ie.confidence,
                   ie.base_def_id IS NOT NULL AS resolved
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base ON base.id = ie.base_def_id
            JOIN files f ON f.id = child.file_id
            WHERE f.repo_id = $1 AND child.qualified_name = 'Policy.SingleExecutorPolicy'
            """,
            repo_id,
        )
        assert len(rows) == 1, rows
        assert rows[0]["base_name"] == "Policy.Policy"
        assert rows[0]["resolved"] is True
        assert rows[0]["confidence"] == "certain"


async def test_solidity_cross_file_inheritance(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        # LoggingPolicy (PolicyExtended.sol) inherits Policy (Policy.sol) — must
        # be resolved by the heuristic resolver after import linking.
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            JOIN definitions base  ON base.id  = ie.base_def_id
            JOIN files f ON f.id = child.file_id
            WHERE f.repo_id = $1 AND child.qualified_name = 'PolicyExtended.LoggingPolicy'
            """,
            repo_id,
        )
        assert row is not None
        assert row["base_name"] == "Policy.Policy"


async def test_solidity_overrides(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        # SingleExecutorPolicy.isPolicyActive overrides Policy.isPolicyActive.
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            JOIN files f ON f.id = child.file_id
            WHERE f.repo_id = $1
              AND child.qualified_name = 'Policy.SingleExecutorPolicy.isPolicyActive'
            """,
            repo_id,
        )
        assert row is not None
        assert row["base_name"] == "Policy.Policy.isPolicyActive"

        # Cross-file override: LoggingPolicy.isPolicyActive → Policy.isPolicyActive.
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            JOIN files f ON f.id = child.file_id
            WHERE f.repo_id = $1
              AND child.qualified_name = 'PolicyExtended.LoggingPolicy.isPolicyActive'
            """,
            repo_id,
        )
        assert row is not None
        assert row["base_name"] == "Policy.Policy.isPolicyActive"


# ────────────────────────────────────────────────────────────────────
# Python
# ────────────────────────────────────────────────────────────────────


async def test_python_intra_file_inheritance_and_override(
    clean_repo, python_fixture_root: Path,
):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    async with pool.acquire() as conn:
        # StrictPolicy(BasePolicy) → inherits_edges row, intra-file.
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            JOIN definitions base  ON base.id  = ie.base_def_id
            JOIN files f ON f.id = child.file_id
            WHERE f.repo_id = $1 AND child.qualified_name = 'policies.StrictPolicy'
            """,
            repo_id,
        )
        assert row is not None
        assert row["base_name"] == "policies.BasePolicy"
        assert row["confidence"] == "certain"

        # Override: StrictPolicy.is_active → BasePolicy.is_active.
        ovr = await conn.fetch(
            """
            SELECT base.qualified_name AS base_name
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            JOIN files f ON f.id = child.file_id
            WHERE f.repo_id = $1
              AND child.qualified_name LIKE 'policies.StrictPolicy.%'
            ORDER BY base.qualified_name
            """,
            repo_id,
        )
        names = [r["base_name"] for r in ovr]
        assert "policies.BasePolicy.is_active" in names
        assert "policies.BasePolicy.check" in names


# ────────────────────────────────────────────────────────────────────
# Retrieval: bidirectional expansion pulls in overrides
# ────────────────────────────────────────────────────────────────────


async def test_chunk_enrichment_for_overriding_method(
    clean_repo, solidity_fixture_root: Path,
):
    """The chunk for an overriding method should include:
       - an `# overrides:` block with the base method's signature,
       - an `# inheritance chain:` line naming the ancestor classes,
       - metadata fields `overrides` and `inheritance_chain`.
    """
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.content, c.metadata
            FROM chunks c
            JOIN definitions d ON d.id = c.anchor_def_id
            JOIN files f ON f.id = c.file_id
            WHERE f.repo_id = $1
              AND d.qualified_name = 'Policy.SingleExecutorPolicy.isPolicyActive'
              AND c.granularity = 'function'
            """,
            repo_id,
        )
        assert row is not None, "chunk for SingleExecutorPolicy.isPolicyActive missing"
        meta = json.loads(row["metadata"])
        assert meta["overrides"] == "Policy.Policy.isPolicyActive"
        assert "Policy.Policy" in meta["inheritance_chain"]
        # IPolicy is also an ancestor (via Policy is IPolicy).
        assert "Policy.IPolicy" in meta["inheritance_chain"]
        content = row["content"]
        assert "# overrides: Policy.Policy.isPolicyActive" in content
        assert "# inheritance chain:" in content
        # Inherited members from Policy/IPolicy should appear (signatures only).
        assert "onlyManager" in content or "_requireSender" in content


async def test_hybrid_pulls_in_overrides(clean_repo, solidity_fixture_root: Path):
    """Regression: a query that primarily matches a base method should also
    surface its overrides via inheritance/override edges."""
    pool, repo_id = clean_repo
    embed_fn = await _seed(pool, repo_id, solidity_fixture_root)
    # Hand-craft a query that lexically resembles only Policy.isPolicyActive's
    # body; the deterministic stub embedder will rank that base highest. The
    # bidirectional expansion should still pull in the overrides.
    chunks = await hybrid_query(
        pool, repo_id, "is policy active manager check", embed_fn,
        top_k=10, candidate_pool=10,
    )
    qualified = {c.qualified_name for c in chunks}
    # Base in the result set is the seed.
    assert any("Policy.Policy.isPolicyActive" in q for q in qualified if q), qualified
    # At least one override should have been pulled in via the graph.
    overrides = [
        q for q in qualified
        if q and (q.endswith("SingleExecutorPolicy.isPolicyActive")
                  or q.endswith("LoggingPolicy.isPolicyActive"))
    ]
    assert overrides, f"no override surfaced; got: {sorted(qualified)}"
