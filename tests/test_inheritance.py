"""Inheritance + overrides: schema, intra-file resolution, cross-file linking,
override edge generation, retrieval expansion, and chunk-content enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.chunk_assembler import assemble_chunks
from core.embedder import embed_branch_chunks, make_fake_embedder
from core.extractor import index_repo
from core.heuristic_resolver import resolve_branch_imports
from core.retrieval import hybrid_query, semantic_query
from core.semantic_resolver import resolve_repo


async def _seed(pool, repo_id: int, branch_id: int, root: Path):
    await index_repo(pool, repo_id, branch_id, root)
    await resolve_repo(pool, repo_id, branch_id)
    await resolve_branch_imports(pool, repo_id, branch_id)
    await assemble_chunks(pool, repo_id, branch_id)
    embed_fn, model = make_fake_embedder()
    await embed_branch_chunks(pool, branch_id, embed_fn, model)
    return embed_fn


# ────────────────────────────────────────────────────────────────────
# Solidity
# ────────────────────────────────────────────────────────────────────


async def test_solidity_intra_file_inheritance(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT child.qualified_name AS child_name,
                   base.qualified_name  AS base_name,
                   ie.confidence,
                   ie.base_def_id IS NOT NULL AS resolved
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base ON base.id = ie.base_def_id
            WHERE ie.branch_id = $1 AND child.qualified_name = 'Policy.SingleExecutorPolicy'
            """,
            branch_id,
        )
        assert len(rows) == 1, rows
        assert rows[0]["base_name"] == "Policy.Policy"
        assert rows[0]["resolved"] is True
        assert rows[0]["confidence"] == "certain"


async def test_solidity_cross_file_inheritance(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            JOIN definitions base  ON base.id  = ie.base_def_id
            WHERE ie.branch_id = $1 AND child.qualified_name = 'PolicyExtended.LoggingPolicy'
            """,
            branch_id,
        )
        assert row is not None
        assert row["base_name"] == "Policy.Policy"


async def test_solidity_overrides(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            WHERE oe.branch_id = $1
              AND child.qualified_name = 'Policy.SingleExecutorPolicy.isPolicyActive'
            """,
            branch_id,
        )
        assert row is not None
        assert row["base_name"] == "Policy.Policy.isPolicyActive"

        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            WHERE oe.branch_id = $1
              AND child.qualified_name = 'PolicyExtended.LoggingPolicy.isPolicyActive'
            """,
            branch_id,
        )
        assert row is not None
        assert row["base_name"] == "Policy.Policy.isPolicyActive"


# ────────────────────────────────────────────────────────────────────
# Python
# ────────────────────────────────────────────────────────────────────


async def test_python_intra_file_inheritance_and_override(
    clean_repo, python_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_name, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            JOIN definitions base  ON base.id  = ie.base_def_id
            WHERE ie.branch_id = $1 AND child.qualified_name = 'policies.StrictPolicy'
            """,
            branch_id,
        )
        assert row is not None
        assert row["base_name"] == "policies.BasePolicy"
        assert row["confidence"] == "certain"

        ovr = await conn.fetch(
            """
            SELECT base.qualified_name AS base_name
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            WHERE oe.branch_id = $1
              AND child.qualified_name LIKE 'policies.StrictPolicy.%'
            ORDER BY base.qualified_name
            """,
            branch_id,
        )
        names = [r["base_name"] for r in ovr]
        assert "policies.BasePolicy.is_active" in names
        assert "policies.BasePolicy.check" in names


# ────────────────────────────────────────────────────────────────────
# Go
# ────────────────────────────────────────────────────────────────────


async def test_go_intra_file_interface_embedding(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, go_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ie.base_name, base.qualified_name AS base_qname
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base  ON base.id  = ie.base_def_id
            WHERE ie.branch_id = $1
              AND child.qualified_name LIKE 'composed.ReadWriter%'
            ORDER BY ie.base_name
            """,
            branch_id,
        )
        base_names = [r["base_name"] for r in rows]
        assert base_names == ["Reader", "Writer"], base_names
        resolved_qnames = {r["base_qname"] for r in rows if r["base_qname"]}
        assert "basic.Reader" in resolved_qnames
        assert "basic.Writer" in resolved_qnames


async def test_go_cross_file_interface_embedding(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, go_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ie.base_name, base.qualified_name AS base_qname
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base  ON base.id  = ie.base_def_id
            WHERE ie.branch_id = $1
              AND child.qualified_name LIKE 'composed.FullIO%'
            """,
            branch_id,
        )
        base_to_qname = {r["base_name"]: r["base_qname"] for r in rows}
        assert base_to_qname.get("ReadWriter") == "composed.ReadWriter"
        assert base_to_qname.get("Closer") == "basic.Closer"


async def test_go_struct_embedding(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, go_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ie.base_name, base.qualified_name AS base_qname, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base  ON base.id  = ie.base_def_id
            WHERE ie.branch_id = $1
              AND child.qualified_name = 'main.Dog'
            ORDER BY ie.ord
            """,
            branch_id,
        )
        base_to_qname = {r["base_name"]: r["base_qname"] for r in rows}
        assert "Animal" in base_to_qname
        assert "Closer" in base_to_qname
        assert "Breed" not in base_to_qname
        assert base_to_qname["Animal"] == "main.Animal"
        assert base_to_qname["Closer"] == "basic.Closer"


async def test_go_interface_method_overrides(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, go_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base_qname
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            WHERE oe.branch_id = $1
              AND child.qualified_name = 'composed.FullIO.Read'
            """,
            branch_id,
        )
        assert row is not None, "expected FullIO.Read → Reader.Read override edge"
        assert row["base_qname"] == "basic.Reader.Read"


# ────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript
# ────────────────────────────────────────────────────────────────────


async def test_node_class_extends_resolves_cross_file(
    clean_repo, node_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, node_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ie.base_name, base.qualified_name AS base, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base ON base.id = ie.base_def_id
            WHERE ie.branch_id = $1 AND child.qualified_name = 'pets.Dog'
            """,
            branch_id,
        )
        by_name = {r["base_name"]: r for r in rows}
        assert "Animal" in by_name
        assert by_name["Animal"]["base"] == "lib.Animal"
        assert by_name["Animal"]["confidence"] == "certain"


async def test_node_class_implements_multi_resolves_cross_file(
    clean_repo, node_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, node_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ie.base_name, base.qualified_name AS base
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            LEFT JOIN definitions base ON base.id = ie.base_def_id
            WHERE ie.branch_id = $1 AND child.qualified_name = 'pets.Cat'
            """,
            branch_id,
        )
        by_name = {r["base_name"]: r["base"] for r in rows}
        assert by_name == {
            "Animal":  "lib.Animal",
            "Greeter": "lib.Greeter",
            "Closer":  "helpers.Closer",
        }


async def test_node_interface_extends_resolves_intra_file(
    clean_repo, node_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, node_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT base.qualified_name AS base, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id = ie.child_def_id
            JOIN definitions base  ON base.id  = ie.base_def_id
            WHERE ie.branch_id = $1 AND child.qualified_name = 'lib.Bilingual'
            """,
            branch_id,
        )
        assert row is not None
        assert row["base"] == "lib.Greeter"
        assert row["confidence"] == "certain"


async def test_node_overrides_via_implements(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, node_fixture_root)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT child.qualified_name AS child, base.qualified_name AS base
            FROM overrides_edges oe
            JOIN definitions child ON child.id = oe.child_def_id
            JOIN definitions base  ON base.id  = oe.base_def_id
            WHERE oe.branch_id = $1
            ORDER BY child.qualified_name, base.qualified_name
            """,
            branch_id,
        )
        pairs = {(r["child"], r["base"]) for r in rows}
        assert ("pets.Dog.greet", "lib.Greeter.greet") in pairs
        assert ("pets.Cat.greet", "lib.Greeter.greet") in pairs
        assert ("pets.Cat.close", "helpers.Closer.close") in pairs


async def test_chunk_enrichment_for_overriding_method(
    clean_repo, solidity_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, solidity_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.content, c.metadata
            FROM chunks c
            JOIN definitions d ON d.id = c.anchor_def_id
            WHERE c.branch_id = $1
              AND d.qualified_name = 'Policy.SingleExecutorPolicy.isPolicyActive'
              AND c.granularity = 'function'
            """,
            branch_id,
        )
        assert row is not None, "chunk for SingleExecutorPolicy.isPolicyActive missing"
        meta = json.loads(row["metadata"])
        assert meta["overrides"] == "Policy.Policy.isPolicyActive"
        assert "Policy.Policy" in meta["inheritance_chain"]
        assert "Policy.IPolicy" in meta["inheritance_chain"]
        content = row["content"]
        assert "# overrides: Policy.Policy.isPolicyActive" in content
        assert "# inheritance chain:" in content
        assert "onlyManager" in content or "_requireSender" in content


async def test_hybrid_pulls_in_overrides(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    embed_fn = await _seed(pool, repo_id, branch_id, solidity_fixture_root)
    chunks = await hybrid_query(
        pool, branch_id, "is policy active manager check", embed_fn,
        top_k=10, candidate_pool=10,
    )
    qualified = {c.qualified_name for c in chunks}
    assert any("Policy.Policy.isPolicyActive" in q for q in qualified if q), qualified
    overrides = [
        q for q in qualified
        if q and (q.endswith("SingleExecutorPolicy.isPolicyActive")
                  or q.endswith("LoggingPolicy.isPolicyActive"))
    ]
    assert overrides, f"no override surfaced; got: {sorted(qualified)}"
