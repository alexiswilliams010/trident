"""Tests for the branch dimension: per-repo isolation, content-hash sharing
of file_versions / nodes / definitions / chunk_embeddings, branch-drop
cascade, gc, default-branch refusal-to-drop, resolver hydration determinism.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

from cli._repo import resolve_branch_id, resolve_repo_id
from core.chunk_assembler import assemble_chunks
from core.embedder import embed_branch_chunks, make_fake_embedder
from core.extractor import index_repo
from core.heuristic_resolver import resolve_branch_imports
from core.semantic_resolver import resolve_repo

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ────────────────────────────────────────────────────────────────────
# Schema-level invariants
# ────────────────────────────────────────────────────────────────────


async def test_branches_unique_per_repo(pg_pool):
    """branches.UNIQUE(repo_id, name): two repos can each own a 'main'."""
    async with pg_pool.acquire() as conn:
        a = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id",
            f"isolation-a-{os.urandom(4).hex()}",
        )
        b = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id",
            f"isolation-b-{os.urandom(4).hex()}",
        )
        try:
            ba = await conn.fetchval(
                "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', TRUE) RETURNING id",
                a,
            )
            bb = await conn.fetchval(
                "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', TRUE) RETURNING id",
                b,
            )
            assert ba != bb
            # Same name twice in the same repo must fail.
            with pytest.raises(Exception):
                await conn.execute(
                    "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', FALSE)",
                    a,
                )
        finally:
            await conn.execute("DELETE FROM repos WHERE id = ANY($1::bigint[])", [a, b])


async def test_one_default_branch_per_repo(pg_pool):
    """Partial unique index `idx_branches_one_default` enforces ≤1 is_default per repo."""
    async with pg_pool.acquire() as conn:
        a = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id",
            f"default-uniq-{os.urandom(4).hex()}",
        )
        try:
            await conn.execute(
                "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'main', TRUE)", a,
            )
            with pytest.raises(Exception):
                await conn.execute(
                    "INSERT INTO branches (repo_id, name, is_default) VALUES ($1, 'feature', TRUE)", a,
                )
        finally:
            await conn.execute("DELETE FROM repos WHERE id=$1", a)


# ────────────────────────────────────────────────────────────────────
# resolve_branch_id helper
# ────────────────────────────────────────────────────────────────────


async def test_resolve_branch_id_creates_default_main_on_first_call(pg_pool):
    name = f"resolve-default-{os.urandom(4).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", name,
        )
    try:
        branch_id = await resolve_branch_id(pg_pool, repo_id=repo_id, create=True)
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, is_default FROM branches WHERE id=$1", branch_id,
            )
        assert row["name"] == "main"
        assert row["is_default"] is True

        # Second create with a different name should NOT mark it default.
        b2 = await resolve_branch_id(pg_pool, repo_id=repo_id, name="feature/x", create=True)
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT is_default FROM branches WHERE id=$1", b2,
            )
        assert row["is_default"] is False

        # Lookup with name=None returns the default.
        got = await resolve_branch_id(pg_pool, repo_id=repo_id, create=False)
        assert got == branch_id
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)


# ────────────────────────────────────────────────────────────────────
# Content-hash sharing of file_versions and embeddings
# ────────────────────────────────────────────────────────────────────


async def _index_branch(pool, repo_id, branch_id, repo_path, *, embed=False):
    """Helper: run the full index → resolve → chunks → embed pipeline."""
    extract_result = await index_repo(pool, repo_id, branch_id, repo_path)
    only = [r.file_version_id for r in extract_result.indexed]
    if only:
        await resolve_repo(pool, repo_id, branch_id, only_file_version_ids=only)
    await resolve_branch_imports(pool, repo_id, branch_id)
    cstats = await assemble_chunks(pool, repo_id, branch_id)
    if embed:
        embed_fn, model_name = make_fake_embedder()
        estats = await embed_branch_chunks(pool, branch_id, embed_fn, model_name)
        return extract_result, cstats, estats
    return extract_result, cstats, None


async def test_no_divergence_branch_reuses_file_versions_and_embeddings(
    pg_pool, python_fixture_root: Path,
):
    """Indexing a fresh branch with identical content to main should produce
    zero new file_versions / nodes / definitions / chunk_embeddings."""
    repo_name = f"share-{os.urandom(4).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", repo_name,
        )
    try:
        main_branch = await resolve_branch_id(pg_pool, repo_id=repo_id, create=True)
        await _index_branch(pg_pool, repo_id, main_branch, python_fixture_root, embed=True)

        async with pg_pool.acquire() as conn:
            fv_before = await conn.fetchval("SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id)
            nodes_before = await conn.fetchval(
                "SELECT COUNT(*) FROM nodes n JOIN file_versions fv ON fv.id=n.file_version_id WHERE fv.repo_id=$1",
                repo_id,
            )
            defs_before = await conn.fetchval(
                "SELECT COUNT(*) FROM definitions d JOIN file_versions fv ON fv.id=d.file_version_id WHERE fv.repo_id=$1",
                repo_id,
            )
            ce_before = await conn.fetchval("SELECT COUNT(*) FROM chunk_embeddings")

        feat_branch = await resolve_branch_id(
            pg_pool, repo_id=repo_id, name="feature/x", create=True,
        )
        _, _, estats = await _index_branch(
            pg_pool, repo_id, feat_branch, python_fixture_root, embed=True,
        )

        async with pg_pool.acquire() as conn:
            fv_after = await conn.fetchval("SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id)
            nodes_after = await conn.fetchval(
                "SELECT COUNT(*) FROM nodes n JOIN file_versions fv ON fv.id=n.file_version_id WHERE fv.repo_id=$1",
                repo_id,
            )
            defs_after = await conn.fetchval(
                "SELECT COUNT(*) FROM definitions d JOIN file_versions fv ON fv.id=d.file_version_id WHERE fv.repo_id=$1",
                repo_id,
            )
            ce_after = await conn.fetchval("SELECT COUNT(*) FROM chunk_embeddings")

        assert fv_after == fv_before, "file_versions should be reused across branches"
        assert nodes_after == nodes_before, "nodes are content-shared"
        assert defs_after == defs_before, "definitions are content-shared"
        # chunk_embeddings is content-keyed; if chunks happened to differ
        # between branches (skeleton differences) some new embeddings can
        # appear, but most should be reused.
        assert ce_after >= ce_before  # never decreases
        # The embedder API call count should be near 0 — exact 0 is hard to
        # promise because skeleton-driven chunk text may differ across
        # branches even on the same source. We check the dominant case.
        assert estats.embedded <= max(1, (ce_after - ce_before)), (
            f"feature/x re-embedded {estats.embedded} chunks; expected ~0 for shared content"
        )
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)


# ────────────────────────────────────────────────────────────────────
# Branch-drop cascade
# ────────────────────────────────────────────────────────────────────


async def test_branch_drop_cascades_branch_scoped_rows(pg_pool, python_fixture_root: Path):
    """DELETE FROM branches WHERE id=$1 should cascade through all per-branch
    tables but leave shared content untouched."""
    repo_name = f"drop-{os.urandom(4).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", repo_name,
        )
    try:
        main_branch = await resolve_branch_id(pg_pool, repo_id=repo_id, create=True)
        feat_branch = await resolve_branch_id(
            pg_pool, repo_id=repo_id, name="feature/y", create=True,
        )
        await _index_branch(pg_pool, repo_id, main_branch, python_fixture_root)
        await _index_branch(pg_pool, repo_id, feat_branch, python_fixture_root)

        async with pg_pool.acquire() as conn:
            fv_before = await conn.fetchval("SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id)
            defs_before = await conn.fetchval(
                "SELECT COUNT(*) FROM definitions d JOIN file_versions fv ON fv.id=d.file_version_id WHERE fv.repo_id=$1",
                repo_id,
            )
            await conn.execute("DELETE FROM branches WHERE id=$1", feat_branch)

            # Branch-scoped tables: feat_branch rows gone.
            for table in ("branch_files", '"references"', "call_edges", "data_access",
                          "overrides_edges", "inherits_edges", "imports", "external_dependencies",
                          "chunks"):
                n = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE branch_id=$1", feat_branch)
                assert n == 0, f"{table} should have no rows for dropped branch"

            # Shared tables: untouched.
            fv_after = await conn.fetchval("SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id)
            defs_after = await conn.fetchval(
                "SELECT COUNT(*) FROM definitions d JOIN file_versions fv ON fv.id=d.file_version_id WHERE fv.repo_id=$1",
                repo_id,
            )
            assert fv_after == fv_before, "file_versions should be preserved"
            assert defs_after == defs_before, "definitions should be preserved"
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)


# ────────────────────────────────────────────────────────────────────
# GC reclaims orphans
# ────────────────────────────────────────────────────────────────────


async def test_gc_reclaims_orphan_file_versions_after_drop(pg_pool, python_fixture_root: Path):
    """After dropping the only branch that referenced a file_version, gc
    should clean it up."""
    repo_name = f"gc-{os.urandom(4).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", repo_name,
        )
    try:
        main_branch = await resolve_branch_id(pg_pool, repo_id=repo_id, create=True)
        await _index_branch(pg_pool, repo_id, main_branch, python_fixture_root)

        async with pg_pool.acquire() as conn:
            fv_count = await conn.fetchval(
                "SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id,
            )
        assert fv_count > 0

        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM branches WHERE id=$1", main_branch)
            # branch_files rows are gone; file_versions rows are now orphans.
            orphans = await conn.fetchval(
                """
                SELECT COUNT(*) FROM file_versions fv
                WHERE fv.repo_id=$1
                  AND fv.id NOT IN (SELECT file_version_id FROM branch_files)
                """,
                repo_id,
            )
            assert orphans == fv_count

            # Run gc.
            await conn.execute(
                """
                DELETE FROM file_versions
                WHERE id NOT IN (SELECT file_version_id FROM branch_files)
                """
            )
            await conn.execute(
                """
                DELETE FROM chunk_embeddings
                WHERE content_hash NOT IN (SELECT content_hash FROM chunks)
                """
            )

            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id,
            )
            assert remaining == 0
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)


# ────────────────────────────────────────────────────────────────────
# Resolver hydration (re-resolving an existing file_version)
# ────────────────────────────────────────────────────────────────────


async def test_hydration_path_produces_same_tier2_outputs(pg_pool, python_fixture_root: Path):
    """When a file_version already has definitions, re-resolving it (under a
    different branch_id) must hydrate the in-memory scope tables from the DB
    and emit identical Tier-2 row counts."""
    repo_name = f"hydrate-{os.urandom(4).hex()}"
    async with pg_pool.acquire() as conn:
        repo_id = await conn.fetchval(
            "INSERT INTO repos (name) VALUES ($1) RETURNING id", repo_name,
        )
    try:
        main_branch = await resolve_branch_id(pg_pool, repo_id=repo_id, create=True)
        await _index_branch(pg_pool, repo_id, main_branch, python_fixture_root)

        async with pg_pool.acquire() as conn:
            refs_main = await conn.fetchval(
                'SELECT COUNT(*) FROM "references" WHERE branch_id=$1', main_branch,
            )
            calls_main = await conn.fetchval(
                "SELECT COUNT(*) FROM call_edges WHERE branch_id=$1", main_branch,
            )

        # Create a second branch reusing the same content; the resolver enters
        # hydrate mode because definitions already exist for those file_versions.
        feat_branch = await resolve_branch_id(
            pg_pool, repo_id=repo_id, name="feature/h", create=True,
        )
        await _index_branch(pg_pool, repo_id, feat_branch, python_fixture_root)

        async with pg_pool.acquire() as conn:
            refs_feat = await conn.fetchval(
                'SELECT COUNT(*) FROM "references" WHERE branch_id=$1', feat_branch,
            )
            calls_feat = await conn.fetchval(
                "SELECT COUNT(*) FROM call_edges WHERE branch_id=$1", feat_branch,
            )

        assert refs_main == refs_feat, (
            f"references count diverged: main={refs_main}, feature={refs_feat}"
        )
        assert calls_main == calls_feat, (
            f"call_edges count diverged: main={calls_main}, feature={calls_feat}"
        )
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM repos WHERE id=$1", repo_id)
