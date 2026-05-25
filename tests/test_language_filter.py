"""`languages=` filter pushdown on graph + retrieval queries.

Indexes a single-language (Python) fixture, then asserts that filtering to a
*different* language returns nothing, filtering to the fixture's own language
(or a set containing it) is unaffected, and `languages=None` is identical to the
pre-filter behavior. This is the trident half of Neptune's multi-language scoping
(it lets Neptune drop its client-side post-filter).
"""

from __future__ import annotations

from pathlib import Path

from core.chunk_assembler import assemble_chunks
from core.embedder import embed_branch_chunks, make_fake_embedder
from core.extractor import index_repo
from core.graph import (
    entrypoints,
    get_source,
    resolve_definitions,
)
from core.heuristic_resolver import resolve_branch_imports
from core.retrieval import lexical_query, structural_query
from core.semantic_resolver import resolve_repo


async def _seed(pool, repo_id: int, branch_id: int, root: Path):
    """Index + resolve, then assemble + embed chunks (needed for retrieval)."""
    await index_repo(pool, repo_id, branch_id, root)
    await resolve_repo(pool, repo_id, branch_id)
    await resolve_branch_imports(pool, repo_id, branch_id)
    await assemble_chunks(pool, repo_id, branch_id)
    embed_fn, model_name = make_fake_embedder()
    await embed_branch_chunks(pool, branch_id, embed_fn, model_name)


async def test_entrypoints_language_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)

    baseline = await entrypoints(pool, branch_id)
    assert baseline, "fixture should have entrypoints"

    # Same behavior as baseline when filtering to the fixture's own language.
    py_only = await entrypoints(pool, branch_id, languages=["python"])
    assert {d.def_id for d in py_only} == {d.def_id for d in baseline}

    # A set that includes the fixture's language keeps everything.
    union = await entrypoints(pool, branch_id, languages=["solidity", "python"])
    assert {d.def_id for d in union} == {d.def_id for d in baseline}

    # Filtering to a language that isn't present drops every row.
    assert await entrypoints(pool, branch_id, languages=["solidity"]) == []

    # languages=None is unchanged from today.
    assert {d.def_id for d in await entrypoints(pool, branch_id, languages=None)} == {
        d.def_id for d in baseline
    }


async def test_resolve_and_get_source_language_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)

    assert await resolve_definitions(pool, branch_id, "helper", languages=["python"])
    assert await resolve_definitions(pool, branch_id, "helper", languages=["solidity"]) == []

    assert await get_source(pool, branch_id, "Calculator.add", languages=["python"])
    assert await get_source(pool, branch_id, "Calculator.add", languages=["solidity"]) == []


async def test_retrieval_language_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)

    assert await structural_query(pool, branch_id, "run", depth=2, languages=["python"])
    assert await structural_query(pool, branch_id, "run", depth=2, languages=["solidity"]) == []

    assert await lexical_query(pool, branch_id, "helper", top_k=20, languages=["python"])
    assert await lexical_query(pool, branch_id, "helper", top_k=20, languages=["solidity"]) == []
