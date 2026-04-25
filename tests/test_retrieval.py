"""Phase 4b/4c acceptance tests using the deterministic fake embedder."""

from __future__ import annotations

from pathlib import Path

from core.chunk_assembler import assemble_chunks
from core.embedder import embed_repo_chunks, make_fake_embedder
from core.extractor import index_repo
from core.heuristic_resolver import resolve_repo_imports
from core.retrieval import (
    assemble_context,
    hybrid_query,
    semantic_query,
    structural_query,
)
from core.semantic_resolver import resolve_repo


async def _seed_python(pool, repo_id: int, root: Path):
    await index_repo(pool, repo_id, root)
    await resolve_repo(pool, repo_id)
    await resolve_repo_imports(pool, repo_id)
    await assemble_chunks(pool, repo_id)
    embed_fn, model_name = make_fake_embedder()
    await embed_repo_chunks(pool, repo_id, embed_fn, model_name)
    return embed_fn


# ────────────────────────────────────────────────────────────────────
# Embedder smoke
# ────────────────────────────────────────────────────────────────────


async def test_fake_embedder_inserts_rows(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed_python(pool, repo_id, python_fixture_root)
    async with pool.acquire() as conn:
        n_chunks = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id=c.file_id WHERE f.repo_id=$1",
            repo_id,
        )
        n_embeds = await conn.fetchval(
            "SELECT COUNT(*) FROM chunk_embeddings ce "
            "JOIN chunks c ON c.id=ce.chunk_id "
            "JOIN files f ON f.id=c.file_id "
            "WHERE f.repo_id=$1",
            repo_id,
        )
        assert n_chunks > 0
        assert n_embeds == n_chunks


# ────────────────────────────────────────────────────────────────────
# Retrieval modes
# ────────────────────────────────────────────────────────────────────


async def test_structural_query_returns_callees(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed_python(pool, repo_id, python_fixture_root)

    chunks = await structural_query(pool, repo_id, "run", depth=2, granularity="function")
    qns = {c.qualified_name for c in chunks}
    # `run` calls Calculator(...) (kind=class — no function chunk) and calc.add (Calculator.add).
    assert "main.run" in qns
    assert "main.Calculator.add" in qns
    # Score decays with hop depth.
    by_q = {c.qualified_name: c.score for c in chunks}
    assert by_q["main.run"] >= by_q["main.Calculator.add"]


async def test_semantic_query_returns_top_k(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    embed_fn = await _seed_python(pool, repo_id, python_fixture_root)
    chunks = await semantic_query(pool, repo_id, "Calculator add helper", embed_fn, top_k=5)
    assert chunks
    assert all(c.score >= -1.0 and c.score <= 1.0 for c in chunks)
    # With the deterministic stub, querying for the EXACT chunk content returns it first.
    async with pool.acquire() as conn:
        seed_content = await conn.fetchval(
            "SELECT c.content FROM chunks c JOIN files f ON f.id=c.file_id "
            "JOIN definitions d ON d.id=c.anchor_def_id "
            "WHERE f.repo_id=$1 AND d.qualified_name='main.Calculator.add' "
            "  AND c.granularity='function' LIMIT 1",
            repo_id,
        )
    same = await semantic_query(pool, repo_id, seed_content, embed_fn, top_k=1)
    assert same and same[0].qualified_name == "main.Calculator.add"


async def test_hybrid_query_expands_via_graph(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    embed_fn = await _seed_python(pool, repo_id, python_fixture_root)
    # Seed the query with main.run's content; hybrid should also surface its callees.
    async with pool.acquire() as conn:
        seed = await conn.fetchval(
            "SELECT c.content FROM chunks c "
            "JOIN definitions d ON d.id=c.anchor_def_id "
            "WHERE d.qualified_name='main.run' AND c.granularity='function'",
        )
    chunks = await hybrid_query(pool, repo_id, seed, embed_fn, top_k=5)
    qns = {c.qualified_name for c in chunks}
    assert "main.run" in qns
    # Calculator.add is a 1-hop neighbour and should be lifted into the rank.
    assert "main.Calculator.add" in qns


def test_assemble_context_dedupes_and_budgets():
    from core.retrieval import RetrievedChunk

    a = RetrievedChunk(chunk_id=1, anchor_def_id=10, qualified_name="x", granularity="function",
                       file_path="a.py", file_id=1, token_count=300, content="A " * 100, score=0.9)
    a2 = RetrievedChunk(chunk_id=2, anchor_def_id=10, qualified_name="x", granularity="function",
                        file_path="a.py", file_id=1, token_count=300, content="DUP " * 100, score=0.8)
    b = RetrievedChunk(chunk_id=3, anchor_def_id=11, qualified_name="y", granularity="function",
                       file_path="b.py", file_id=2, token_count=300, content="B " * 100, score=0.7)
    c = RetrievedChunk(chunk_id=4, anchor_def_id=12, qualified_name="z", granularity="function",
                       file_path="c.py", file_id=3, token_count=900, content="C " * 100, score=0.6)
    out = assemble_context([a, a2, b, c], token_budget=700)
    assert "A" in out and "B" in out          # both fit and are unique anchors
    assert "DUP" not in out                    # duplicate anchor dropped
    assert "C" not in out                      # over budget
