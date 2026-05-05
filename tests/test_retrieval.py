"""Phase 4b/4c acceptance tests using the deterministic fake embedder.

Covers cross-repo retrieval, RRF fusion, identifier-aware FTS, and MMR
diversity reranking.
"""

from __future__ import annotations

from pathlib import Path

from core.chunk_assembler import _expand_idents, _fts_text, assemble_chunks
from core.embedder import embed_branch_chunks, make_fake_embedder
from core.extractor import index_repo
from core.heuristic_resolver import resolve_branch_imports
from core.retrieval import (
    RetrievedChunk,
    _build_tsquery,
    _fuse_scores,
    _mmr_rerank,
    assemble_context,
    hybrid_query,
    lexical_query,
    semantic_query,
    structural_query,
)
from core.semantic_resolver import resolve_repo


async def _seed_python(pool, repo_id: int, branch_id: int, root: Path):
    await index_repo(pool, repo_id, branch_id, root)
    await resolve_repo(pool, repo_id, branch_id)
    await resolve_branch_imports(pool, repo_id, branch_id)
    await assemble_chunks(pool, repo_id, branch_id)
    embed_fn, model_name = make_fake_embedder()
    await embed_branch_chunks(pool, branch_id, embed_fn, model_name)
    return embed_fn


# ────────────────────────────────────────────────────────────────────
# Embedder smoke
# ────────────────────────────────────────────────────────────────────


async def test_fake_embedder_inserts_rows(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed_python(pool, repo_id, branch_id, python_fixture_root)
    async with pool.acquire() as conn:
        n_chunks = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE branch_id=$1", branch_id,
        )
        # chunk_embeddings is keyed by content_hash; one row per distinct
        # chunk content (typically same as chunk count for an unseeded DB).
        n_distinct_hashes = await conn.fetchval(
            "SELECT COUNT(DISTINCT content_hash) FROM chunks WHERE branch_id=$1",
            branch_id,
        )
        n_embeds = await conn.fetchval(
            """
            SELECT COUNT(*) FROM chunk_embeddings ce
            WHERE ce.content_hash IN (
                SELECT DISTINCT content_hash FROM chunks WHERE branch_id=$1
            )
            """,
            branch_id,
        )
        assert n_chunks > 0
        assert n_embeds == n_distinct_hashes


# ────────────────────────────────────────────────────────────────────
# Retrieval modes
# ────────────────────────────────────────────────────────────────────


async def test_structural_query_returns_callees(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed_python(pool, repo_id, branch_id, python_fixture_root)

    chunks = await structural_query(pool, branch_id, "run", depth=2, granularity="function")
    qns = {c.qualified_name for c in chunks}
    assert "main.run" in qns
    assert "main.Calculator.add" in qns
    by_q = {c.qualified_name: c.score for c in chunks}
    assert by_q["main.run"] >= by_q["main.Calculator.add"]


async def test_semantic_query_returns_top_k(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    embed_fn = await _seed_python(pool, repo_id, branch_id, python_fixture_root)
    chunks = await semantic_query(pool, branch_id, "Calculator add helper", embed_fn, top_k=5)
    assert chunks
    assert all(c.score >= -1.0 and c.score <= 1.0 for c in chunks)
    async with pool.acquire() as conn:
        seed_content = await conn.fetchval(
            """
            SELECT c.content FROM chunks c
            JOIN definitions d ON d.id=c.anchor_def_id
            WHERE c.branch_id=$1 AND d.qualified_name='main.Calculator.add'
              AND c.granularity='function' LIMIT 1
            """,
            branch_id,
        )
    same = await semantic_query(pool, branch_id, seed_content, embed_fn, top_k=1)
    assert same and same[0].qualified_name == "main.Calculator.add"


def _mk_test_chunk(
    *, chunk_id: int, anchor_def_id: int, qualified_name: str, file_path: str,
    file_version_id: int, branch_id: int, token_count: int, content: str,
    score: float, repo_id: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        anchor_def_id=anchor_def_id,
        qualified_name=qualified_name,
        granularity="function",
        file_path=file_path,
        file_version_id=file_version_id,
        branch_id=branch_id,
        token_count=token_count,
        content=content,
        score=score,
        repo_id=repo_id,
    )


def test_assemble_context_dedupes_and_budgets():
    a = _mk_test_chunk(
        chunk_id=1, anchor_def_id=10, qualified_name="x", file_path="a.py",
        file_version_id=1, branch_id=1, token_count=300, content="A " * 100, score=0.9,
    )
    a2 = _mk_test_chunk(
        chunk_id=2, anchor_def_id=10, qualified_name="x", file_path="a.py",
        file_version_id=1, branch_id=1, token_count=300, content="DUP " * 100, score=0.8,
    )
    b = _mk_test_chunk(
        chunk_id=3, anchor_def_id=11, qualified_name="y", file_path="b.py",
        file_version_id=2, branch_id=1, token_count=300, content="B " * 100, score=0.7,
    )
    c = _mk_test_chunk(
        chunk_id=4, anchor_def_id=12, qualified_name="z", file_path="c.py",
        file_version_id=3, branch_id=1, token_count=900, content="C " * 100, score=0.6,
    )
    out = assemble_context([a, a2, b, c], token_budget=700)
    assert "A" in out and "B" in out
    assert "DUP" not in out
    assert "C" not in out


# ────────────────────────────────────────────────────────────────────
# Cross-repo retrieval
# ────────────────────────────────────────────────────────────────────


async def test_cross_repo_semantic_query_no_leak(two_repos, python_fixture_root: Path):
    """Two repos with the same fixture indexed; querying one repo must not
    return chunks from the other."""
    pool, (a, ba), (b, bb) = two_repos
    embed_fn = await _seed_python(pool, a, ba, python_fixture_root)
    await _seed_python(pool, b, bb, python_fixture_root)
    chunks = await semantic_query(pool, ba, "Calculator add", embed_fn, top_k=20)
    assert chunks
    assert all(c.repo_id == a for c in chunks)


async def test_cross_repo_returns_both_repos(two_repos, python_fixture_root: Path):
    pool, (a, ba), (b, bb) = two_repos
    embed_fn = await _seed_python(pool, a, ba, python_fixture_root)
    await _seed_python(pool, b, bb, python_fixture_root)
    chunks = await semantic_query(pool, [ba, bb], "Calculator add", embed_fn, top_k=50)
    repos = {c.repo_id for c in chunks}
    assert a in repos and b in repos


async def test_structural_query_cross_repo(two_repos, python_fixture_root: Path):
    pool, (a, ba), (b, bb) = two_repos
    await _seed_python(pool, a, ba, python_fixture_root)
    await _seed_python(pool, b, bb, python_fixture_root)
    chunks_both = await structural_query(pool, [ba, bb], "run", depth=1)
    repos_both = {c.repo_id for c in chunks_both}
    assert a in repos_both and b in repos_both
    chunks_only_a = await structural_query(pool, ba, "run", depth=1)
    assert all(c.repo_id == a for c in chunks_only_a)


# ────────────────────────────────────────────────────────────────────
# MMR diversity
# ────────────────────────────────────────────────────────────────────


async def test_mmr_diversifies_across_repos(two_repos, python_fixture_root: Path):
    pool, (a, ba), (b, bb) = two_repos
    embed_fn = await _seed_python(pool, a, ba, python_fixture_root)
    await _seed_python(pool, b, bb, python_fixture_root)
    chunks = await hybrid_query(
        pool, [ba, bb], "Calculator add helper", embed_fn,
        top_k=4, candidate_pool=20,
    )
    assert chunks
    repos = {c.repo_id for c in chunks}
    assert a in repos and b in repos, (
        f"MMR didn't span both repos; got: {[(c.repo_id, c.qualified_name) for c in chunks]}"
    )


def test_mmr_noop_single_scope():
    """When all candidates share one repo and one file, MMR should preserve
    the original ordering (no diversity penalty has anything to do)."""
    chunks = [
        _mk_test_chunk(
            chunk_id=i, anchor_def_id=i, qualified_name=f"x{i}", file_path="a.py",
            file_version_id=1, branch_id=1, token_count=100, content="",
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]
    scores = {c.chunk_id: c.score for c in chunks}
    out = _mmr_rerank(chunks, scores, top_k=3)
    assert [c.chunk_id for c in out] == [0, 1, 2]


def test_mmr_diversifies_across_files():
    """Single repo, multiple files. Without MMR, top-3 picks are {0,1,2} all
    in file 1. With file_lambda > 0 the second pick should jump to a
    different file."""
    chunks = [
        _mk_test_chunk(
            chunk_id=0, anchor_def_id=0, qualified_name="a0", file_path="a.py",
            file_version_id=1, branch_id=1, token_count=100, content="", score=1.0,
        ),
        _mk_test_chunk(
            chunk_id=1, anchor_def_id=1, qualified_name="a1", file_path="a.py",
            file_version_id=1, branch_id=1, token_count=100, content="", score=0.9,
        ),
        _mk_test_chunk(
            chunk_id=2, anchor_def_id=2, qualified_name="b0", file_path="b.py",
            file_version_id=2, branch_id=1, token_count=100, content="", score=0.85,
        ),
        _mk_test_chunk(
            chunk_id=3, anchor_def_id=3, qualified_name="c0", file_path="c.py",
            file_version_id=3, branch_id=1, token_count=100, content="", score=0.7,
        ),
    ]
    scores = {c.chunk_id: c.score for c in chunks}
    out = _mmr_rerank(chunks, scores, top_k=3, file_lambda=0.5, repo_lambda=0.0)
    fvs_out = [c.file_version_id for c in out]
    assert fvs_out[0] == 1
    assert fvs_out[1] == 2


# ────────────────────────────────────────────────────────────────────
# Identifier-aware FTS
# ────────────────────────────────────────────────────────────────────


def test_expand_idents_camel_split():
    out = _expand_idents("def getUserById(): return findUserId()")
    assert "get User By Id" in out
    assert "find User Id" in out


def test_expand_idents_snake_split():
    out = _expand_idents("process_payment_request")
    assert " process " in f" {out} "
    assert " payment " in f" {out} "
    assert " request " in f" {out} "


def test_fts_text_includes_qn_and_body_fanout():
    out = _fts_text("pkg.UserService.getById", "def getById(self, id): return _lookup(id)")
    assert "pkg.UserService.getById" in out
    assert "User Service" in out
    assert "get By Id" in out


def test_tsquery_query_side_fanout():
    tsq = _build_tsquery("getUserById")
    assert tsq is not None
    assert "getuserbyid:*" in tsq
    for sub in ("get:*", "user:*", "by:*", "id:*"):
        assert sub in tsq, f"missing {sub} in {tsq}"


def test_tsquery_snake_fanout():
    tsq = _build_tsquery("process_payment")
    assert tsq is not None
    assert "process_payment:*" in tsq
    assert "process:*" in tsq
    assert "payment:*" in tsq


async def test_lexical_query_finds_camel_body_match(
    clean_repo, python_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _seed_python(pool, repo_id, branch_id, python_fixture_root)
    chunks = await lexical_query(pool, branch_id, "helper", top_k=20)
    qns = {c.qualified_name for c in chunks}
    assert "utils.helper" in qns


# ────────────────────────────────────────────────────────────────────
# RRF fusion
# ────────────────────────────────────────────────────────────────────


def _mk_chunk(cid: int, score: float = 0.0) -> RetrievedChunk:
    return _mk_test_chunk(
        chunk_id=cid, anchor_def_id=cid, qualified_name=f"x{cid}",
        file_path=f"f{cid}.py", file_version_id=cid, branch_id=1,
        token_count=10, content="", score=score,
    )


def test_rrf_fusion_unaffected_by_score_skew():
    sem = [_mk_chunk(1, score=999.0)]
    sem += [_mk_chunk(i, score=0.001) for i in range(2, 12)]
    lex = [_mk_chunk(100 + i, score=10.0 - i) for i in range(10)]
    by_id, scores = _fuse_scores(sem, lex)
    assert 1 in by_id and 100 in by_id
    assert abs(scores[1] - scores[100]) < 1e-9


def test_rrf_normalized_for_graph_expansion():
    sem = [_mk_chunk(i) for i in range(1, 6)]
    lex = [_mk_chunk(100 + i) for i in range(5)]
    _, scores = _fuse_scores(sem, lex)
    assert scores
    assert max(scores.values()) == 1.0
    assert min(scores.values()) > 0.0
