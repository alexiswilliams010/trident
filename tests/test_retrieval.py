"""Phase 4b/4c acceptance tests using the deterministic fake embedder.

Covers cross-repo retrieval, RRF fusion, identifier-aware FTS, and MMR
diversity reranking.
"""

from __future__ import annotations

from pathlib import Path

from core.chunk_assembler import _expand_idents, _fts_text, assemble_chunks
from core.embedder import embed_repo_chunks, make_fake_embedder
from core.extractor import index_repo
from core.heuristic_resolver import resolve_repo_imports
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


def test_assemble_context_dedupes_and_budgets():
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


# ────────────────────────────────────────────────────────────────────
# Cross-repo retrieval
# ────────────────────────────────────────────────────────────────────


async def test_cross_repo_semantic_query_no_leak(two_repos, python_fixture_root: Path):
    """Two repos with the same fixture indexed; querying one repo must not
    return chunks from the other."""
    pool, a, b = two_repos
    embed_fn = await _seed_python(pool, a, python_fixture_root)
    await _seed_python(pool, b, python_fixture_root)
    chunks = await semantic_query(pool, a, "Calculator add", embed_fn, top_k=20)
    assert chunks
    assert all(c.repo_id == a for c in chunks)


async def test_cross_repo_returns_both_repos(two_repos, python_fixture_root: Path):
    pool, a, b = two_repos
    embed_fn = await _seed_python(pool, a, python_fixture_root)
    await _seed_python(pool, b, python_fixture_root)
    chunks = await semantic_query(pool, [a, b], "Calculator add", embed_fn, top_k=50)
    repos = {c.repo_id for c in chunks}
    assert a in repos and b in repos


async def test_structural_query_cross_repo(two_repos, python_fixture_root: Path):
    pool, a, b = two_repos
    await _seed_python(pool, a, python_fixture_root)
    await _seed_python(pool, b, python_fixture_root)
    # Both repos have a `run` definition.
    chunks_both = await structural_query(pool, [a, b], "run", depth=1)
    repos_both = {c.repo_id for c in chunks_both}
    assert a in repos_both and b in repos_both
    chunks_only_a = await structural_query(pool, a, "run", depth=1)
    assert all(c.repo_id == a for c in chunks_only_a)


# ────────────────────────────────────────────────────────────────────
# MMR diversity
# ────────────────────────────────────────────────────────────────────


async def test_mmr_diversifies_across_repos(two_repos, python_fixture_root: Path):
    pool, a, b = two_repos
    embed_fn = await _seed_python(pool, a, python_fixture_root)
    await _seed_python(pool, b, python_fixture_root)
    chunks = await hybrid_query(
        pool, [a, b], "Calculator add helper", embed_fn,
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
        RetrievedChunk(chunk_id=i, anchor_def_id=i, qualified_name=f"x{i}",
                       granularity="function", file_path="a.py", file_id=1,
                       token_count=100, content="", score=1.0 - i * 0.1,
                       repo_id=1)
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
        RetrievedChunk(chunk_id=0, anchor_def_id=0, qualified_name="a0",
                       granularity="function", file_path="a.py", file_id=1,
                       token_count=100, content="", score=1.0, repo_id=1),
        RetrievedChunk(chunk_id=1, anchor_def_id=1, qualified_name="a1",
                       granularity="function", file_path="a.py", file_id=1,
                       token_count=100, content="", score=0.9, repo_id=1),
        RetrievedChunk(chunk_id=2, anchor_def_id=2, qualified_name="b0",
                       granularity="function", file_path="b.py", file_id=2,
                       token_count=100, content="", score=0.85, repo_id=1),
        RetrievedChunk(chunk_id=3, anchor_def_id=3, qualified_name="c0",
                       granularity="function", file_path="c.py", file_id=3,
                       token_count=100, content="", score=0.7, repo_id=1),
    ]
    scores = {c.chunk_id: c.score for c in chunks}
    out = _mmr_rerank(chunks, scores, top_k=3, file_lambda=0.5, repo_lambda=0.0)
    files_out = [c.file_id for c in out]
    # First pick is the top score in file 1; second should switch to file 2
    # (penalty 0.5 on file 1 makes b0=0.85 beat a1=0.9-0.5=0.4).
    assert files_out[0] == 1
    assert files_out[1] == 2


# ────────────────────────────────────────────────────────────────────
# Identifier-aware FTS
# ────────────────────────────────────────────────────────────────────


def test_expand_idents_camel_split():
    out = _expand_idents("def getUserById(): return findUserId()")
    assert "get User By Id" in out  # camel-split appended
    assert "find User Id" in out


def test_expand_idents_snake_split():
    out = _expand_idents("process_payment_request")
    # The snake-split parts are appended.
    assert " process " in f" {out} "
    assert " payment " in f" {out} "
    assert " request " in f" {out} "


def test_fts_text_includes_qn_and_body_fanout():
    out = _fts_text("pkg.UserService.getById", "def getById(self, id): return _lookup(id)")
    # qn raw + qn camel-split + body + body camel-split.
    assert "pkg.UserService.getById" in out
    assert "User Service" in out  # qn camel-split
    assert "get By Id" in out  # body camel-split (`getById`)


def test_tsquery_query_side_fanout():
    tsq = _build_tsquery("getUserById")
    assert tsq is not None
    # Original lowercase token preserved.
    assert "getuserbyid:*" in tsq
    # Camel-split parts emitted as separate prefix-matched tokens.
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
    """The python fixture has `helper(x)` and `Calculator.add(self, x)`.
    A lexical query for the body identifier `helper` should find chunks
    that mention it, including those whose anchor isn't `helper` itself."""
    pool, repo_id = clean_repo
    await _seed_python(pool, repo_id, python_fixture_root)
    chunks = await lexical_query(pool, repo_id, "helper", top_k=20)
    qns = {c.qualified_name for c in chunks}
    assert "utils.helper" in qns


# ────────────────────────────────────────────────────────────────────
# RRF fusion
# ────────────────────────────────────────────────────────────────────


def _mk_chunk(cid: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, anchor_def_id=cid, qualified_name=f"x{cid}",
        granularity="function", file_path=f"f{cid}.py", file_id=cid,
        token_count=10, content="", score=score, repo_id=1,
    )


def test_rrf_fusion_unaffected_by_score_skew():
    """A single dominant cosine score should NOT drown out lexical hits.
    Under the old normalize-by-max fusion, the lexical's top item barely
    moves the dial; under RRF the rank-1 lexical hit is comparable to
    rank-1 semantic."""
    sem = [_mk_chunk(1, score=999.0)]  # one outlier-high "cosine" score
    sem += [_mk_chunk(i, score=0.001) for i in range(2, 12)]
    lex = [_mk_chunk(100 + i, score=10.0 - i) for i in range(10)]
    by_id, scores = _fuse_scores(sem, lex)
    # The chunk-100 (rank-1 lexical, only-lexical) should appear in the top
    # results with a score competitive with chunk-1 (rank-1 semantic only).
    assert 1 in by_id and 100 in by_id
    # Both top-1 ranks contribute SEM_WEIGHT/(K+1) and LEX_WEIGHT/(K+1)
    # respectively, which are equal. So scores are equal.
    assert abs(scores[1] - scores[100]) < 1e-9


def test_rrf_normalized_for_graph_expansion():
    """Fused scores should be normalized to [0, 1] so that graph bonus
    constants (`graph_weight=0.3`) keep their "fraction of seed rank"
    meaning."""
    sem = [_mk_chunk(i) for i in range(1, 6)]
    lex = [_mk_chunk(100 + i) for i in range(5)]
    _, scores = _fuse_scores(sem, lex)
    assert scores
    assert max(scores.values()) == 1.0
    assert min(scores.values()) > 0.0
