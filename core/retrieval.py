"""Tier 3c: dual retrieval interface (Architecture §6.4).

    structural_query(name, depth)  — start at a named definition, walk
                                     call_edges outbound, return chunks.
    semantic_query(query, top_k)   — embed the query, pgvector cosine
                                     nearest-neighbour over chunk_embeddings.
    lexical_query(query, top_k)    — Postgres full-text search (BM25 via
                                     ts_rank_cd) over chunks.fts_doc.
    hybrid_query(query, top_k)     — semantic + lexical candidates fused
                                     via Reciprocal Rank Fusion, then 1-hop
                                     graph expansion + simple weighted
                                     rerank.
    assemble_context(chunks, budget) — dedupe overlapping byte ranges,
                                       greedy fit under token budget.

Branch model: every retrieval is scoped to a list of branch_ids. Each chunk
is per-branch (the inlined skeleton encodes branch-resolved cross-file
targets). chunk_embeddings is content-hash keyed and shared across branches
so identical chunk text reuses one vector.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import asyncpg

from .chunk_assembler import _split_camel
from .embedder import EmbedFn, _vector_literal


@dataclass
class RetrievedChunk:
    chunk_id: int
    anchor_def_id: int | None
    qualified_name: str | None
    granularity: str
    file_path: str
    file_version_id: int
    branch_id: int
    token_count: int
    content: str
    score: float                     # higher is better; meaning depends on query mode
    repo_id: int | None = None       # which repo this chunk came from (multi-repo aware)


def _norm_branch_ids(x: int | list[int]) -> list[int]:
    """`semantic_query(branch_ids=42)` is sugar for `branch_ids=[42]`."""
    if isinstance(x, int):
        return [x]
    return list(x)


# ────────────────────────────────────────────────────────────────────
# Structural — graph traversal from a named definition
# ────────────────────────────────────────────────────────────────────


async def structural_query(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    definition_name: str,
    *,
    depth: int = 2,
    granularity: str = "function",
) -> list[RetrievedChunk]:
    """Find a definition by name (or qualified_name suffix) and return its
    chunk plus chunks for transitive callees up to `depth` hops, scoped to
    the given branches. Score = 1.0 at hop 0, halved per hop.

    Definitions are content-shared across branches but the caller is asking
    for results visible in `branch_ids`, so we restrict the seed lookup to
    defs whose file_version is mapped by at least one of those branches.
    Graph walks over `call_edges` are filtered by branch_id too — calls
    can resolve to different callees in different branches.
    """
    bids = _norm_branch_ids(branch_ids)
    async with pool.acquire() as conn:
        seeds = await conn.fetch(
            """
            SELECT DISTINCT d.id FROM definitions d
            JOIN branch_files bf ON bf.file_version_id = d.file_version_id
            WHERE bf.branch_id = ANY($1::bigint[])
              AND (d.name = $2 OR d.qualified_name = $2 OR d.qualified_name LIKE '%.' || $2)
            """,
            bids, definition_name,
        )
        if not seeds:
            return []
        scores: dict[int, float] = {row["id"]: 1.0 for row in seeds}
        frontier: set[int] = set(scores.keys())
        for hop in range(1, depth + 1):
            next_frontier: set[int] = set()
            for caller in frontier:
                edges = await conn.fetch(
                    """
                    SELECT DISTINCT callee_def_id FROM call_edges
                    WHERE caller_def_id=$1 AND callee_def_id IS NOT NULL
                      AND branch_id = ANY($2::bigint[])
                    """,
                    caller, bids,
                )
                for e in edges:
                    cid = e["callee_def_id"]
                    if cid in scores:
                        continue
                    scores[cid] = 1.0 / (2**hop)
                    next_frontier.add(cid)
            if not next_frontier:
                break
            frontier = next_frontier

        return await _fetch_chunks_for_anchors(
            conn, list(scores.keys()), scores, granularity, branch_ids=bids,
        )


# ────────────────────────────────────────────────────────────────────
# Semantic — embed query, pgvector cosine nearest
# ────────────────────────────────────────────────────────────────────


SKELETON_PENALTY_FLOOR = 0.3
SKELETON_GRAPH_REF = 3       # def-level signal at which the factor reaches 1.0
SKELETON_FILE_GRAPH_REF = 10 # file-level signal threshold for module chunks

_GRAPH_SIGNAL_SQL = "(COALESCE(out_calls.n, 0) + COALESCE(out_data.n, 0))"
_FILE_GRAPH_SIGNAL_SQL = "COALESCE(file_sig.n, 0)"
_SKELETON_FACTOR_SQL = (
    f"(CASE WHEN d.kind = 'module' "
    f"      THEN GREATEST({SKELETON_PENALTY_FLOOR}, "
    f"                    LEAST(1.0, {_FILE_GRAPH_SIGNAL_SQL}::float / {SKELETON_FILE_GRAPH_REF})) "
    f"      ELSE GREATEST({SKELETON_PENALTY_FLOOR}, "
    f"                    LEAST(1.0, {_GRAPH_SIGNAL_SQL}::float / {SKELETON_GRAPH_REF})) END)"
)
# JOIN block reused by every retriever that applies the skeleton factor.
# Aggregations are keyed by (def_id, branch_id) / (file_version_id, branch_id)
# so we can JOIN on the outer chunk's branch_id without needing LATERAL.
# Postgres lazily evaluates subqueries with predicate pushdown; the WHERE
# branch_id = ANY($BRANCHES) on the outer query effectively restricts these
# aggregations to the queried branches via the join key.
_SKELETON_JOINS_SQL = """
                LEFT JOIN (
                    SELECT caller_def_id AS did, branch_id, COUNT(*) AS n
                    FROM call_edges
                    WHERE caller_def_id IS NOT NULL
                    GROUP BY caller_def_id, branch_id
                ) out_calls ON out_calls.did = d.id AND out_calls.branch_id = c.branch_id
                LEFT JOIN (
                    SELECT accessor_def_id AS did, branch_id, COUNT(*) AS n
                    FROM data_access
                    GROUP BY accessor_def_id, branch_id
                ) out_data ON out_data.did = d.id AND out_data.branch_id = c.branch_id
                LEFT JOIN (
                    SELECT defs.file_version_id, defs.branch_id, COUNT(*) AS n
                    FROM (
                        SELECT d2.file_version_id, ce2.branch_id
                        FROM call_edges ce2
                        JOIN definitions d2 ON d2.id = ce2.caller_def_id
                        WHERE ce2.caller_def_id IS NOT NULL
                        UNION ALL
                        SELECT d2.file_version_id, da2.branch_id
                        FROM data_access da2
                        JOIN definitions d2 ON d2.id = da2.accessor_def_id
                    ) AS defs
                    GROUP BY defs.file_version_id, defs.branch_id
                ) file_sig
                    ON file_sig.file_version_id = c.file_version_id
                       AND file_sig.branch_id = c.branch_id
"""
_SCORE_EXPR = f"(1.0 - (ce.embedding <=> $1::vector)) * {_SKELETON_FACTOR_SQL}"


async def semantic_query(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    query: str,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    granularities: tuple[str, ...] | None = None,
) -> list[RetrievedChunk]:
    """Embed `query` and return the top-k nearest chunks by cosine distance,
    with a graph-signal penalty applied so signature-only defs don't crowd
    out real implementations."""
    vectors = await embed_fn([query])
    if not vectors:
        return []
    qvec = _vector_literal(vectors[0])
    bids = _norm_branch_ids(branch_ids)

    async with pool.acquire() as conn:
        if granularities:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       c.branch_id, c.file_version_id,
                       d.qualified_name, bf.path AS file_path,
                       b.repo_id AS repo_id,
                       {_SCORE_EXPR} AS score
                FROM chunks c
                JOIN branches b ON b.id = c.branch_id
                JOIN branch_files bf ON bf.branch_id = c.branch_id AND bf.file_version_id = c.file_version_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                JOIN chunk_embeddings ce ON ce.content_hash = c.content_hash
                WHERE c.branch_id = ANY($2::bigint[]) AND c.granularity = ANY($3::text[])
                ORDER BY score DESC
                LIMIT $4
                """,
                qvec, bids, list(granularities), top_k,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       c.branch_id, c.file_version_id,
                       d.qualified_name, bf.path AS file_path,
                       b.repo_id AS repo_id,
                       {_SCORE_EXPR} AS score
                FROM chunks c
                JOIN branches b ON b.id = c.branch_id
                JOIN branch_files bf ON bf.branch_id = c.branch_id AND bf.file_version_id = c.file_version_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                JOIN chunk_embeddings ce ON ce.content_hash = c.content_hash
                WHERE c.branch_id = ANY($2::bigint[])
                ORDER BY score DESC
                LIMIT $3
                """,
                qvec, bids, top_k,
            )
    return [_row_to_chunk(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Lexical — Postgres FTS over chunks.fts_doc (BM25-style ranking)
# ────────────────────────────────────────────────────────────────────


_TSQUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LEX_SCORE_EXPR = f"ts_rank_cd(c.fts_doc, to_tsquery('english', $1)) * {_SKELETON_FACTOR_SQL}"


def _build_tsquery(query: str) -> str | None:
    """Sanitize a free-text query into an OR'd `tsquery` string."""
    raw_tokens = _TSQUERY_TOKEN_RE.findall(query)
    if not raw_tokens:
        return None
    expanded: list[str] = []
    for t in raw_tokens:
        expanded.append(t.lower())
        if "_" in t:
            expanded.extend(p.lower() for p in t.split("_") if p)
        camel = _split_camel(t)
        if camel != t:
            expanded.extend(p.lower() for p in camel.split() if p)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in expanded:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    if not deduped:
        return None
    return " | ".join(f"{t}:*" for t in deduped)


async def lexical_query(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    query: str,
    *,
    top_k: int = 10,
    granularities: tuple[str, ...] | None = None,
) -> list[RetrievedChunk]:
    """Lexical retrieval via Postgres full-text search."""
    tsq = _build_tsquery(query)
    if tsq is None:
        return []
    bids = _norm_branch_ids(branch_ids)

    async with pool.acquire() as conn:
        if granularities:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       c.branch_id, c.file_version_id,
                       d.qualified_name, bf.path AS file_path,
                       b.repo_id AS repo_id,
                       {_LEX_SCORE_EXPR} AS score
                FROM chunks c
                JOIN branches b ON b.id = c.branch_id
                JOIN branch_files bf ON bf.branch_id = c.branch_id AND bf.file_version_id = c.file_version_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                WHERE c.branch_id = ANY($2::bigint[]) AND c.granularity = ANY($3::text[])
                  AND c.fts_doc @@ to_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $4
                """,
                tsq, bids, list(granularities), top_k,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       c.branch_id, c.file_version_id,
                       d.qualified_name, bf.path AS file_path,
                       b.repo_id AS repo_id,
                       {_LEX_SCORE_EXPR} AS score
                FROM chunks c
                JOIN branches b ON b.id = c.branch_id
                JOIN branch_files bf ON bf.branch_id = c.branch_id AND bf.file_version_id = c.file_version_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                WHERE c.branch_id = ANY($2::bigint[])
                  AND c.fts_doc @@ to_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $3
                """,
                tsq, bids, top_k,
            )
    return [_row_to_chunk(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Hybrid — semantic + 1-hop graph expansion
# ────────────────────────────────────────────────────────────────────


RRF_K = 60
SEM_WEIGHT = 0.5
LEX_WEIGHT = 0.5


def _fuse_scores(
    sem_results: list[RetrievedChunk],
    lex_results: list[RetrievedChunk],
) -> tuple[dict[int, RetrievedChunk], dict[int, float]]:
    """Reciprocal Rank Fusion."""
    by_id: dict[int, RetrievedChunk] = {}
    score: dict[int, float] = {}

    for rank, c in enumerate(sem_results, start=1):
        score[c.chunk_id] = score.get(c.chunk_id, 0.0) + SEM_WEIGHT / (RRF_K + rank)
        by_id[c.chunk_id] = c

    for rank, c in enumerate(lex_results, start=1):
        score[c.chunk_id] = score.get(c.chunk_id, 0.0) + LEX_WEIGHT / (RRF_K + rank)
        by_id.setdefault(c.chunk_id, c)

    if score:
        m = max(score.values())
        if m > 0:
            for k in score:
                score[k] = score[k] / m

    return by_id, score


async def hybrid_query(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    query: str,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    candidate_pool: int = 20,
    graph_weight: float = 0.3,
    override_weight: float = 0.5,
    mmr_repo_lambda: float = 0.3,
    mmr_file_lambda: float = 0.15,
) -> list[RetrievedChunk]:
    """Pull `candidate_pool` candidates from BOTH the semantic (cosine) and
    lexical (BM25 / FTS) retrievers, fuse via Reciprocal Rank Fusion, then
    expand each anchor via call_edges + inherits_edges + overrides_edges
    (all branch-scoped), then MMR-rerank for repo/file diversity.
    """
    bids = _norm_branch_ids(branch_ids)
    sem_results, lex_results = await asyncio.gather(
        semantic_query(pool, bids, query, embed_fn, top_k=candidate_pool),
        lexical_query(pool, bids, query, top_k=candidate_pool),
    )
    if not sem_results and not lex_results:
        return []

    by_id, scores = _fuse_scores(sem_results, lex_results)
    candidates = sorted(by_id.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    for c in candidates:
        c.score = scores[c.chunk_id]

    cosine_by_chunk: dict[int, float] = {c.chunk_id: c.score for c in sem_results}

    anchor_scores: dict[int, float] = {}
    for c in candidates:
        if c.anchor_def_id is None:
            continue
        cs = cosine_by_chunk.get(c.chunk_id, 0.0)
        if cs > anchor_scores.get(c.anchor_def_id, float("-inf")):
            anchor_scores[c.anchor_def_id] = cs

    if not anchor_scores:
        ranked = sorted(by_id.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        for c in ranked:
            c.score = scores[c.chunk_id]
        return _mmr_rerank(
            ranked, scores, top_k,
            repo_lambda=mmr_repo_lambda, file_lambda=mmr_file_lambda,
        )

    anchor_ids = list(anchor_scores.keys())
    conf_weight = {"certain": 1.0, "inferred": 0.7, "uncertain": 0.4}

    related_bonus: dict[int, float] = {}

    def _add_bonus(target_def_id: int, source_anchor: int, weight: float) -> None:
        bonus = anchor_scores.get(source_anchor, 0.0) * weight * graph_weight
        if bonus > related_bonus.get(target_def_id, 0.0):
            related_bonus[target_def_id] = bonus

    async with pool.acquire() as conn:
        # ── 1. call_edges: caller → callees (1 hop outbound) ──
        call_rows = await conn.fetch(
            """
            SELECT caller_def_id, callee_def_id, confidence
            FROM call_edges
            WHERE caller_def_id = ANY($1::bigint[]) AND callee_def_id IS NOT NULL
              AND branch_id = ANY($2::bigint[])
            """,
            anchor_ids, bids,
        )
        for r in call_rows:
            _add_bonus(r["callee_def_id"], r["caller_def_id"],
                       conf_weight.get(r["confidence"], 0.5))

        effective_anchor_scores = dict(anchor_scores)
        for cid, bonus in related_bonus.items():
            if cid in effective_anchor_scores:
                effective_anchor_scores[cid] = effective_anchor_scores[cid] + bonus
        callee_seed_score: dict[int, float] = {}
        for r in call_rows:
            seed = effective_anchor_scores.get(r["caller_def_id"], 0.0)
            if seed > callee_seed_score.get(r["callee_def_id"], 0.0):
                callee_seed_score[r["callee_def_id"]] = seed
        expanded_scores = dict(effective_anchor_scores)
        for cid, seed in callee_seed_score.items():
            if seed > expanded_scores.get(cid, float("-inf")):
                expanded_scores[cid] = seed
        expanded_ids = list(expanded_scores.keys())

        # ── 2. inherits_edges: bidirectional ──
        inh_rows = await conn.fetch(
            """
            SELECT child_def_id, base_def_id, confidence
            FROM inherits_edges
            WHERE base_def_id IS NOT NULL
              AND branch_id = ANY($2::bigint[])
              AND (base_def_id = ANY($1::bigint[]) OR child_def_id = ANY($1::bigint[]))
            """,
            expanded_ids, bids,
        )
        for r in inh_rows:
            w = conf_weight.get(r["confidence"], 0.5)
            if r["base_def_id"] in expanded_scores:
                src_score = expanded_scores[r["base_def_id"]]
                bonus = src_score * w * graph_weight
                if bonus > related_bonus.get(r["child_def_id"], 0.0):
                    related_bonus[r["child_def_id"]] = bonus
            if r["child_def_id"] in expanded_scores:
                src_score = expanded_scores[r["child_def_id"]]
                bonus = src_score * w * graph_weight
                if bonus > related_bonus.get(r["base_def_id"], 0.0):
                    related_bonus[r["base_def_id"]] = bonus

        # ── 3. overrides_edges: bidirectional ──
        ovr_rows = await conn.fetch(
            """
            SELECT child_def_id, base_def_id
            FROM overrides_edges
            WHERE branch_id = ANY($2::bigint[])
              AND (child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[]))
            """,
            expanded_ids, bids,
        )
        for r in ovr_rows:
            if r["base_def_id"] in expanded_scores:
                src_score = expanded_scores[r["base_def_id"]]
                bonus = src_score * override_weight
                if bonus > related_bonus.get(r["child_def_id"], 0.0):
                    related_bonus[r["child_def_id"]] = bonus
            if r["child_def_id"] in expanded_scores:
                src_score = expanded_scores[r["child_def_id"]]
                bonus = src_score * override_weight
                if bonus > related_bonus.get(r["base_def_id"], 0.0):
                    related_bonus[r["base_def_id"]] = bonus

        if related_bonus:
            related_ids = sorted(related_bonus.keys())
            related_chunks = await _fetch_chunks_for_anchors(
                conn, related_ids, scores={cid: 1.0 for cid in related_ids},
                granularity="function", branch_ids=bids,
            )
            for nc in related_chunks:
                bonus = related_bonus.get(nc.anchor_def_id or -1, 0.0)
                if bonus == 0.0:
                    continue
                if nc.chunk_id in by_id:
                    scores[nc.chunk_id] = scores[nc.chunk_id] + bonus
                else:
                    by_id[nc.chunk_id] = nc
                    scores[nc.chunk_id] = bonus

    ranked = sorted(by_id.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    for c in ranked:
        c.score = scores[c.chunk_id]
    return _mmr_rerank(
        ranked, scores, top_k,
        repo_lambda=mmr_repo_lambda, file_lambda=mmr_file_lambda,
    )


def _mmr_rerank(
    candidates: list[RetrievedChunk],
    scores: dict[int, float],
    top_k: int,
    *,
    repo_lambda: float = 0.3,
    file_lambda: float = 0.15,
) -> list[RetrievedChunk]:
    """Greedy MMR with per-repo + per-file diversity penalties."""
    if not candidates:
        return []
    distinct_repos = {c.repo_id for c in candidates if c.repo_id is not None}
    distinct_files = {c.file_version_id for c in candidates}
    if len(distinct_repos) <= 1 and len(distinct_files) <= 1:
        return candidates[:top_k]

    remaining = list(candidates)
    picked: list[RetrievedChunk] = []
    repo_count: dict[int | None, int] = {}
    file_count: dict[int, int] = {}

    while remaining and len(picked) < top_k:
        best_idx = 0
        best_adj = float("-inf")
        for i, c in enumerate(remaining):
            penalty = (
                repo_lambda * repo_count.get(c.repo_id, 0)
                + file_lambda * file_count.get(c.file_version_id, 0)
            )
            adj = scores.get(c.chunk_id, 0.0) - penalty
            if adj > best_adj:
                best_adj = adj
                best_idx = i
        chosen = remaining.pop(best_idx)
        picked.append(chosen)
        repo_count[chosen.repo_id] = repo_count.get(chosen.repo_id, 0) + 1
        file_count[chosen.file_version_id] = file_count.get(chosen.file_version_id, 0) + 1
    return picked


# ────────────────────────────────────────────────────────────────────
# Context assembly
# ────────────────────────────────────────────────────────────────────


def assemble_context(chunks: list[RetrievedChunk], token_budget: int) -> str:
    """Greedy fit chunks under `token_budget`, deduping overlapping chunks
    that share the same anchor."""
    seen_anchors: set[tuple[int, str]] = set()
    used: list[RetrievedChunk] = []
    remaining = token_budget
    for c in chunks:
        key = (c.anchor_def_id or -c.chunk_id, c.granularity)
        if key in seen_anchors:
            continue
        if c.token_count > remaining:
            continue
        seen_anchors.add(key)
        used.append(c)
        remaining -= c.token_count
    return "\n\n".join(c.content for c in used)


# ────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────


def _row_to_chunk(r: asyncpg.Record) -> RetrievedChunk:
    keys = set(r.keys())
    return RetrievedChunk(
        chunk_id=r["id"],
        anchor_def_id=r["anchor_def_id"],
        qualified_name=r["qualified_name"],
        granularity=r["granularity"],
        file_path=r["file_path"],
        file_version_id=r["file_version_id"],
        branch_id=r["branch_id"],
        token_count=r["token_count"] or 0,
        content=r["content"],
        score=float(r["score"]) if "score" in keys else 0.0,
        repo_id=r["repo_id"] if "repo_id" in keys else None,
    )


async def _fetch_chunks_for_anchors(
    conn: asyncpg.Connection,
    anchor_ids: list[int],
    scores: dict[int, float],
    granularity: str,
    *,
    branch_ids: list[int] | None = None,
) -> list[RetrievedChunk]:
    if not anchor_ids:
        return []
    if branch_ids:
        rows = await conn.fetch(
            """
            SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                   c.branch_id, c.file_version_id,
                   d.qualified_name, bf.path AS file_path,
                   b.repo_id AS repo_id,
                   0::float AS score
            FROM chunks c
            JOIN branches b ON b.id = c.branch_id
            JOIN branch_files bf ON bf.branch_id = c.branch_id AND bf.file_version_id = c.file_version_id
            LEFT JOIN definitions d ON d.id = c.anchor_def_id
            WHERE c.anchor_def_id = ANY($1::bigint[]) AND c.granularity = $2
              AND c.branch_id = ANY($3::bigint[])
            """,
            anchor_ids, granularity, branch_ids,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                   c.branch_id, c.file_version_id,
                   d.qualified_name, bf.path AS file_path,
                   b.repo_id AS repo_id,
                   0::float AS score
            FROM chunks c
            JOIN branches b ON b.id = c.branch_id
            JOIN branch_files bf ON bf.branch_id = c.branch_id AND bf.file_version_id = c.file_version_id
            LEFT JOIN definitions d ON d.id = c.anchor_def_id
            WHERE c.anchor_def_id = ANY($1::bigint[]) AND c.granularity = $2
            """,
            anchor_ids, granularity,
        )
    out: list[RetrievedChunk] = []
    for r in rows:
        c = _row_to_chunk(r)
        c.score = scores.get(c.anchor_def_id or 0, 0.0)
        out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    return out
