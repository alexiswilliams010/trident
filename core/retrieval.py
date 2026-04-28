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
    file_id: int
    token_count: int
    content: str
    score: float                     # higher is better; meaning depends on query mode
    repo_id: int | None = None       # which repo this chunk came from (multi-repo aware)
    matched_view: str | None = None  # which embedding view produced the top score


def _norm_repo_ids(x: int | list[int]) -> list[int]:
    """`semantic_query(repo_ids=42)` is sugar for `repo_ids=[42]`. A single
    int is by far the most common shape; the list form is what cross-repo
    callers pass."""
    if isinstance(x, int):
        return [x]
    return list(x)


# ────────────────────────────────────────────────────────────────────
# Structural — graph traversal from a named definition
# ────────────────────────────────────────────────────────────────────


async def structural_query(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    definition_name: str,
    *,
    depth: int = 2,
    granularity: str = "function",
) -> list[RetrievedChunk]:
    """Find a definition by name (or qualified_name suffix) and return its
    chunk plus chunks for transitive callees up to `depth` hops.

    Score = 1.0 at hop 0, halved per hop.

    `repo_ids` can be a single int or a list. Cross-repo callers pass a
    list; the seed lookup spans all listed repos. Graph walks are intra-repo
    by construction (def IDs are globally unique and edges are created by
    intra-repo resolution), so no extra filter is needed during the walk.
    """
    rids = _norm_repo_ids(repo_ids)
    async with pool.acquire() as conn:
        seeds = await conn.fetch(
            """
            SELECT d.id FROM definitions d
            JOIN files f ON f.id = d.file_id
            WHERE f.repo_id = ANY($1::bigint[])
              AND (d.name = $2 OR d.qualified_name = $2 OR d.qualified_name LIKE '%.' || $2)
            """,
            rids, definition_name,
        )
        if not seeds:
            return []
        scores: dict[int, float] = {row["id"]: 1.0 for row in seeds}
        frontier: set[int] = set(scores.keys())
        for hop in range(1, depth + 1):
            next_frontier: set[int] = set()
            for caller in frontier:
                edges = await conn.fetch(
                    "SELECT callee_def_id FROM call_edges "
                    "WHERE caller_def_id=$1 AND callee_def_id IS NOT NULL",
                    caller,
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
            conn, list(scores.keys()), scores, granularity, repo_ids=rids,
        )


# ────────────────────────────────────────────────────────────────────
# Semantic — embed query, pgvector cosine nearest
# ────────────────────────────────────────────────────────────────────


# Skeleton-penalty knobs. We deprioritize chunks whose anchor def has no
# outgoing graph signal — declarations without bodies have zero rows in
# `call_edges WHERE caller_def_id = anchor` and zero rows in
# `data_access WHERE accessor_def_id = anchor`, while real implementations
# accumulate at least a few. The check is purely structural: a def with no
# outgoing graph edges *is* a skeleton by definition, regardless of how the
# source language spells the construct.
#
# For module-granularity chunks (whose anchor is the synthetic module def —
# file-level, no edges of its own), we use the SUM of all defs' outgoing
# edges across the file. A file containing only declarations ends up at
# zero; a normal implementation file has dozens. Same penalty applies.
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
# `out_calls` / `out_data` give the per-anchor (def-level) signal; `file_sig`
# gives the per-file signal used by the module branch above. The latter is
# computed once per query as a UNION ALL aggregate over the two edge tables.
_SKELETON_JOINS_SQL = """
                LEFT JOIN (
                    SELECT caller_def_id AS did, COUNT(*) AS n
                    FROM call_edges
                    WHERE caller_def_id IS NOT NULL
                    GROUP BY caller_def_id
                ) out_calls ON out_calls.did = d.id
                LEFT JOIN (
                    SELECT accessor_def_id AS did, COUNT(*) AS n
                    FROM data_access
                    GROUP BY accessor_def_id
                ) out_data ON out_data.did = d.id
                LEFT JOIN (
                    SELECT defs.file_id, COUNT(*) AS n
                    FROM (
                        SELECT d.file_id FROM call_edges ce
                        JOIN definitions d ON d.id = ce.caller_def_id
                        WHERE ce.caller_def_id IS NOT NULL
                        UNION ALL
                        SELECT d.file_id FROM data_access da
                        JOIN definitions d ON d.id = da.accessor_def_id
                    ) AS defs
                    GROUP BY defs.file_id
                ) file_sig ON file_sig.file_id = c.file_id
"""
# Lateral that picks the best-scoring view per chunk. Each chunk has one row
# per (view_kind, model) in `chunk_embeddings`; we want the closest one.
# `$2` is the optional view_kinds filter (NULL for "all views"); $1 is the
# query vector (text representation, cast to vector).
_BEST_VIEW_LATERAL = """
LEFT JOIN LATERAL (
    SELECT ce.view_kind, ce.embedding <=> $1::vector AS dist
    FROM chunk_embeddings ce
    WHERE ce.chunk_id = c.id
      AND ($2::text[] IS NULL OR ce.view_kind = ANY($2::text[]))
    ORDER BY dist
    LIMIT 1
) best_view ON TRUE
"""

_SCORE_EXPR = f"(1.0 - best_view.dist) * {_SKELETON_FACTOR_SQL}"


async def semantic_query(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    query: str,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    granularities: tuple[str, ...] | None = None,
    view_kinds: tuple[str, ...] | None = None,
) -> list[RetrievedChunk]:
    """Embed `query` and return the top-k nearest chunks by cosine distance,
    with a graph-signal penalty applied so signature-only defs don't crowd
    out real implementations.

    Each chunk has multiple embedding "views" (e.g. raw source vs. source +
    inlined referenced types — see `core/embed_views.py`). The lateral inside
    the SQL picks the *best-scoring view per chunk* against the query vector,
    so a query that matches the enriched view will surface the chunk even if
    the source view alone wouldn't have.

    `view_kinds` filters which views to consider (e.g. `("source",)` to
    exactly reproduce pre-multi-view behavior); `None` means all views.
    """
    vectors = await embed_fn([query])
    if not vectors:
        return []
    qvec = _vector_literal(vectors[0])
    rids = _norm_repo_ids(repo_ids)
    vks = list(view_kinds) if view_kinds else None

    async with pool.acquire() as conn:
        if granularities:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       d.qualified_name, f.path AS file_path, f.id AS file_id,
                       f.repo_id AS repo_id, best_view.view_kind AS matched_view,
                       {_SCORE_EXPR} AS score
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                {_BEST_VIEW_LATERAL}
                WHERE f.repo_id = ANY($3::bigint[])
                  AND c.granularity = ANY($4::text[])
                  AND best_view.dist IS NOT NULL
                ORDER BY score DESC
                LIMIT $5
                """,
                qvec, vks, rids, list(granularities), top_k,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       d.qualified_name, f.path AS file_path, f.id AS file_id,
                       f.repo_id AS repo_id, best_view.view_kind AS matched_view,
                       {_SCORE_EXPR} AS score
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                {_BEST_VIEW_LATERAL}
                WHERE f.repo_id = ANY($3::bigint[])
                  AND best_view.dist IS NOT NULL
                ORDER BY score DESC
                LIMIT $4
                """,
                qvec, vks, rids, top_k,
            )
    return [_row_to_chunk(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Lexical — Postgres FTS over chunks.fts_doc (BM25-style ranking)
# ────────────────────────────────────────────────────────────────────


_TSQUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LEX_SCORE_EXPR = f"ts_rank_cd(c.fts_doc, to_tsquery('english', $1)) * {_SKELETON_FACTOR_SQL}"


def _build_tsquery(query: str) -> str | None:
    """Sanitize a free-text query into an OR'd `tsquery` string. Returns None
    if no valid tokens.

    Each token is suffixed with `:*` for prefix matching, and compound
    identifiers are fanned out to mirror the indexer (`chunk_assembler.
    _expand_idents`): `getUserById` becomes `getuserbyid:* | get:* |
    user:* | by:* | id:*`, so a body containing `getUserById` matches
    a query for any of those parts and vice-versa.

    OR semantics (rather than `plainto_tsquery`'s implicit AND) because
    multi-word queries often mention several alternative terms — a query
    like "user permission check" should still surface chunks that match
    `permission` even without `user` or `check` present. The cover-density
    rank function will rank chunks matching multiple terms above
    single-term hits.

    Tokens are constrained to `[A-Za-z0-9_]` so user-supplied content can't
    inject tsquery operators (`!`, `&`, `|`, parens).
    """
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
    repo_ids: int | list[int],
    query: str,
    *,
    top_k: int = 10,
    granularities: tuple[str, ...] | None = None,
) -> list[RetrievedChunk]:
    """Lexical retrieval via Postgres full-text search. Uses `ts_rank_cd`
    (cover density — weights term-proximity highly) over the precomputed
    `chunks.fts_doc` column. Same skeleton-penalty multiplier as
    `semantic_query` so this path doesn't surface signature-only defs
    either.

    Lexical retrieval is single-channel by design (no view aggregation):
    `chunks.fts_doc` is one tsvector per chunk, populated at chunk-assembly
    time with body-identifier fan-out (see `chunk_assembler._fts_text`).
    """
    tsq = _build_tsquery(query)
    if tsq is None:
        return []
    rids = _norm_repo_ids(repo_ids)

    async with pool.acquire() as conn:
        if granularities:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       d.qualified_name, f.path AS file_path, f.id AS file_id,
                       f.repo_id AS repo_id,
                       {_LEX_SCORE_EXPR} AS score
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                WHERE f.repo_id = ANY($2::bigint[]) AND c.granularity = ANY($3::text[])
                  AND c.fts_doc @@ to_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $4
                """,
                tsq, rids, list(granularities), top_k,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       d.qualified_name, f.path AS file_path, f.id AS file_id,
                       f.repo_id AS repo_id,
                       {_LEX_SCORE_EXPR} AS score
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                {_SKELETON_JOINS_SQL}
                WHERE f.repo_id = ANY($2::bigint[])
                  AND c.fts_doc @@ to_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $3
                """,
                tsq, rids, top_k,
            )
    return [_row_to_chunk(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Hybrid — semantic + 1-hop graph expansion
# ────────────────────────────────────────────────────────────────────


RRF_K = 60        # Standard constant from the RRF paper. Damps the head-of-list
                  # advantage so rank 1 isn't disproportionately above rank 2.
SEM_WEIGHT = 0.5  # Per-retriever weight on the RRF contribution. Equal weights
LEX_WEIGHT = 0.5  # by default; bump SEM > LEX for prose-y queries, LEX > SEM
                  # for identifier-heavy queries.


def _fuse_scores(
    sem_results: list[RetrievedChunk],
    lex_results: list[RetrievedChunk],
) -> tuple[dict[int, RetrievedChunk], dict[int, float]]:
    """Reciprocal Rank Fusion. Each retriever contributes
        weight_i / (RRF_K + rank_i)
    per chunk it returned (1-indexed ranks). Chunks present in both
    retrievers sum the contributions; chunks in one still get that
    retriever's contribution.

    Robust to score-distribution skew: a single dominant cosine hit no
    longer drowns out the lexical retriever, which was the failure mode
    that the previous score-fusion variant kept tripping under the
    deterministic test embedder.

    Final scores are normalized to [0, 1] (divide by the max) so downstream
    graph bonuses (`graph_weight=0.3`, `override_weight=0.5`) keep their
    "fraction of the seed's rank" meaning. Without this normalization the
    raw RRF values are tiny (head of list ≈ 1/61 ≈ 0.016) and the bonus
    constants would need re-tuning.
    """
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
    repo_ids: int | list[int],
    query: str,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    candidate_pool: int = 20,
    graph_weight: float = 0.3,
    override_weight: float = 0.5,
    view_kinds: tuple[str, ...] | None = None,
    mmr_repo_lambda: float = 0.3,
    mmr_file_lambda: float = 0.15,
) -> list[RetrievedChunk]:
    """Pull `candidate_pool` candidates from BOTH the semantic (cosine) and
    lexical (BM25 / FTS) retrievers, fuse via Reciprocal Rank Fusion, then
    expand each anchor via:

      - call_edges, 1 hop outbound (caller → callees);
      - inherits_edges, bidirectional (children of a high-ranking base, bases
        of a high-ranking child) — `graph_weight`;
      - overrides_edges, bidirectional (overrides of a base method, base of an
        override) — `override_weight`, higher than `graph_weight` because an
        override IS the implementation of the base, not just structurally
        related.

    Then MMR-rerank with per-repo and per-file diversity penalties so the
    top-k spans repos and files instead of collapsing into the most-cosine-
    similar cluster. `mmr_repo_lambda` / `mmr_file_lambda` control the
    penalty magnitudes; both are no-ops when only one repo/file appears in
    the candidate pool.

    The lexical layer fixes the case where rare domain identifiers don't
    embed into recognizable neighbours but a plain keyword search nails
    them. Bonuses are blended additively, then results are re-ranked.
    Bidirectional walking on inheritance is what surfaces a child override
    when the query happens to match the base first, and vice versa.

    `repo_ids` may be a single int (single-repo query, sugar) or a list
    (cross-repo query). `view_kinds` filters which embedding views the
    semantic side is allowed to consider; `None` means all views.
    """
    rids = _norm_repo_ids(repo_ids)
    # Run cosine + lexical concurrently — they hit different indexes and
    # don't share state, so the second one is essentially free in wall time.
    sem_results, lex_results = await asyncio.gather(
        semantic_query(
            pool, rids, query, embed_fn,
            top_k=candidate_pool, view_kinds=view_kinds,
        ),
        lexical_query(pool, rids, query, top_k=candidate_pool),
    )
    if not sem_results and not lex_results:
        return []

    by_id, scores = _fuse_scores(sem_results, lex_results)
    # `scores` is already in [0, 1] by construction (each retriever
    # contributes ≤ its weight). Sort the candidate pool by fused score
    # for the graph-expansion seeding step below.
    candidates = sorted(by_id.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    for c in candidates:
        c.score = scores[c.chunk_id]

    # Cosine score by chunk_id, default 0 for chunks the cosine path didn't
    # surface. Used only for anchor_scores → bonus magnitudes; the final
    # ranking still goes through fused (normalized RRF) + bonuses.
    cosine_by_chunk: dict[int, float] = {c.chunk_id: c.score for c in sem_results}

    # Take the MAX cosine score per anchor (a function chunk and its
    # cross-module chunk share an anchor_def_id; we must not let the lower
    # one win). Anchor_scores feeds bonus computation only.
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

    # related_def_id → max bonus across all incoming edges from the candidate pool.
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
            """,
            anchor_ids,
        )
        for r in call_rows:
            _add_bonus(r["callee_def_id"], r["caller_def_id"],
                       conf_weight.get(r["confidence"], 0.5))

        # Promote call callees into the anchor set for the inheritance/override
        # passes below. Without this, a query that hits the public dispatcher
        # (e.g. `Policy.onExecute`) never reaches the virtual hook's overrides
        # (`Policy._onExecute → SingleExecutorPolicy._onExecute`), since the
        # override chain hangs off the *callee*, not the seed.
        #
        # Use the *effective* caller score (raw seed + any inbound call bonus
        # already accumulated in this pass) so that a callee transitively
        # dispatching to overrides receives a magnitude comparable to its
        # caller's effective rank. Without the effective score, a chain like
        # `top_candidate → onExecute → _onExecute → override_impl` collapses to
        # noise by the time it reaches the override.
        effective_anchor_scores = dict(anchor_scores)
        for cid, bonus in related_bonus.items():
            # If an anchor was also a call target from another anchor, its
            # effective score is the max(raw_semantic, raw_semantic + bonus).
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
        # Downward (base in pool → its children): "show me the overrides of
        # this base." Upward (child in pool → its bases): "show me what this
        # inherits from."
        inh_rows = await conn.fetch(
            """
            SELECT child_def_id, base_def_id, confidence
            FROM inherits_edges
            WHERE base_def_id IS NOT NULL
              AND (base_def_id = ANY($1::bigint[]) OR child_def_id = ANY($1::bigint[]))
            """,
            expanded_ids,
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

        # ── 3. overrides_edges: bidirectional, weighted higher than calls/inherits ──
        ovr_rows = await conn.fetch(
            """
            SELECT child_def_id, base_def_id
            FROM overrides_edges
            WHERE child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[])
            """,
            expanded_ids,
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
                granularity="function", repo_ids=rids,
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
    """Greedy MMR with per-repo + per-file diversity penalties.

    At each pick:
        adjusted = base_score - repo_lambda*n_repo_picked - file_lambda*n_file_picked

    `n_repo_picked` / `n_file_picked` are counts of items already chosen
    sharing the same repo / file. Returns up to `top_k`.

    No-op when the candidate pool spans only one repo and one file — the
    natural single-repo, single-file case shouldn't shuffle results.
    """
    if not candidates:
        return []
    distinct_repos = {c.repo_id for c in candidates if c.repo_id is not None}
    distinct_files = {c.file_id for c in candidates}
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
                + file_lambda * file_count.get(c.file_id, 0)
            )
            adj = scores.get(c.chunk_id, 0.0) - penalty
            if adj > best_adj:
                best_adj = adj
                best_idx = i
        chosen = remaining.pop(best_idx)
        picked.append(chosen)
        repo_count[chosen.repo_id] = repo_count.get(chosen.repo_id, 0) + 1
        file_count[chosen.file_id] = file_count.get(chosen.file_id, 0) + 1
    return picked


# ────────────────────────────────────────────────────────────────────
# Context assembly
# ────────────────────────────────────────────────────────────────────


def assemble_context(chunks: list[RetrievedChunk], token_budget: int) -> str:
    """Greedy fit chunks under `token_budget`, deduping overlapping chunks
    that share the same anchor file path."""
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
    # `r.keys()` returns a one-shot iterator on asyncpg.Record, so materialize
    # to a set before doing repeated containment checks.
    keys = set(r.keys())
    return RetrievedChunk(
        chunk_id=r["id"],
        anchor_def_id=r["anchor_def_id"],
        qualified_name=r["qualified_name"],
        granularity=r["granularity"],
        file_path=r["file_path"],
        file_id=r["file_id"],
        token_count=r["token_count"] or 0,
        content=r["content"],
        score=float(r["score"]) if "score" in keys else 0.0,
        repo_id=r["repo_id"] if "repo_id" in keys else None,
        matched_view=r["matched_view"] if "matched_view" in keys else None,
    )


async def _fetch_chunks_for_anchors(
    conn: asyncpg.Connection,
    anchor_ids: list[int],
    scores: dict[int, float],
    granularity: str,
    *,
    repo_ids: list[int] | None = None,
) -> list[RetrievedChunk]:
    if not anchor_ids:
        return []
    if repo_ids:
        rows = await conn.fetch(
            """
            SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                   d.qualified_name, f.path AS file_path, f.id AS file_id,
                   f.repo_id AS repo_id,
                   0::float AS score
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            LEFT JOIN definitions d ON d.id = c.anchor_def_id
            WHERE c.anchor_def_id = ANY($1::bigint[]) AND c.granularity = $2
              AND f.repo_id = ANY($3::bigint[])
            """,
            anchor_ids, granularity, repo_ids,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                   d.qualified_name, f.path AS file_path, f.id AS file_id,
                   f.repo_id AS repo_id,
                   0::float AS score
            FROM chunks c
            JOIN files f ON f.id = c.file_id
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
