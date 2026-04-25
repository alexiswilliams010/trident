"""Tier 3c: dual retrieval interface (Architecture §6.4).

    structural_query(name, depth)  — start at a named definition, walk
                                     call_edges outbound, return chunks.
    semantic_query(query, top_k)   — embed the query, pgvector cosine
                                     nearest-neighbour over chunk_embeddings.
    hybrid_query(query, top_k)     — semantic candidates + 1-hop graph
                                     expansion + simple weighted rerank.
    assemble_context(chunks, budget) — dedupe overlapping byte ranges,
                                       greedy fit under token budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

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


# ────────────────────────────────────────────────────────────────────
# Structural — graph traversal from a named definition
# ────────────────────────────────────────────────────────────────────


async def structural_query(
    pool: asyncpg.Pool,
    repo_id: int,
    definition_name: str,
    *,
    depth: int = 2,
    granularity: str = "function",
) -> list[RetrievedChunk]:
    """Find a definition by name (or qualified_name suffix) and return its
    chunk plus chunks for transitive callees up to `depth` hops.

    Score = 1.0 at hop 0, halved per hop.
    """
    async with pool.acquire() as conn:
        seeds = await conn.fetch(
            """
            SELECT d.id FROM definitions d
            JOIN files f ON f.id = d.file_id
            WHERE f.repo_id = $1
              AND (d.name = $2 OR d.qualified_name = $2 OR d.qualified_name LIKE '%.' || $2)
            """,
            repo_id, definition_name,
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

        return await _fetch_chunks_for_anchors(conn, list(scores.keys()), scores, granularity)


# ────────────────────────────────────────────────────────────────────
# Semantic — embed query, pgvector cosine nearest
# ────────────────────────────────────────────────────────────────────


async def semantic_query(
    pool: asyncpg.Pool,
    repo_id: int,
    query: str,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    granularities: tuple[str, ...] | None = None,
) -> list[RetrievedChunk]:
    """Embed `query` and return the top-k nearest chunks by cosine distance."""
    vectors = await embed_fn([query])
    if not vectors:
        return []
    qvec = _vector_literal(vectors[0])

    async with pool.acquire() as conn:
        if granularities:
            rows = await conn.fetch(
                """
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       d.qualified_name, f.path AS file_path, f.id AS file_id,
                       1.0 - (ce.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                WHERE f.repo_id = $2 AND c.granularity = ANY($3::text[])
                ORDER BY ce.embedding <=> $1::vector
                LIMIT $4
                """,
                qvec, repo_id, list(granularities), top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
                       d.qualified_name, f.path AS file_path, f.id AS file_id,
                       1.0 - (ce.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                LEFT JOIN definitions d ON d.id = c.anchor_def_id
                JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                WHERE f.repo_id = $2
                ORDER BY ce.embedding <=> $1::vector
                LIMIT $3
                """,
                qvec, repo_id, top_k,
            )
    return [_row_to_chunk(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Hybrid — semantic + 1-hop graph expansion
# ────────────────────────────────────────────────────────────────────


async def hybrid_query(
    pool: asyncpg.Pool,
    repo_id: int,
    query: str,
    embed_fn: EmbedFn,
    *,
    top_k: int = 10,
    candidate_pool: int = 20,
    graph_weight: float = 0.3,
) -> list[RetrievedChunk]:
    """Pull `candidate_pool` semantic candidates, expand each anchor by 1 hop
    via call_edges, blend graph proximity with semantic score, return top-k.
    """
    candidates = await semantic_query(pool, repo_id, query, embed_fn, top_k=candidate_pool)
    if not candidates:
        return []

    # Score map: chunk_id → score (start with semantic).
    by_id: dict[int, RetrievedChunk] = {c.chunk_id: c for c in candidates}
    scores: dict[int, float] = {c.chunk_id: c.score for c in candidates}

    # Expand 1 hop: for each candidate's anchor, boost (or add) callee chunks.
    async with pool.acquire() as conn:
        anchor_ids = [c.anchor_def_id for c in candidates if c.anchor_def_id is not None]
        if anchor_ids:
            neighbour_rows = await conn.fetch(
                """
                SELECT ce.caller_def_id, ce.callee_def_id, ce.confidence
                FROM call_edges ce
                WHERE ce.caller_def_id = ANY($1::bigint[]) AND ce.callee_def_id IS NOT NULL
                """,
                anchor_ids,
            )
            if neighbour_rows:
                # Take the MAX semantic score per anchor; a function and its
                # cross-module chunk share the same anchor_def_id and we must
                # not let the lower-scoring one overwrite the higher.
                caller_scores: dict[int, float] = {}
                for c in candidates:
                    if c.anchor_def_id is None:
                        continue
                    if c.score > caller_scores.get(c.anchor_def_id, float("-inf")):
                        caller_scores[c.anchor_def_id] = c.score
                conf_weight = {"certain": 1.0, "inferred": 0.7, "uncertain": 0.4}

                # callee_def_id → max bonus across incoming edges (any caller in the candidate pool).
                callee_bonus: dict[int, float] = {}
                for r in neighbour_rows:
                    cid = r["callee_def_id"]
                    bonus = caller_scores.get(r["caller_def_id"], 0.0) * \
                            conf_weight.get(r["confidence"], 0.5) * graph_weight
                    if bonus > callee_bonus.get(cid, 0.0):
                        callee_bonus[cid] = bonus

                callee_ids = sorted(callee_bonus.keys())
                neighbour_chunks = await _fetch_chunks_for_anchors(
                    conn, callee_ids, scores={cid: 1.0 for cid in callee_ids},
                    granularity="function",
                )
                for nc in neighbour_chunks:
                    bonus = callee_bonus.get(nc.anchor_def_id or -1, 0.0)
                    if bonus == 0.0:
                        continue
                    if nc.chunk_id in by_id:
                        # Boost an existing candidate.
                        scores[nc.chunk_id] = scores[nc.chunk_id] + bonus
                    else:
                        # Pull in a new neighbour chunk.
                        by_id[nc.chunk_id] = nc
                        scores[nc.chunk_id] = bonus

    ranked = sorted(by_id.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    for c in ranked:
        c.score = scores[c.chunk_id]
    return ranked[:top_k]


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
    return RetrievedChunk(
        chunk_id=r["id"],
        anchor_def_id=r["anchor_def_id"],
        qualified_name=r["qualified_name"],
        granularity=r["granularity"],
        file_path=r["file_path"],
        file_id=r["file_id"],
        token_count=r["token_count"] or 0,
        content=r["content"],
        score=float(r["score"]) if "score" in r.keys() else 0.0,
    )


async def _fetch_chunks_for_anchors(
    conn: asyncpg.Connection,
    anchor_ids: list[int],
    scores: dict[int, float],
    granularity: str,
) -> list[RetrievedChunk]:
    if not anchor_ids:
        return []
    rows = await conn.fetch(
        """
        SELECT c.id, c.anchor_def_id, c.granularity, c.token_count, c.content,
               d.qualified_name, f.path AS file_path, f.id AS file_id,
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
