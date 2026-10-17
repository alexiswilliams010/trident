"""Bounded cross-repository candidate retrieval; no relationship decisions or edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from uuid import UUID

import asyncpg

from core.embedder import EmbedFn, _vector_literal
from core.embedding_identity import EmbeddingIdentity, identity_for
from core.retrieval import RRF_K, _build_tsquery
from .snapshots import fingerprint, load_snapshot

RETRIEVAL_VERSION = "connections-rrf-v1"


@dataclass(frozen=True)
class RetrievalOptions:
    candidate_limit: int = 30
    channel_limit: int = 50
    exact_reserve: int = 8

    def __post_init__(self):
        if not 1 <= self.candidate_limit <= self.channel_limit <= 1000:
            raise ValueError("Require 1 <= candidate_limit <= channel_limit <= 1000")
        if not 0 <= self.exact_reserve <= self.candidate_limit:
            raise ValueError("Exact reserve must fit within the candidate budget")


@dataclass(frozen=True)
class CandidateHit:
    target_site_id: str
    target_repo_id: int
    path: str
    kind: str
    start_byte: int
    end_byte: int
    content_hash: str
    retrieval_hash: str
    retrieval_text: str
    metadata: dict
    fused_score: float
    channels: dict

    def signature(self):
        return [self.target_repo_id, self.path, self.kind, self.start_byte, self.end_byte,
                self.content_hash, self.retrieval_hash, self.channels]


@dataclass(frozen=True)
class RetrievalRun:
    id: str
    source_count: int
    candidate_count: int
    corpus_fingerprint: str


def _space_params(identity):
    return (identity.model_name, identity.model_version, identity.dimensions,
            identity.representation_version)


async def _ready(conn, sid, iv, identity):
    await load_snapshot(conn, sid)
    row = await conn.fetchrow("""
        SELECT representation_version,completed_at FROM connection_site_indexes
        WHERE snapshot_id=$1 AND index_version=$2
    """, sid, iv)
    if row is None or row["completed_at"] is None:
        raise ValueError("A complete site index is required")
    if identity.representation_version != row["representation_version"]:
        raise ValueError("Embedding identity must use the site index representation version")


async def embed_snapshot_sites(
    pool: asyncpg.Pool, snapshot_id: str | UUID, index_version: str, embed_fn: EmbedFn,
    *, embedding_identity: EmbeddingIdentity | None = None, batch_size: int = 64,
) -> int:
    """Embed unique retrieval inputs once per vector space, across snapshots/repos.

    Returns the number of inputs submitted. Concurrent callers may duplicate API
    work, but the unique key prevents duplicate stored vectors.
    """
    sid = UUID(str(snapshot_id))
    # Eligibility is checked before touching the embedder, including its identity.
    async with pool.acquire() as conn:
        await load_snapshot(conn, sid)
        identity = identity_for(embed_fn, embedding_identity)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        await _ready(conn, sid, index_version, identity)
        rows = await conn.fetch("""
            SELECT DISTINCT s.retrieval_hash,s.retrieval_text FROM interaction_sites s
            LEFT JOIN interaction_embeddings e ON e.input_hash=s.retrieval_hash
              AND e.model_name=$3 AND e.model_version=$4 AND e.dimensions=$5
              AND e.representation_version=$6
            WHERE s.snapshot_id=$1 AND s.index_version=$2 AND e.input_hash IS NULL
            ORDER BY s.retrieval_hash
        """, sid, index_version, *_space_params(identity))
        submitted = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            vectors = await embed_fn([r["retrieval_text"] for r in batch])
            if len(vectors) != len(batch) or any(
                len(v) != identity.dimensions or not all(math.isfinite(x) for x in v)
                or not any(x != 0 for x in v) for v in vectors
            ):
                raise ValueError("Embedder returned invalid, zero, or wrong-dimension vectors")
            await conn.executemany("""
                INSERT INTO interaction_embeddings
                    (input_hash,model_name,model_version,dimensions,representation_version,embedding)
                VALUES ($1,$2,$3,$4,$5,$6::vector) ON CONFLICT DO NOTHING
            """, [(row["retrieval_hash"], *_space_params(identity), _vector_literal(vector))
                  for row, vector in zip(batch, vectors)])
            submitted += len(batch)
        return submitted


async def _require_embeddings(conn, sid, iv, identity):
    missing = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM interaction_sites s LEFT JOIN interaction_embeddings e
              ON e.input_hash=s.retrieval_hash AND e.model_name=$3 AND e.model_version=$4
              AND e.dimensions=$5 AND e.representation_version=$6
            WHERE s.snapshot_id=$1 AND s.index_version=$2 AND e.input_hash IS NULL
        )
    """, sid, iv, *_space_params(identity))
    if missing:
        raise ValueError("Site index has missing embeddings for the selected vector space")


def _overlap(a, b):
    return (a["repo_id"] == b["repo_id"] and a["path"] == b["path"]
            and max(a["start_byte"], b["start_byte"]) < min(a["end_byte"], b["end_byte"]))


def _fuse_channels(channels, options):
    by_id, scores, provenance = {}, {}, {}
    for channel, rows in channels.items():
        for rank, row in enumerate(rows, 1):
            key = row["id"]
            by_id[key] = row
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank)
            provenance.setdefault(key, {})[channel] = {"rank": rank, "score": row["score"]}
    def tie(key):
        return (by_id[key]["repo_id"], by_id[key]["path"], by_id[key]["start_byte"],
                by_id[key]["end_byte"], by_id[key]["kind"])
    ordered = sorted(by_id, key=lambda key: (-scores[key], tie(key)))
    eligible, passages = [], []
    for key in ordered:
        row = by_id[key]
        if row["kind"] in {"configuration", "declaration"}:
            # A passage can contain declarations absent from a clipped AST input.
            # Only collapse overlapping passages, never against calls/definitions.
            if any(_overlap(row, other) for other in passages):
                continue
            passages.append(row)
        eligible.append(key)
    exact = sorted((key for key in eligible if "exact" in provenance[key]),
                   key=lambda key: provenance[key]["exact"]["rank"])[:options.exact_reserve]
    selected = (exact + [key for key in eligible if key not in exact])[:options.candidate_limit]
    return [CandidateHit(
        str(key), by_id[key]["repo_id"], by_id[key]["path"], by_id[key]["kind"],
        by_id[key]["start_byte"], by_id[key]["end_byte"], by_id[key]["content_hash"],
        by_id[key]["retrieval_hash"], by_id[key]["retrieval_text"],
        json.loads(by_id[key]["metadata"]), scores[key], provenance[key],
    ) for key in selected]


async def _retrieve(conn, sid, iv, source, identity, options):
    vector = await conn.fetchval("""
        SELECT embedding::text FROM interaction_embeddings WHERE input_hash=$1
          AND model_name=$2 AND model_version=$3 AND dimensions=$4 AND representation_version=$5
    """, source["retrieval_hash"], *_space_params(identity))
    if vector is None:
        raise ValueError("Source call has no compatible embedding")
    semantic = await conn.fetch("""
        SELECT s.*, 1-(e.embedding <=> $4::vector) AS score FROM interaction_sites s
        JOIN interaction_embeddings e ON e.input_hash=s.retrieval_hash
          AND e.model_name=$5 AND e.model_version=$6 AND e.dimensions=$7
          AND e.representation_version=$8
        WHERE s.snapshot_id=$1 AND s.index_version=$2 AND s.repo_id<>$3
        ORDER BY score DESC,s.repo_id,s.path,s.start_byte,s.end_byte,s.kind LIMIT $9
    """, sid, iv, source["repo_id"], vector, *_space_params(identity), options.channel_limit)
    terms = source["terms"][:128]
    exact = await conn.fetch("""
        SELECT s.*, (SELECT count(*)::float FROM unnest(s.terms) t WHERE t=ANY($4::text[])) AS score
        FROM interaction_sites s
        WHERE s.snapshot_id=$1 AND s.index_version=$2 AND s.repo_id<>$3 AND s.terms && $4::text[]
        ORDER BY score DESC,s.repo_id,s.path,s.start_byte,s.end_byte,s.kind LIMIT $5
    """, sid, iv, source["repo_id"], terms, options.channel_limit)
    query = _build_tsquery(" ".join(terms) + " " + source["retrieval_text"][:2048])
    lexical = []
    if query:
        lexical = await conn.fetch("""
            SELECT s.*,ts_rank_cd(s.fts_doc,to_tsquery('simple',$4))::float AS score
            FROM interaction_sites s
            WHERE s.snapshot_id=$1 AND s.index_version=$2 AND s.repo_id<>$3
              AND s.fts_doc @@ to_tsquery('simple',$4)
            ORDER BY score DESC,s.repo_id,s.path,s.start_byte,s.end_byte,s.kind LIMIT $5
        """, sid, iv, source["repo_id"], query, options.channel_limit)
    return _fuse_channels({"semantic": semantic, "lexical": lexical, "exact": exact}, options)


async def retrieve_candidates(
    pool: asyncpg.Pool, snapshot_id: str | UUID, index_version: str,
    source_site_id: str | UUID, identity: EmbeddingIdentity,
    options: RetrievalOptions = RetrievalOptions(),
) -> list[CandidateHit]:
    """Inspect candidates for one call; does not persist a run or create edges."""
    sid = UUID(str(snapshot_id))
    async with pool.acquire() as conn:
        await _ready(conn, sid, index_version, identity)
        await _require_embeddings(conn, sid, index_version, identity)
        source = await conn.fetchrow("""
            SELECT * FROM interaction_sites
            WHERE snapshot_id=$1 AND index_version=$2 AND id=$3 AND kind='call'
        """, sid, index_version, UUID(str(source_site_id)))
        if source is None:
            raise ValueError("Source must be a call site in the selected snapshot index")
        return await _retrieve(conn, sid, index_version, source, identity, options)


async def retrieve_all_candidates(
    pool: asyncpg.Pool, snapshot_id: str | UUID, index_version: str,
    identity: EmbeddingIdentity, options: RetrievalOptions = RetrievalOptions(),
    *, source_site_ids: tuple[str | UUID, ...] | None = None,
) -> RetrievalRun:
    """Persist bounded candidate sets for every call, including empty sets.

    An explicit source subset supports bounded evaluation against the full target
    corpus. Its IDs are recorded in run settings; None retains all-call behavior.
    Rankings are run-scoped. Reranking, evidence assembly, and relationship
    classification consume these records separately.
    """
    sid = UUID(str(snapshot_id))
    async with pool.acquire() as conn, conn.transaction():
        await _ready(conn, sid, index_version, identity)
        await conn.execute("SELECT id FROM connection_snapshots WHERE id=$1 FOR UPDATE", sid)
        await _require_embeddings(conn, sid, index_version, identity)
        sites = await conn.fetch("""
            SELECT * FROM interaction_sites WHERE snapshot_id=$1 AND index_version=$2
            ORDER BY repo_id,path,start_byte,end_byte,kind
        """, sid, index_version)
        corpus = fingerprint([[r["repo_id"], r["path"], r["start_byte"], r["end_byte"],
                               r["kind"], r["content_hash"], r["retrieval_hash"]] for r in sites])
        settings = {"algorithm": RETRIEVAL_VERSION, **asdict(options)}
        selected = None if source_site_ids is None else {UUID(str(value)) for value in source_site_ids}
        if selected is not None:
            available = {row["id"] for row in sites if row["kind"] == "call"}
            if not selected <= available:
                raise ValueError("Selected sources must be call sites in this snapshot index")
            settings["source_site_ids"] = sorted(str(value) for value in selected)
        run_key = fingerprint([str(sid), index_version, corpus, asdict(identity), settings])
        existing = await conn.fetchrow("SELECT * FROM connection_retrieval_runs WHERE run_key=$1", run_key)
        if existing:
            if existing["completed_at"] is None:
                raise ValueError("An incomplete retrieval run already exists")
            counts = await conn.fetchrow("""
                SELECT count(*) AS sources,coalesce(sum(candidate_count),0) AS candidates
                FROM connection_candidate_sets WHERE run_id=$1
            """, existing["id"])
            return RetrievalRun(str(existing["id"]), counts["sources"], counts["candidates"], corpus)
        run_id = await conn.fetchval("""
            INSERT INTO connection_retrieval_runs
                (snapshot_id,index_version,run_key,corpus_fingerprint,embedding_identity,settings)
            VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb) RETURNING id
        """, sid, index_version, run_key, corpus, json.dumps(asdict(identity)), json.dumps(settings))
        source_count = candidate_count = 0
        cache = {}
        for source in sites:
            if source["kind"] != "call" or (selected is not None and source["id"] not in selected):
                continue
            source_count += 1
            # Equal input in the same repo has the same retrieval neighborhood,
            # but each source occurrence retains its own candidates and decisions.
            key = (source["repo_id"], source["retrieval_hash"], tuple(source["terms"]))
            if key not in cache:
                cache[key] = await _retrieve(conn, sid, index_version, source, identity, options)
            hits = cache[key]
            digest = fingerprint({"source": [source["repo_id"], source["path"], source["start_byte"],
                                              source["end_byte"], source["retrieval_hash"]],
                                  "settings": settings, "embedding": asdict(identity),
                                  "candidates": [hit.signature() for hit in hits]})
            await conn.execute("""
                INSERT INTO connection_candidate_sets VALUES ($1,$2,$3,$4,$5)
            """, sid, run_id, source["id"], digest, len(hits))
            for rank, hit in enumerate(hits, 1):
                target_id = UUID(hit.target_site_id)
                await conn.execute("""
                    INSERT INTO connection_candidates
                        (snapshot_id,source_site_id,source_repo_id,target_site_id,target_repo_id)
                    VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING
                """, sid, source["id"], source["repo_id"], target_id, hit.target_repo_id)
                candidate_id = await conn.fetchval("""
                    SELECT id FROM connection_candidates
                    WHERE snapshot_id=$1 AND source_site_id=$2 AND target_site_id=$3
                """, sid, source["id"], target_id)
                await conn.execute("""
                    INSERT INTO connection_candidate_matches
                        (snapshot_id,run_id,source_site_id,candidate_id,rank,fused_score,channels)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb)
                """, sid, run_id, source["id"], candidate_id, rank, hit.fused_score,
                    json.dumps(hit.channels))
                candidate_count += 1
        await conn.execute("UPDATE connection_retrieval_runs SET completed_at=now() WHERE id=$1", run_id)
    return RetrievalRun(str(run_id), source_count, candidate_count, corpus)
