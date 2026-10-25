"""Rerank retrieved alternatives and assemble bounded, source-cited context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import asyncio
import json
import math
from typing import Awaitable, Callable
from uuid import UUID, uuid5

from .snapshots import fingerprint, load_snapshot

PREPARATION_VERSION = "evidence-packet-v1"
# Scores align with documents in input order, not a provider-specific sorted list.
RerankFn = Callable[[str, list[str]], Awaitable[list[float]]]


@dataclass(frozen=True)
class RerankerIdentity:
    provider: str
    model_version: str
    input_version: str = "site-pair-v1"

    def __post_init__(self):
        if any(not value.strip() for value in asdict(self).values()):
            raise ValueError("Explicit reranker provider, model and input versions are required")


@dataclass(frozen=True)
class PreparationOptions:
    target_limit: int = 5
    rerank_chars: int = 4000
    excerpt_bytes: int = 3000
    context_bytes: int = 500
    total_bytes: int = 24000
    max_excerpts: int = 12
    timeout_seconds: float = 60

    def __post_init__(self):
        if not 1 <= self.target_limit <= 100:
            raise ValueError("Target limit must be between 1 and 100")
        if self.rerank_chars < 100 or self.excerpt_bytes < 100 or self.context_bytes < 0:
            raise ValueError("Invalid reranking or excerpt budgets")
        if self.max_excerpts < self.target_limit + 1:
            raise ValueError("Reserve an excerpt for the source and every selected target")
        if self.total_bytes < (self.target_limit + 1) * self.excerpt_bytes:
            raise ValueError("Total budget must reserve full endpoint excerpt budgets")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Reranker timeout must be positive and finite")


@dataclass(frozen=True)
class PreparationRun:
    id: str
    packet_count: int


def source_window(raw: bytes, start: int, end: int, budget: int, context: int) -> dict:
    """A UTF-8 aligned byte window prioritizing the endpoint over nearby context."""
    raw.decode("utf-8")
    if not 0 <= start <= end <= len(raw) or budget < 4:
        raise ValueError("Invalid source range or excerpt budget")
    # Site offsets are valid boundaries; cap large endpoints before adding context.
    left = max(0, start - min(context, max(0, budget - (end - start)) // 2))
    right = min(len(raw), left + budget, end + context)
    while left < len(raw) and raw[left] & 0xC0 == 0x80:
        left += 1
    while right < len(raw) and right > left and raw[right] & 0xC0 == 0x80:
        right -= 1
    return {
        "start_byte": left,
        "end_byte": right,
        "excerpt": raw[left:right].decode(),
        "site_clipped": left > start or right < end,
        "context_clipped": left > max(0, start - context) or right < min(len(raw), end + context),
    }


async def prepare_candidates(
    pool,
    retrieval_run_id: str | UUID,
    rerank_fn: RerankFn,
    identity: RerankerIdentity,
    options: PreparationOptions = PreparationOptions(),
    *,
    additional_paths: tuple[tuple[int, str], ...] = (),
) -> PreparationRun:
    """Prepare every retrieved source, preserving scores and unselected alternatives.

    Additional paths name captured configuration/context, never live filesystem
    paths. Missing, binary and budget-omitted inputs remain explicit dependencies.
    No score threshold constitutes a relationship decision.
    """
    rid = UUID(str(retrieval_run_id))
    paths = sorted(set(additional_paths))
    if len(paths) > 32:
        raise ValueError("At most 32 additional context paths may be requested")
    settings = {"version": PREPARATION_VERSION, **asdict(options), "additional_paths": paths}
    async with pool.acquire() as conn, conn.transaction():
        retrieval = await conn.fetchrow("SELECT * FROM connection_retrieval_runs WHERE id=$1", rid)
        if retrieval is None or retrieval["completed_at"] is None:
            raise ValueError("A completed retrieval run is required")
        sid = retrieval["snapshot_id"]
        snapshot = await load_snapshot(conn, sid)
        if any(repo not in {m.repo_id for m in snapshot.members} for repo, _ in paths):
            raise ValueError("Additional context is outside the snapshot")
        # Serialize publication and reuse. A failed scorer rolls back all packets.
        await conn.execute("SELECT id FROM connection_snapshots WHERE id=$1 FOR UPDATE", sid)
        key = fingerprint([str(rid), asdict(identity), settings])
        existing = await conn.fetchrow(
            "SELECT * FROM connection_preparation_runs WHERE run_key=$1", key
        )
        if existing:
            if existing["completed_at"] is None:
                raise ValueError("Incomplete preparation run")
            count = await conn.fetchval(
                "SELECT count(*) FROM connection_evidence_packets WHERE preparation_run_id=$1",
                existing["id"],
            )
            return PreparationRun(str(existing["id"]), count)
        run = await conn.fetchval(
            """
            INSERT INTO connection_preparation_runs
                (snapshot_id,retrieval_run_id,run_key,reranker_identity,settings)
            VALUES ($1,$2,$3,$4::jsonb,$5::jsonb) RETURNING id
        """,
            sid,
            rid,
            key,
            json.dumps(asdict(identity)),
            json.dumps(settings),
        )
        sources = await conn.fetch(
            """
            SELECT s.*, cs.candidate_set_fingerprint FROM connection_candidate_sets cs
            JOIN interaction_sites s ON s.id=cs.source_site_id
            WHERE cs.run_id=$1 ORDER BY s.repo_id,s.path,s.start_byte,s.id
        """,
            rid,
        )
        for source in sources:
            targets = await conn.fetch(
                """
                SELECT s.*, c.id AS candidate_id, m.rank AS retrieval_rank, m.channels
                FROM connection_candidate_matches m
                JOIN connection_candidates c ON c.id=m.candidate_id
                JOIN interaction_sites s ON s.id=c.target_site_id
                WHERE m.run_id=$1 AND m.source_site_id=$2 ORDER BY m.rank
            """,
                rid,
                source["id"],
            )
            scores = []
            if targets:
                scores = await asyncio.wait_for(
                    rerank_fn(
                        source["retrieval_text"][: options.rerank_chars],
                        [t["retrieval_text"][: options.rerank_chars] for t in targets],
                    ),
                    timeout=options.timeout_seconds,
                )

            if (
                not isinstance(scores, (list, tuple))
                or len(scores) != len(targets)
                or any(
                    isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s)
                    for s in scores
                )
            ):
                raise ValueError("Reranker must return one finite score per input document")
            ordered = sorted(
                zip(targets, scores), key=lambda pair: (-pair[1], pair[0]["retrieval_rank"])
            )
            selected = ordered[: options.target_limit]
            payload = {
                "version": PREPARATION_VERSION,
                "source_site_id": str(source["id"]),
                "candidate_set_fingerprint": source["candidate_set_fingerprint"],
                "reranker": asdict(identity),
                "settings": settings,
                "alternatives": [],
                "excerpts": [],
                "dependencies": [],
                "omissions": [],
                "empty_candidates": not targets,
                "query_clipped": len(source["retrieval_text"]) > options.rerank_chars,
            }
            for rank, (target, score) in enumerate(ordered, 1):
                payload["alternatives"].append(
                    {
                        "candidate_id": str(target["candidate_id"]),
                        "target_site_id": str(target["id"]),
                        "rank": rank,
                        "score": score,
                        "retrieval_rank": target["retrieval_rank"],
                        "channels": json.loads(target["channels"]),
                        "selected": rank <= options.target_limit,
                        "input_clipped": len(target["retrieval_text"]) > options.rerank_chars,
                    }
                )
            # Endpoint excerpts consume the reserved budget before optional context.
            remaining = options.total_bytes
            requests = [(source["repo_id"], source["path"], source, "source")]
            requests += [(t["repo_id"], t["path"], t, "target") for t, _ in selected]
            requests += [(repo, path, None, "additional") for repo, path in paths]
            for repo, path, site, role in requests:
                file = await conn.fetchrow(
                    """
                    SELECT f.content_hash,f.mode,b.content FROM connection_snapshot_files f
                    JOIN connection_source_blobs b USING(content_hash)
                    WHERE f.snapshot_id=$1 AND f.repo_id=$2 AND f.path=$3
                """,
                    sid,
                    repo,
                    path,
                )
                dependency = {
                    "repo_id": repo,
                    "path": path,
                    "content_hash": file["content_hash"] if file else None,
                    "mode": file["mode"] if file else None,
                }
                if dependency not in payload["dependencies"]:
                    payload["dependencies"].append(dependency)
                reason = None
                if file is None:
                    reason = "missing"
                elif file["mode"] not in {"100644", "100755"}:
                    reason = "symlink_or_gitlink"
                elif len(payload["excerpts"]) >= options.max_excerpts or remaining < 4:
                    reason = "budget"
                else:
                    raw = bytes(file["content"])
                    try:
                        raw.decode("utf-8")
                        if b"\0" in raw:
                            reason = "binary"
                    except UnicodeDecodeError:
                        reason = "non_utf8"
                if reason:
                    payload["omissions"].append({**dependency, "reason": reason, "role": role})
                    continue
                start, end = (site["start_byte"], site["end_byte"]) if site else (0, len(raw))
                window = source_window(
                    raw, start, end, min(options.excerpt_bytes, remaining), options.context_bytes
                )
                eid = uuid5(
                    sid,
                    fingerprint(
                        [repo, path, file["content_hash"], window["start_byte"], window["end_byte"]]
                    ),
                )
                await conn.execute(
                    """
                    INSERT INTO connection_evidence
                        (id,snapshot_id,repo_id,path,start_byte,end_byte,content_hash,excerpt)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(id) DO NOTHING
                """,
                    eid,
                    sid,
                    repo,
                    path,
                    window["start_byte"],
                    window["end_byte"],
                    file["content_hash"],
                    window["excerpt"],
                )
                payload["excerpts"].append(
                    {
                        **dependency,
                        **window,
                        "evidence_id": str(eid),
                        "site_id": str(site["id"]) if site else None,
                        "role": role,
                    }
                )
                remaining -= window["end_byte"] - window["start_byte"]
            payload["excerpt_bytes"] = options.total_bytes - remaining
            digest = fingerprint(payload)
            packet = await conn.fetchval(
                """
                INSERT INTO connection_evidence_packets
                    (snapshot_id,preparation_run_id,source_site_id,packet_fingerprint,payload)
                VALUES ($1,$2,$3,$4,$5::jsonb) RETURNING id
            """,
                sid,
                run,
                source["id"],
                digest,
                json.dumps(payload),
            )
            await conn.executemany(
                """
                INSERT INTO connection_packet_evidence VALUES ($1,$2,$3) ON CONFLICT DO NOTHING
            """,
                [(sid, packet, UUID(e["evidence_id"])) for e in payload["excerpts"]],
            )
        await conn.execute(
            "UPDATE connection_preparation_runs SET completed_at=now() WHERE id=$1", run
        )
        return PreparationRun(str(run), len(sources))
