"""Track decision inputs and assess reuse without rewriting historical graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from uuid import UUID

import asyncpg

from .snapshots import fingerprint, load_snapshot


@dataclass(frozen=True)
class ResolutionVersions:
    provider: str
    model_version: str
    policy_version: str
    vocabulary_version: int
    representation_version: str
    retrieval_version: str
    reranker_version: str
    candidate_set_fingerprint: str

    def __post_init__(self):
        if self.vocabulary_version < 1 or any(
            not value.strip() for value in asdict(self).values() if isinstance(value, str)
        ):
            raise ValueError("Resolution versions and candidate-set identity must be explicit")


@dataclass(frozen=True)
class ReuseAssessment:
    reusable: bool
    reasons: tuple[str, ...]


async def _decision_state(conn, decision_id):
    decision = await conn.fetchrow("SELECT * FROM connection_decisions WHERE id=$1", decision_id)
    if decision is None:
        raise ValueError("Unknown decision")
    sites = await conn.fetch("""
        SELECT s.repo_id, s.path, s.start_byte, s.end_byte, s.content_hash, s.kind,
               CASE WHEN s.id=c.source_site_id THEN 'source' ELSE 'target' END AS endpoint_role
        FROM connection_candidates c JOIN interaction_sites s
          ON s.id IN (c.source_site_id, c.target_site_id)
        WHERE c.id=$1 ORDER BY s.repo_id, s.path, s.start_byte
    """, decision["candidate_id"])
    evidence = await conn.fetch("""
        SELECT e.repo_id, e.path, e.start_byte, e.end_byte, e.content_hash, e.excerpt, de.role
        FROM connection_decision_evidence de JOIN connection_evidence e ON e.id=de.evidence_id
        WHERE de.decision_id=$1 ORDER BY e.repo_id, e.path, e.start_byte, e.end_byte, de.role, e.id
    """, decision_id)
    payload = {"outcome": decision["outcome"], "provider": decision["provider"],
               "model": decision["model_version"], "policy": decision["policy_version"],
               "evidence_fingerprint": decision["evidence_fingerprint"],
               "score": decision["score"],
               "conditions": json.loads(decision["conditions"]),
               "sites": [dict(row) for row in sites], "evidence": [dict(row) for row in evidence]}
    paths = {(row["repo_id"], row["path"]) for row in (*sites, *evidence)}
    return decision, fingerprint(payload), paths


async def _file_states(conn, snapshot, paths):
    repos = {m.repo_id for m in snapshot.members}
    states = []
    for repo_id, path in sorted(paths):
        if repo_id not in repos:
            raise ValueError("Dependency repository is outside the snapshot")
        row = await conn.fetchrow("""
            SELECT content_hash, mode FROM connection_snapshot_files
            WHERE snapshot_id=$1 AND repo_id=$2 AND path=$3
        """, UUID(snapshot.id), repo_id, path)
        states.append((repo_id, path, row["content_hash"] if row else None,
                       row["mode"] if row else None))
    return states


def _input_fingerprint(snapshot, versions, states):
    # Exact revisions are deliberately excluded: unchanged dependencies can be
    # reused across commits. Repository membership and config context still matter.
    return fingerprint({"repos": sorted(m.repo_id for m in snapshot.members),
                        "context": snapshot.context_fingerprint,
                        "versions": asdict(versions), "dependencies": states})


async def record_decision_inputs(
    conn: asyncpg.Connection, decision_id: str | UUID, versions: ResolutionVersions,
    *, additional_paths: tuple[tuple[int, str], ...] = (),
) -> str:
    """Record caller, target, all evidence, and inspected/missing configuration files.

    Candidate-set identity must describe the actual reranked inputs, including
    alternatives and conditions. File dependencies cannot detect newly discovered
    alternatives by themselves; the retrieval stage must refresh that identity.
    """
    did = UUID(str(decision_id))
    async with conn.transaction():
        await conn.execute("SELECT id FROM connection_decisions WHERE id=$1 FOR UPDATE", did)
        decision, signature, paths = await _decision_state(conn, did)
        snapshot = await load_snapshot(conn, decision["snapshot_id"])
        if (versions.provider, versions.model_version, versions.policy_version) != (
            decision["provider"], decision["model_version"], decision["policy_version"]
        ):
            raise ValueError("Resolution versions disagree with recorded decision provenance")
        states = await _file_states(conn, snapshot, paths | set(additional_paths))
        digest = _input_fingerprint(snapshot, versions, states)
        existing = await conn.fetchrow("SELECT * FROM connection_decision_inputs WHERE decision_id=$1", did)
        if existing:
            if (existing["input_fingerprint"], existing["decision_fingerprint"]) != (digest, signature):
                raise ValueError("Decision inputs are already recorded; create a new decision")
            return digest
        await conn.execute("""
            INSERT INTO connection_decision_inputs
                (decision_id, input_fingerprint, decision_fingerprint, versions)
            VALUES ($1,$2,$3,$4::jsonb)
        """, did, digest, signature, json.dumps(asdict(versions)))
        await conn.executemany("""
            INSERT INTO connection_decision_dependencies
                (decision_id, repo_id, path, content_hash, mode) VALUES ($1,$2,$3,$4,$5)
        """, [(did, *state) for state in states])
        return digest


async def assess_decision_reuse(
    conn: asyncpg.Connection, decision_id: str | UUID, target_snapshot_id: str | UUID,
    versions: ResolutionVersions,
) -> ReuseAssessment:
    """Return invalidation reasons; never copy snapshot-bound IDs or mutate old edges."""
    did = UUID(str(decision_id))
    inputs = await conn.fetchrow("SELECT * FROM connection_decision_inputs WHERE decision_id=$1", did)
    if inputs is None:
        return ReuseAssessment(False, ("untracked_inputs",))
    decision, signature, _ = await _decision_state(conn, did)
    source = await load_snapshot(conn, decision["snapshot_id"])
    target = await load_snapshot(conn, target_snapshot_id)
    reasons = []
    if signature != inputs["decision_fingerprint"]:
        reasons.append("decision_or_evidence_changed")
    if asdict(versions) != json.loads(inputs["versions"]):
        reasons.append("resolution_versions_or_candidates_changed")
    if {m.repo_id for m in source.members} != {m.repo_id for m in target.members}:
        reasons.append("repository_membership_changed")
    if source.context_fingerprint != target.context_fingerprint:
        reasons.append("configuration_context_changed")
    old_states = await conn.fetch("""
        SELECT repo_id, path, content_hash, mode FROM connection_decision_dependencies
        WHERE decision_id=$1 ORDER BY repo_id, path
    """, did)
    if "repository_membership_changed" not in reasons:
        states = await _file_states(conn, target, {(r["repo_id"], r["path"]) for r in old_states})
        if states != [tuple(row) for row in old_states]:
            reasons.append("source_target_or_evidence_changed")
        if not reasons and _input_fingerprint(target, versions, states) != inputs["input_fingerprint"]:
            reasons.append("dependency_record_changed")
    return ReuseAssessment(not reasons, tuple(reasons))
