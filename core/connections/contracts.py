"""Contracts shared by retrieval, decision, and graph stages.

Repository IDs are canonical repos.id values, never branch IDs or display names.
Candidates pair code sites in different repositories. A supported candidate may
yield a direct edge or two resource edges; resources never establish eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class Outcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS = "ambiguous"


class EndpointKind(StrEnum):
    SITE = "site"
    RESOURCE = "resource"


@dataclass(frozen=True)
class RepoRevision:
    repo_id: int
    revision: str

    def __post_init__(self) -> None:
        if self.repo_id <= 0 or not self.revision.strip():
            raise ValueError("A positive canonical repo ID and exact revision are required")


@dataclass(frozen=True)
class Snapshot:
    id: str
    members: tuple[RepoRevision, ...]
    context_fingerprint: str

    def __post_init__(self) -> None:
        ids = [member.repo_id for member in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("Select one revision per repository in a snapshot")
        if not self.id or not self.context_fingerprint:
            raise ValueError("Snapshot identity and configuration fingerprint are required")

    def require_cross_repo(self) -> None:
        if len({member.repo_id for member in self.members}) < 2:
            raise ValueError("Cross-connections require at least two distinct repositories")


@dataclass(frozen=True)
class Site:
    id: str
    snapshot_id: str
    repo_id: int
    path: str
    start_byte: int
    end_byte: int
    content_hash: str
    kind: str = "call"

    def __post_init__(self) -> None:
        if not all((self.id, self.snapshot_id, self.path, self.content_hash)):
            raise ValueError("Sites require identity, source path, and content hash")
        if self.repo_id <= 0 or self.start_byte < 0 or self.end_byte < self.start_byte:
            raise ValueError("Invalid site repository or byte range")
        if self.kind not in {"call", "definition", "declaration", "configuration"}:
            raise ValueError("Unknown site kind")


@dataclass(frozen=True)
class Candidate:
    id: str
    snapshot_id: str
    source: Site
    target: Site

    def validate(self, snapshot: Snapshot) -> None:
        snapshot.require_cross_repo()
        if not self.id or self.snapshot_id != snapshot.id:
            raise ValueError("Candidate belongs to a different snapshot")
        repos = {member.repo_id for member in snapshot.members}
        for site in (self.source, self.target):
            if site.snapshot_id != snapshot.id or site.repo_id not in repos:
                raise ValueError("Candidate endpoint is outside the snapshot")
        if self.source.repo_id == self.target.repo_id:
            raise ValueError("Candidate endpoints must belong to different repositories")


@dataclass(frozen=True)
class Evidence:
    id: str
    snapshot_id: str
    repo_id: int
    path: str
    start_byte: int
    end_byte: int
    content_hash: str
    role: str

    def validate(self, snapshot: Snapshot) -> None:
        if self.snapshot_id != snapshot.id or self.repo_id not in {
            member.repo_id for member in snapshot.members
        }:
            raise ValueError("Evidence is outside the snapshot")
        if not all((self.id, self.path, self.content_hash)):
            raise ValueError("Evidence requires a source identity")
        if self.start_byte < 0 or self.end_byte < self.start_byte:
            raise ValueError("Invalid evidence byte range")
        if self.role not in {"support", "contradiction", "context"}:
            raise ValueError("Unknown evidence role")


@dataclass(frozen=True)
class Decision:
    id: str
    candidate_id: str
    outcome: Outcome
    provider: str
    model_version: str
    policy_version: str
    evidence_fingerprint: str
    evidence_ids: tuple[str, ...]
    score: float | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.id,
                self.candidate_id,
                self.provider,
                self.model_version,
                self.policy_version,
                self.evidence_fingerprint,
            )
        ):
            raise ValueError("Decision provenance must be explicit")
        if self.outcome not in set(Outcome):
            raise ValueError("Unknown decision outcome")
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("Decision score must be finite and between zero and one")
        if self.outcome == Outcome.SUPPORTED and not self.evidence_ids:
            raise ValueError("Supported decisions require evidence")


@dataclass(frozen=True)
class Endpoint:
    id: str
    kind: EndpointKind


@dataclass(frozen=True)
class Connection:
    """One directed edge. Shared resources require two edges backed by a candidate."""

    source: Endpoint
    target: Endpoint
    verb: str
    decision_id: str


def validate_connections(
    snapshot: Snapshot,
    candidate: Candidate,
    decision: Decision,
    edges: tuple[Connection, ...],
    evidence: tuple[Evidence, ...],
    resource_ids: frozenset[str] = frozenset(),
) -> None:
    """Validate a complete publication unit, including shared-resource participation.

    resource_ids must be obtained from this snapshot's stored resources. No model
    output can create evidence of cross-repository participation by naming a resource.
    """
    from .verbs import VERBS

    candidate.validate(snapshot)
    if decision.candidate_id != candidate.id or decision.outcome != Outcome.SUPPORTED:
        raise ValueError("Only supported decisions for this candidate can produce connections")
    if not edges:
        raise ValueError("A supported publication requires at least one edge")
    for item in evidence:
        item.validate(snapshot)
    evidence_by_id = {item.id: item for item in evidence}
    if any(eid not in evidence_by_id for eid in decision.evidence_ids):
        raise ValueError("Decision references missing evidence")
    if not any(evidence_by_id[eid].role == "support" for eid in decision.evidence_ids):
        raise ValueError("Connections require supporting evidence")
    sites = {candidate.source.id, candidate.target.id}
    participation: dict[str, set[str]] = {}
    for edge in edges:
        if edge.decision_id != decision.id or edge.source == edge.target:
            raise ValueError("Invalid edge decision or self-edge")
        spec = VERBS.get(edge.verb)
        if spec is None or (edge.source.kind, edge.target.kind) not in spec.endpoint_pairs:
            raise ValueError("Unknown verb or incompatible endpoint kinds")
        for endpoint in (edge.source, edge.target):
            allowed = sites if endpoint.kind == EndpointKind.SITE else resource_ids
            if endpoint.id not in allowed:
                raise ValueError("Endpoint is outside the candidate or snapshot resources")
        if edge.target.kind == EndpointKind.RESOURCE:
            participation.setdefault(edge.target.id, set()).add(edge.source.id)
    if any(participants != sites for participants in participation.values()):
        raise ValueError("Each shared resource must connect both candidate repositories")
