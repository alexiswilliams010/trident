from dataclasses import replace

import pytest

from core.connections.contracts import (
    Candidate, Connection, Decision, Endpoint, EndpointKind as K, Evidence,
    Outcome, RepoRevision, Site, Snapshot, validate_connections,
)
from core.connections.verbs import VERBS, normalize_verb


def fixture():
    snapshot = Snapshot("snapshot", (RepoRevision(1, "commit-a"), RepoRevision(2, "commit-b")), "cfg")
    source = Site("a", snapshot.id, 1, "a.py", 0, 10, "hash-a")
    target = Site("b", snapshot.id, 2, "b.go", 0, 10, "hash-b", "definition")
    candidate = Candidate("candidate", snapshot.id, source, target)
    evidence = Evidence("ev", snapshot.id, 1, "config.yml", 0, 10, "hash-cfg", "support")
    decision = Decision("decision", candidate.id, Outcome.SUPPORTED,
                        "fake", "model-1", "policy-1", "fingerprint", (evidence.id,), 0.9)
    return snapshot, candidate, decision, (evidence,)


def test_single_repository_and_two_branches_cannot_qualify():
    with pytest.raises(ValueError, match="at least two"):
        Snapshot("s", (RepoRevision(1, "a"),), "cfg").require_cross_repo()
    with pytest.raises(ValueError, match="one revision"):
        Snapshot("s", (RepoRevision(1, "a"), RepoRevision(1, "b")), "cfg")


def test_candidate_rejects_same_repo_and_foreign_snapshot():
    snapshot, candidate, _, _ = fixture()
    candidate.validate(snapshot)
    for target in (replace(candidate.target, repo_id=1),
                   replace(candidate.target, repo_id=3),
                   replace(candidate.target, snapshot_id="other")):
        with pytest.raises(ValueError):
            replace(candidate, target=target).validate(snapshot)


def test_direct_connection_and_unsupported_outcomes():
    snapshot, candidate, decision, evidence = fixture()
    edges = (Connection(Endpoint("a", K.SITE), Endpoint("b", K.SITE), "INVOKES", decision.id),)
    validate_connections(snapshot, candidate, decision, edges, evidence)
    for outcome in set(Outcome) - {Outcome.SUPPORTED}:
        with pytest.raises(ValueError, match="supported"):
            validate_connections(snapshot, candidate, replace(decision, outcome=outcome),
                                 edges, evidence)


def test_shared_resource_requires_both_repositories():
    snapshot, candidate, decision, evidence = fixture()
    resource = Endpoint("topic", K.RESOURCE)
    edges = (
        Connection(Endpoint("a", K.SITE), resource, "PRODUCES", decision.id),
        Connection(Endpoint("b", K.SITE), resource, "CONSUMES", decision.id),
    )
    validate_connections(snapshot, candidate, decision, edges, evidence, frozenset({"topic"}))
    with pytest.raises(ValueError, match="both candidate"):
        validate_connections(snapshot, candidate, decision, edges[:1], evidence,
                             frozenset({"topic"}))
    with pytest.raises(ValueError, match="outside"):
        validate_connections(snapshot, candidate, decision, edges, evidence)


def test_connections_require_matching_evidence_and_endpoint_types():
    snapshot, candidate, decision, evidence = fixture()
    edge = Connection(Endpoint("a", K.SITE), Endpoint("b", K.SITE), "INVOKES", decision.id)
    for bad_evidence in ((), (replace(evidence[0], role="context"),),
                         (replace(evidence[0], snapshot_id="other"),)):
        with pytest.raises(ValueError):
            validate_connections(snapshot, candidate, decision, (edge,), bad_evidence)
    for bad_edge in (replace(edge, verb="PRODUCES"), replace(edge, verb="SIMILAR_TO"),
                     replace(edge, target=Endpoint("c", K.SITE)),
                     replace(edge, decision_id="other")):
        with pytest.raises(ValueError):
            validate_connections(snapshot, candidate, decision, (bad_edge,), evidence)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_decision_scores_are_finite_probabilities(score):
    with pytest.raises(ValueError, match="score"):
        replace(fixture()[2], score=score)


def test_vocabulary_contains_only_interactions():
    assert normalize_verb(" emits ") == "PRODUCES"
    assert normalize_verb("consumes") == "CONSUMES"
    assert not {"CONTAINS", "IMPLEMENTS", "IMPORTS", "SIMILAR_TO", "EMITS"} & VERBS.keys()
    with pytest.raises(ValueError):
        normalize_verb("related")
