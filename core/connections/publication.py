"""Atomic, immutable graph publication from explicitly selected assessed decisions."""

from dataclasses import asdict
import json
from uuid import UUID, uuid5

from .contracts import (
    Candidate,
    Connection,
    Decision,
    Endpoint,
    EndpointKind,
    Evidence,
    Site,
    validate_connections,
)
from .dependencies import ResolutionVersions, assess_decision_reuse
from .snapshots import fingerprint, load_snapshot
from .verbs import VOCABULARY_VERSION


def _site(row):
    fields = {}
    for key in Site.__dataclass_fields__:
        value = row[key]
        if key in {"id", "snapshot_id"}:
            value = str(value)
        fields[key] = value
    return Site(**fields)


async def publish_graph(pool, snapshot_id, decision_ids, *, branch_ids=()):
    """Replace active membership with exactly these decisions, preserving history.

    Optional parser branches contribute captured links with matching endpoint bytes.
    Their resolution context is identified as branch-derived, not commit-certified.
    """
    sid = UUID(str(snapshot_id))
    ids = sorted({UUID(str(did)) for did in decision_ids})
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read"):
        snapshot = await load_snapshot(conn, sid)
        await conn.execute("SELECT id FROM connection_snapshots WHERE id=$1 FOR UPDATE", sid)
        graph = {
            "version": "connection-graph-v1",
            "snapshot_id": str(sid),
            "context_fingerprint": snapshot.context_fingerprint,
            "members": [asdict(m) for m in snapshot.members],
            "decisions": [],
            "nodes": {},
            "edges": [],
            "parser_branches": [],
        }
        candidate_ids = set()
        index_versions = set()
        for did in ids:
            decision = await conn.fetchrow(
                "SELECT * FROM connection_decisions WHERE snapshot_id=$1 AND id=$2", sid, did
            )
            if decision is None or decision["outcome"] != "supported":
                raise ValueError("Publication requires supported decisions from this snapshot")
            if decision["candidate_id"] in candidate_ids:
                raise ValueError("Select only one decision per candidate in a revision")
            candidate_ids.add(decision["candidate_id"])
            tracked = await conn.fetchrow(
                "SELECT * FROM connection_decision_inputs WHERE decision_id=$1", did
            )
            assessment = await conn.fetchrow(
                "SELECT * FROM connection_assessments WHERE decision_id=$1", did
            )
            if tracked is None or assessment is None:
                raise ValueError("Publication requires a tracked assessment")
            versions = ResolutionVersions(**json.loads(tracked["versions"]))
            reuse = await assess_decision_reuse(conn, did, sid, versions)
            if not reuse.reusable:
                raise ValueError(f"Decision inputs changed: {', '.join(reuse.reasons)}")
            proposal = json.loads(assessment["proposal"])
            if (
                proposal["outcome"] != "supported"
                or proposal["vocabulary_version"] != VOCABULARY_VERSION
            ):
                raise ValueError("Proposal outcome or vocabulary is incompatible")
            candidate_row = await conn.fetchrow(
                "SELECT * FROM connection_candidates WHERE id=$1", decision["candidate_id"]
            )
            sites = await conn.fetch(
                "SELECT * FROM interaction_sites WHERE id=ANY($1::uuid[])",
                [candidate_row["source_site_id"], candidate_row["target_site_id"]],
            )
            index_versions.update(s["index_version"] for s in sites)
            if len(index_versions) > 1:
                raise ValueError("Select decisions from one site index version per revision")
            by_id = {str(s["id"]): _site(s) for s in sites}
            candidate = Candidate(
                str(candidate_row["id"]),
                str(sid),
                by_id[str(candidate_row["source_site_id"])],
                by_id[str(candidate_row["target_site_id"])],
            )
            evidence_rows = await conn.fetch(
                """
                SELECT e.*,de.role FROM connection_decision_evidence de
                JOIN connection_evidence e ON e.id=de.evidence_id
                WHERE de.decision_id=$1 ORDER BY e.id,de.role
            """,
                did,
            )
            evidence = tuple(
                Evidence(
                    str(e["id"]),
                    str(sid),
                    e["repo_id"],
                    e["path"],
                    e["start_byte"],
                    e["end_byte"],
                    e["content_hash"],
                    e["role"],
                )
                for e in evidence_rows
            )
            evidence_by_id = {str(e["id"]): e for e in evidence_rows}
            contract = Decision(
                str(did),
                candidate.id,
                decision["outcome"],
                decision["provider"],
                decision["model_version"],
                decision["policy_version"],
                decision["evidence_fingerprint"],
                tuple(e.id for e in evidence),
                decision["score"],
            )
            edges = []
            resources = {}
            for item in proposal["edges"]:
                source = Endpoint(item["source_site_id"], EndpointKind.SITE)
                if "target_site_id" in item:
                    target = Endpoint(item["target_site_id"], EndpointKind.SITE)
                else:
                    anchor = evidence_by_id.get(item.get("resource_evidence_id"))
                    if anchor is None or anchor["role"] != "support":
                        raise ValueError("Shared resource anchor must be supporting evidence")
                    # Resource identity is limited to this candidate pair and its evidence.
                    # A matching name elsewhere cannot merge unrelated resources.
                    identity = fingerprint(
                        [
                            "evidence-scoped-resource-v1",
                            snapshot.context_fingerprint,
                            sorted(by_id),
                            anchor["repo_id"],
                            anchor["path"],
                            anchor["content_hash"],
                            anchor["start_byte"],
                            anchor["end_byte"],
                        ]
                    )
                    resource_id = uuid5(sid, identity)
                    resources[str(resource_id)] = {
                        "id": str(resource_id),
                        "kind": "resource",
                        "identity_key": identity,
                        "identity_scope": "candidate_pair_and_evidence",
                        "anchor": {
                            k: str(anchor[k]) if k == "id" else anchor[k]
                            for k in (
                                "id",
                                "repo_id",
                                "path",
                                "content_hash",
                                "start_byte",
                                "end_byte",
                                "excerpt",
                            )
                        },
                    }
                    target = Endpoint(str(resource_id), EndpointKind.RESOURCE)
                edges.append(Connection(source, target, item["verb"], str(did)))
            validate_connections(
                snapshot, candidate, contract, tuple(edges), evidence, frozenset(resources)
            )
            for resource_id, resource in resources.items():
                await conn.execute(
                    """INSERT INTO connection_resources(id,snapshot_id,identity_key,kind,properties)
                    VALUES ($1,$2,$3,'evidence_scoped',$4::jsonb) ON CONFLICT(snapshot_id,identity_key) DO NOTHING""",
                    UUID(resource_id),
                    sid,
                    resource["identity_key"],
                    json.dumps(resource),
                )
                graph["nodes"]["resource:" + resource_id] = resource
            for site_id, site in by_id.items():
                graph["nodes"]["site:" + site_id] = asdict(site)
            for edge in edges:
                target_site = (
                    UUID(edge.target.id) if edge.target.kind == EndpointKind.SITE else None
                )
                target_resource = (
                    UUID(edge.target.id) if edge.target.kind == EndpointKind.RESOURCE else None
                )
                await conn.execute(
                    """INSERT INTO cross_connections
                    (snapshot_id,decision_id,source_site_id,target_site_id,target_resource_id,verb_version,verb)
                    VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING""",
                    sid,
                    did,
                    UUID(edge.source.id),
                    target_site,
                    target_resource,
                    VOCABULARY_VERSION,
                    edge.verb,
                )
                graph["edges"].append(
                    {
                        "source": "site:" + edge.source.id,
                        "target": edge.target.kind.value + ":" + edge.target.id,
                        "verb": edge.verb,
                        "origin": "cross_repository",
                        "decision_id": str(did),
                        "vocabulary_version": VOCABULARY_VERSION,
                    }
                )
            graph["decisions"].append(
                {
                    "id": str(did),
                    "candidate_id": candidate.id,
                    "provider": decision["provider"],
                    "model_version": decision["model_version"],
                    "policy_version": decision["policy_version"],
                    "conditions": json.loads(decision["conditions"]),
                    "input_fingerprint": tracked["input_fingerprint"],
                    "evidence": [
                        {k: str(v) if isinstance(v, UUID) else v for k, v in row.items()}
                        for row in evidence_rows
                    ],
                }
            )
        if branch_ids:
            from .published_queries import capture_parser_links

            await capture_parser_links(conn, sid, graph, branch_ids)
        # Canonical ordering makes identical publications reuse the same revision.
        graph["edges"].sort(key=lambda edge: json.dumps(edge, sort_keys=True))
        digest = fingerprint(graph)
        revision = await conn.fetchval(
            """INSERT INTO connection_graph_revisions(snapshot_id,fingerprint,graph)
            VALUES ($1,$2,$3::jsonb) ON CONFLICT(snapshot_id,fingerprint) DO NOTHING RETURNING id""",
            sid,
            digest,
            json.dumps(graph),
        )
        if revision is None:
            revision = await conn.fetchval(
                "SELECT id FROM connection_graph_revisions WHERE snapshot_id=$1 AND fingerprint=$2",
                sid,
                digest,
            )
        await conn.execute(
            """INSERT INTO connection_active_graphs VALUES ($1,$2)
            ON CONFLICT(snapshot_id) DO UPDATE SET revision_id=excluded.revision_id""",
            sid,
            revision,
        )
        return str(revision)
