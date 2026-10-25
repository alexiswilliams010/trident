"""Read immutable graph revisions and capture byte-matched parser call links."""

import json
from uuid import UUID

from .snapshots import fingerprint


async def capture_parser_links(conn, sid, graph, branch_ids):
    members = {m["repo_id"] for m in graph["members"]}
    coordinate_ids = {}
    for key, site in sorted(graph["nodes"].items()):
        if key.startswith("site:"):
            coordinate = (
                site["repo_id"],
                site["path"],
                site["content_hash"],
                site["start_byte"],
                site["end_byte"],
                site["kind"],
            )
            coordinate_ids.setdefault(coordinate, key)
    seen_repos = set()
    for bid in sorted(set(branch_ids)):
        branch = await conn.fetchrow("SELECT * FROM branches WHERE id=$1", bid)
        if branch is None or branch["repo_id"] not in members or branch["repo_id"] in seen_repos:
            raise ValueError("Choose at most one parser branch per snapshot repository")
        seen_repos.add(branch["repo_id"])
        mismatch = await conn.fetchval(
            """SELECT EXISTS (
            SELECT 1 FROM branch_files bf JOIN file_versions fv ON fv.id=bf.file_version_id
            LEFT JOIN connection_snapshot_files sf ON sf.snapshot_id=$1 AND sf.repo_id=$3 AND sf.path=bf.path
            WHERE bf.branch_id=$2 AND NOT bf.from_dependency AND
              (sf.content_hash IS NULL OR sf.content_hash<>fv.content_hash OR fv.repo_id<>$3))""",
            sid,
            bid,
            branch["repo_id"],
        )
        if mismatch:
            raise ValueError("Parser branch source differs from the captured snapshot")
        rows = await conn.fetch(
            """
            SELECT ce.id,ce.confidence,ce.caller_def_id,
                sn.start_byte AS source_start,sn.end_byte AS source_end,sf.content_hash AS source_hash,
                sb.path AS source_path,cn.start_byte AS caller_start,cn.end_byte AS caller_end,
                cf.content_hash AS caller_hash,cb.path AS caller_path,
                tn.start_byte AS target_start,tn.end_byte AS target_end,tf.content_hash AS target_hash,
                tb.path AS target_path,cd.name AS caller_name,td.name AS target_name
            FROM call_edges ce JOIN nodes sn ON sn.id=ce.callsite_node_id
            JOIN file_versions sf ON sf.id=sn.file_version_id
            JOIN branch_files sb ON sb.file_version_id=sf.id AND sb.branch_id=ce.branch_id AND NOT sb.from_dependency
            JOIN definitions td ON td.id=ce.callee_def_id JOIN nodes tn ON tn.id=td.node_id
            JOIN file_versions tf ON tf.id=tn.file_version_id
            JOIN branch_files tb ON tb.file_version_id=tf.id AND tb.branch_id=ce.branch_id AND NOT tb.from_dependency
            LEFT JOIN definitions cd ON cd.id=ce.caller_def_id LEFT JOIN nodes cn ON cn.id=cd.node_id
            LEFT JOIN file_versions cf ON cf.id=cn.file_version_id
            LEFT JOIN branch_files cb ON cb.file_version_id=cf.id AND cb.branch_id=ce.branch_id AND NOT cb.from_dependency
            WHERE ce.branch_id=$1 ORDER BY ce.id,sb.path,tb.path,cb.path
        """,
            bid,
        )
        graph["parser_branches"].append(
            {
                "branch_id": bid,
                "repo_id": branch["repo_id"],
                "name": branch["name"],
                "scope": "byte_matched_branch_resolution",
                "resolved_calls": len(rows),
            }
        )
        for row in rows:

            def node(prefix, kind):
                coordinate = (
                    branch["repo_id"],
                    row[prefix + "_path"],
                    row[prefix + "_hash"],
                    row[prefix + "_start"],
                    row[prefix + "_end"],
                    kind,
                )
                key = coordinate_ids.get(coordinate, "parser:" + fingerprint(coordinate))
                coordinate_ids[coordinate] = key
                graph["nodes"].setdefault(
                    key,
                    {
                        "repo_id": coordinate[0],
                        "path": coordinate[1],
                        "content_hash": coordinate[2],
                        "start_byte": coordinate[3],
                        "end_byte": coordinate[4],
                        "kind": kind,
                        "name": row.get(prefix + "_name"),
                    },
                )
                return key

            source = node("source", "call")
            target = node("target", "definition")
            provenance = {
                "origin": "parser",
                "branch_id": bid,
                "parser_edge_id": row["id"],
                "confidence": row["confidence"],
            }
            graph["edges"].append(
                {"source": source, "target": target, "verb": "CALLS", **provenance}
            )
            if row["caller_path"] is not None:
                caller = node("caller", "definition")
                graph["edges"].append(
                    {"source": caller, "target": source, "verb": "HAS_CALL", **provenance}
                )


async def read_graph(pool, *, snapshot_id=None, revision_id=None):
    if (snapshot_id is None) == (revision_id is None):
        raise ValueError("Select either an active snapshot or an explicit graph revision")
    async with pool.acquire() as conn:
        if revision_id is not None:
            row = await conn.fetchrow(
                "SELECT * FROM connection_graph_revisions WHERE id=$1", UUID(str(revision_id))
            )
        else:
            row = await conn.fetchrow(
                """SELECT r.* FROM connection_graph_revisions r
                JOIN connection_active_graphs a ON a.revision_id=r.id WHERE a.snapshot_id=$1""",
                UUID(str(snapshot_id)),
            )
        if row is None:
            raise ValueError("No published graph found")
        return {
            "revision_id": str(row["id"]),
            "fingerprint": row["fingerprint"],
            "graph": json.loads(row["graph"]),
        }


def neighborhood(graph, node_id, *, hops=1, direction="both", include_parser=False, max_nodes=200):
    """Bounded incident traversal; resource legs keep their original directions.

    Both-direction discovery can cross a shared resource, without inventing a
    direct call or claiming a producer consumes its own event.
    """
    if not 0 <= hops <= 10 or not 1 <= max_nodes <= 5000 or direction not in {"both", "out", "in"}:
        raise ValueError("Invalid traversal bounds or direction")
    if node_id not in graph["nodes"]:
        raise ValueError("Unknown graph node")
    edges = [e for e in graph["edges"] if include_parser or e["origin"] != "parser"]
    reached, frontier = {node_id}, {node_id}
    truncated = False
    for _ in range(hops):
        next_nodes = set()
        for edge in edges:
            if direction in {"both", "out"} and edge["source"] in frontier:
                next_nodes.add(edge["target"])
            if direction in {"both", "in"} and edge["target"] in frontier:
                next_nodes.add(edge["source"])
        candidates = sorted(next_nodes - reached)
        room = max_nodes - len(reached)
        truncated |= len(candidates) > room
        frontier = set(candidates[:room])
        reached |= frontier
        if not frontier:
            break
    selected = [e for e in edges if e["source"] in reached and e["target"] in reached]
    decisions = {e["decision_id"] for e in selected if "decision_id" in e}
    return {
        "nodes": {key: graph["nodes"][key] for key in sorted(reached)},
        "edges": selected,
        "decisions": [d for d in graph["decisions"] if d["id"] in decisions],
        "truncated": truncated,
        "hops": hops,
        "direction": direction,
    }
