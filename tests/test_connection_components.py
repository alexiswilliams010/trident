"""Component grouping preserves behavioral meaning and historical evidence."""

from copy import deepcopy
from uuid import UUID

import asyncpg
import pytest

from core.connections.components import (
    ComponentRule,
    project_components,
    create_projection,
    read_projection,
    explain_component_edge,
    component_neighborhood,
)
from core.connections.assessment import assess_packet
from core.connections.publication import publish_graph
from tests.test_connection_assessment import prepared, FixtureClient
from tests.test_connection_publication import ResourceClient
from tests.test_connection_retrieval import corpus as _corpus

corpus = _corpus


def graph():
    return {
        "snapshot_id": "fixture",
        "members": [{"repo_id": 1}, {"repo_id": 2}],
        "nodes": {
            "site:a": {"repo_id": 1, "path": "src/pay/a.py"},
            "site:b": {"repo_id": 1, "path": "src/pay/b.py"},
            "site:c": {"repo_id": 2, "path": "consumer/main.py"},
        },
        "decisions": [
            {"id": "d1", "conditions": {"env": "test"}},
            {"id": "d2", "conditions": {"env": "test"}},
        ],
        "edges": [
            {
                "source": source,
                "target": "site:c",
                "verb": "INVOKES",
                "vocabulary_version": 1,
                "decision_id": decision,
                "origin": "cross_repository",
            }
            for source, decision in [("site:a", "d1"), ("site:b", "d2")]
        ],
    }


def test_default_aggregation_and_drilldown_indices():
    source = graph()
    original = deepcopy(source)
    result = project_components(source)
    assert source == original
    assert len(result["nodes"]) == 2 and len(result["edges"]) == 1
    edge = result["edges"][0]
    assert edge["connection_count"] == edge["decision_count"] == 2
    assert edge["connection_indices"] == [0, 1] and edge["decision_ids"] == ["d1", "d2"]
    assert edge["verb"] == "INVOKES"
    assert len(component_neighborhood(result, edge["source"])["nodes"]) == 2
    assert component_neighborhood(result, edge["source"], max_nodes=1)["truncated"]


def test_longest_prefix_and_path_boundaries():
    source = graph()
    source["nodes"]["site:b"]["path"] = "src/payment/b.py"
    rules = (ComponentRule(1, "all", "All", "src"), ComponentRule(1, "pay", "Payments", "src/pay"))
    result = project_components(source, rules)
    assert result == project_components(source, tuple(reversed(rules)))
    assert result["memberships"]["site:a"]["component_id"] == "component:1:named:pay"
    assert result["memberships"]["site:b"]["component_id"] == "component:1:named:all"
    assert len(result["edges"]) == 2
    assert (
        len(
            project_components(source, [ComponentRule(1, "unused", "Unused", "missing")])[
                "unused_rules"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("prefix", ["/src", "../src", "src//a", "src/", "./src", "src\\a"])
def test_invalid_prefix(prefix):
    with pytest.raises(ValueError):
        ComponentRule(1, "service", "Service", prefix)


def test_ambiguous_rules_and_outside_repository():
    for rules in (
        (ComponentRule(1, "a", "A"), ComponentRule(1, "b", "B")),
        (ComponentRule(1, "a", "A", "a"), ComponentRule(1, "a", "Other", "b")),
        (ComponentRule(99, "a", "A"),),
    ):
        with pytest.raises(ValueError):
            project_components(graph(), rules)
    source = graph()
    source["members"].pop()
    with pytest.raises(ValueError, match="two repositories"):
        project_components(source)


def test_conditions_verbs_and_direction_do_not_collapse():
    for change in ("conditions", "verb", "direction", "vocabulary"):
        source = graph()
        if change == "conditions":
            source["decisions"][1]["conditions"] = {"env": "prod"}
        elif change == "verb":
            source["edges"][1]["verb"] = "CONFIGURES"
        elif change == "vocabulary":
            source["edges"][1]["vocabulary_version"] = 2
        else:
            edge = source["edges"][1]
            edge["source"], edge["target"] = edge["target"], edge["source"]
        assert len(project_components(source)["edges"]) == 2


def test_parser_edges_are_not_projected():
    source = graph()
    source["edges"].append(
        {"source": "site:a", "target": "site:b", "origin": "parser", "verb": "CALLS"}
    )
    result = project_components(source)
    assert result["excluded_parser_edges"] == 1
    assert [edge["verb"] for edge in result["edges"]] == ["INVOKES"]


async def test_immutable_projection_and_historical_drilldown(corpus):
    pool, snapshot, packet = await prepared(corpus)
    decisions = await assess_packet(pool, packet, FixtureClient())
    revision = await publish_graph(pool, snapshot.id, decisions)
    projection_id = await create_projection(pool, revision)
    assert await create_projection(pool, revision) == projection_id
    view = await read_projection(pool, projection_id)
    assert len(view["projection"]["edges"]) == 1
    edge = view["projection"]["edges"][0]
    detail = await explain_component_edge(pool, projection_id, edge["id"])
    assert len(detail["connections"]) == 2
    assert all(decision["evidence"] for decision in detail["decisions"])
    await publish_graph(pool, snapshot.id, ())
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE connection_component_projections SET projection='{}' WHERE id=$1",
                UUID(projection_id),
            )
        await conn.execute(
            "UPDATE connection_decisions SET conditions='{}' WHERE id=$1", UUID(decisions[0])
        )
    assert await read_projection(pool, projection_id) == view
    assert await explain_component_edge(pool, projection_id, edge["id"]) == detail
    repo_id = detail["nodes"][detail["connections"][0]["source"]]["repo_id"]
    changed = await create_projection(
        pool, revision, [ComponentRule(repo_id, "service", "Service")]
    )
    assert changed != projection_id
    with pytest.raises(ValueError, match="Unknown component edge"):
        await explain_component_edge(pool, projection_id, "missing")


async def test_shared_resource_direction_and_identity_survive(corpus):
    pool, snapshot, packet = await prepared(corpus)
    decisions = await assess_packet(pool, packet, ResourceClient())
    revision = await publish_graph(pool, snapshot.id, decisions[:1])
    pid = await create_projection(pool, revision)
    projection = (await read_projection(pool, pid))["projection"]
    assert {e["verb"] for e in projection["edges"]} == {"PRODUCES", "CONSUMES"}
    assert len({e["target"] for e in projection["edges"]}) == 1
    source = projection["edges"][0]["source"]
    assert len(component_neighborhood(projection, source, hops=2)["nodes"]) == 3
    assert len(component_neighborhood(projection, source, hops=2, direction="out")["nodes"]) == 2
    detail = await explain_component_edge(pool, pid, projection["edges"][0]["id"])
    resource = next(n for key, n in detail["nodes"].items() if key.startswith("resource:"))
    assert resource["anchor"]["excerpt"]
    assert len(detail["decision_connections"]) == 2


async def test_component_cli_and_rule_file(corpus, monkeypatch, tmp_path):
    from contextlib import asynccontextmanager
    import json
    import cli.connections as cli

    pool, snapshot, packet = await prepared(corpus)
    decisions = await assess_packet(pool, packet, FixtureClient())
    revision = await publish_graph(pool, snapshot.id, decisions)

    @asynccontextmanager
    async def pool_ctx():
        yield pool

    monkeypatch.setattr(cli, "pool_ctx", pool_ctx)

    async def command(*args):
        return await cli.run(cli.parser().parse_args(list(args)))

    rules = tmp_path / "components.json"
    rules.write_text(
        json.dumps(
            [
                {
                    "repo_id": snapshot.members[0].repo_id,
                    "key": "worker",
                    "label": "Worker",
                    "prefix": "",
                }
            ]
        )
    )
    created = await command("project", "--revision", revision, "--rules", str(rules))
    pid = created["projection_id"]
    view = await command("components", "--projection", pid)
    assert "memberships" not in view["projection"]
    full = await command("components", "--projection", pid, "--memberships")
    assert full["projection"]["memberships"]
    edge = view["projection"]["edges"][0]
    detail = await command("explain-component-edge", "--projection", pid, "--edge", edge["id"])
    assert detail["decisions"][0]["evidence"]
    neighbors = await command("component-neighbors", "--projection", pid, "--node", edge["source"])
    assert neighbors["projection"]["graph_revision_id"] == revision
    assert len(neighbors["projection"]["edges"]) == 1
    rules.write_text('[{"repo_id": true, "key": "bad", "label": "Bad"}]')
    with pytest.raises(ValueError):
        await command("project", "--revision", revision, "--rules", str(rules))
    assert (await command("components", "--projection", pid)) == view


async def test_empty_revision_projection(corpus):
    pool, _, snapshot = corpus
    revision = await publish_graph(pool, snapshot.id, ())
    projection = await create_projection(pool, revision)
    result = (await read_projection(pool, projection))["projection"]
    assert result["nodes"] == {} and result["edges"] == [] and result["memberships"] == {}
