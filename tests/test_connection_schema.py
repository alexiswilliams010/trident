"""Database contract tests. Run against an isolated database with DATABASE_URL set."""

import uuid

import asyncpg
import pytest
import pytest_asyncio

from core.connections.verbs import VERBS, VOCABULARY_VERSION
from db.connection import apply_migrations

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(loop_scope="session")
async def setup_graph(pg_pool):
    async with pg_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            snapshot = await conn.fetchval(
                "INSERT INTO connection_snapshots(context_fingerprint) VALUES ('cfg') RETURNING id"
            )
            repos, sites = [], []
            for i in range(2):
                repo = await conn.fetchval("INSERT INTO repos(name) VALUES ($1) RETURNING id",
                                           f"connection-test-{uuid.uuid4()}")
                repos.append(repo)
                await conn.execute(
                    "INSERT INTO connection_snapshot_members VALUES ($1, $2, $3)",
                    snapshot, repo, f"commit-{i}",
                )
                sites.append(await conn.fetchval("""
                    INSERT INTO interaction_sites
                        (snapshot_id, repo_id, path, start_byte, end_byte, content_hash, kind)
                    VALUES ($1, $2, 'source', 0, 10, 'hash', 'call') RETURNING id
                """, snapshot, repo))
            candidate = await conn.fetchval("""
                INSERT INTO connection_candidates
                    (snapshot_id, source_site_id, source_repo_id, target_site_id, target_repo_id)
                VALUES ($1, $2, $3, $4, $5) RETURNING id
            """, snapshot, sites[0], repos[0], sites[1], repos[1])
            decision = await conn.fetchval("""
                INSERT INTO connection_decisions
                    (snapshot_id, candidate_id, outcome, provider, model_version,
                     policy_version, evidence_fingerprint)
                VALUES ($1, $2, 'supported', 'fake', 'v1', 'v1', 'hash') RETURNING id
            """, snapshot, candidate)
            evidence = await conn.fetchval("""
                INSERT INTO connection_evidence
                    (snapshot_id, repo_id, path, start_byte, end_byte, content_hash, excerpt)
                VALUES ($1, $2, 'config', 0, 4, 'hash', 'data') RETURNING id
            """, snapshot, repos[0])
            await conn.execute("INSERT INTO connection_decision_evidence VALUES ($1,$2,$3,'support')",
                               snapshot, decision, evidence)
            resource = await conn.fetchval("""
                INSERT INTO connection_resources(snapshot_id, identity_key, kind)
                VALUES ($1, 'environment/cluster/topic', 'topic') RETURNING id
            """, snapshot)
            yield conn, snapshot, repos, sites, candidate, decision, resource
        finally:
            await transaction.rollback()


async def insert_edge(g, source, target, verb="INVOKES", resource=False):
    conn, snapshot, _, _, _, decision, _ = g
    column = "target_resource_id" if resource else "target_site_id"
    await conn.execute(f"""
        INSERT INTO cross_connections
            (snapshot_id, decision_id, source_site_id, {column}, verb_version, verb)
        VALUES ($1,$2,$3,$4,1,$5)
    """, snapshot, decision, source, target, verb)


async def test_schema_vocabulary_matches_python_and_migration_is_tracked(pg_pool):
    async with pg_pool.acquire() as conn:
        assert await apply_migrations(conn) == []
        rows = await conn.fetch("SELECT * FROM connection_verbs WHERE version=$1", VOCABULARY_VERSION)
        assert {r["name"] for r in rows} == set(VERBS)
        for row in rows:
            spec = VERBS[row["name"]]
            assert row["definition"] == spec.definition
            assert row["allows_site_target"] == (("site", "site") in spec.endpoint_pairs)
            assert row["allows_resource_target"] == (("site", "resource") in spec.endpoint_pairs)


async def test_candidate_cannot_forge_repository_or_snapshot(setup_graph):
    conn, snapshot, repos, sites, _, _, _ = setup_graph
    for target_repo in (repos[0], repos[1]):
        with pytest.raises(asyncpg.IntegrityConstraintViolationError):
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO connection_candidates
                      (snapshot_id, source_site_id, source_repo_id, target_site_id, target_repo_id)
                    VALUES ($1,$2,$3,$2,$4)
                """, snapshot, sites[0], repos[0], target_repo)
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        async with conn.transaction():
            await conn.execute("""
                INSERT INTO connection_candidates
                  (snapshot_id, source_site_id, source_repo_id, target_site_id, target_repo_id)
                VALUES ($1,$2,$3,$4,$5)
            """, uuid.uuid4(), sites[0], repos[0], sites[1], repos[1])


async def test_direct_connection_requires_supported_decision_and_evidence(setup_graph):
    g = setup_graph
    conn, _, _, sites, _, decision, _ = g
    await insert_edge(g, *sites)
    await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    for sql in (
        "UPDATE connection_decisions SET outcome='ambiguous' WHERE id=$1",
        "DELETE FROM connection_decision_evidence WHERE decision_id=$1",
    ):
        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await conn.execute(sql, decision)


async def test_shared_resource_is_atomic_and_not_a_direct_call(setup_graph):
    g = setup_graph
    conn, _, _, sites, _, _, resource = g
    with pytest.raises(asyncpg.CheckViolationError, match="both repositories"):
        async with conn.transaction():
            await insert_edge(g, sites[0], resource, "PRODUCES", True)
            await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    await insert_edge(g, sites[0], resource, "PRODUCES", True)
    await insert_edge(g, sites[1], resource, "CONSUMES", True)
    await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    assert await conn.fetchval(
        "SELECT count(*) FROM cross_connections WHERE decision_id=$1 AND target_site_id IS NOT NULL",
        g[5],
    ) == 0
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await conn.execute("DELETE FROM cross_connections WHERE decision_id=$1 AND verb='CONSUMES'",
                               g[5])


async def test_incompatible_verb_and_candidate_retargeting_are_rejected(setup_graph):
    g = setup_graph
    conn, snapshot, repos, sites, candidate, _, _ = g
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await insert_edge(g, *sites, verb="PRODUCES")
            await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    await insert_edge(g, *sites)
    await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    other = await conn.fetchval("""
        INSERT INTO interaction_sites
            (snapshot_id, repo_id, path, start_byte, end_byte, content_hash, kind)
        VALUES ($1, $2, 'other', 0, 10, 'hash', 'call') RETURNING id
    """, snapshot, repos[1])
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await conn.execute("UPDATE connection_candidates SET target_site_id=$1 WHERE id=$2",
                               other, candidate)
