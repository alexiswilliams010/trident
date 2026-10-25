"""Committed source isolation, sealing, and selective decision reuse."""

from dataclasses import replace
import asyncio
import hashlib
from pathlib import Path
import subprocess
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio

from core.connections.dependencies import (
    ResolutionVersions,
    assess_decision_reuse,
    record_decision_inputs,
)
from core.connections.snapshots import RepositoryInput, capture_snapshot, read_source

pytestmark = pytest.mark.asyncio(loop_scope="session")


def git(root: Path, *args):
    return (
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


def commit(root):
    git(root, "add", "-A")
    git(
        root,
        "-c",
        "user.name=Trident Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
        "--allow-empty",
    )


@pytest_asyncio.fixture(loop_scope="session")
async def captured(pg_pool, tmp_path):
    inputs = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        git(root, "init")
        (root / "code.py").write_text("call(1)\n")
        (root / "config.json").write_text('{"target":"one"}\n')
        (root / "link").symlink_to("code.py")
        (root / "binary").write_bytes(b"\x00\xff")
        (root / "line\nwith\ttab").write_text("unusual path")
        commit(root)
        async with pg_pool.acquire() as conn:
            rid = await conn.fetchval(
                "INSERT INTO repos(name, root_path) VALUES ($1,$2) RETURNING id",
                f"snapshot-test-{uuid4()}",
                str(root),
            )
        inputs.append(RepositoryInput(rid, root))
    inputs = tuple(inputs)
    snapshot = await capture_snapshot(pg_pool, inputs, context={"environment": "test"})
    try:
        yield pg_pool, inputs, snapshot
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM connection_snapshots WHERE id IN
                  (SELECT snapshot_id FROM connection_snapshot_members WHERE repo_id=ANY($1::bigint[]))
            """,
                [r.repo_id for r in inputs],
            )
            await conn.execute(
                "DELETE FROM repos WHERE id=ANY($1::bigint[])", [r.repo_id for r in inputs]
            )


async def test_capture_requires_two_repositories_before_reading_git():
    repo = RepositoryInput(1, Path("/does-not-exist"))
    for inputs in ((repo,), (repo, replace(repo, revision="other"))):
        with pytest.raises(ValueError, match="two distinct"):
            await capture_snapshot(None, inputs, context={})
    with pytest.raises(ValueError, match="same checkout"):
        await capture_snapshot(None, (repo, replace(repo, repo_id=2)), context={})


async def test_capture_reuses_identity_and_ignores_dirty_worktree(captured):
    pool, inputs, snapshot = captured
    (inputs[0].root / "code.py").write_text("call(2)\n")
    (inputs[0].root / "untracked").write_text("not committed")
    repeated = await capture_snapshot(
        pool, tuple(reversed(inputs)), context={"environment": "test"}
    )
    assert repeated.id == snapshot.id
    async with pool.acquire() as conn:
        assert await read_source(conn, snapshot.id, inputs[0].repo_id, "code.py") == b"call(1)\n"
        assert await read_source(conn, snapshot.id, inputs[0].repo_id, "link") == b"code.py"
        assert await read_source(conn, snapshot.id, inputs[0].repo_id, "binary") == b"\x00\xff"
        assert (
            await read_source(conn, snapshot.id, inputs[0].repo_id, "line\nwith\ttab")
            == b"unusual path"
        )
        with pytest.raises(ValueError, match="absent"):
            await read_source(conn, snapshot.id, inputs[0].repo_id, "untracked")
    commit(inputs[0].root)
    newer = await capture_snapshot(pool, inputs, context={"environment": "test"})
    assert newer.id != snapshot.id
    async with pool.acquire() as conn:
        assert await read_source(conn, newer.id, inputs[0].repo_id, "code.py") == b"call(2)\n"
        assert await read_source(conn, snapshot.id, inputs[0].repo_id, "code.py") == b"call(1)\n"


async def test_concurrent_capture_converges_on_one_sealed_snapshot(captured):
    pool, inputs, _ = captured
    snapshots = await asyncio.gather(
        *(capture_snapshot(pool, inputs, context={"concurrent": True}) for _ in range(2))
    )
    assert snapshots[0].id == snapshots[1].id


async def test_worktrees_of_same_repository_do_not_qualify(captured, tmp_path):
    pool, inputs, _ = captured
    worktree = tmp_path / "worktree"
    git(inputs[0].root, "worktree", "add", "--detach", str(worktree), "HEAD")
    with pytest.raises(ValueError, match="Worktrees"):
        await capture_snapshot(pool, (inputs[0], replace(inputs[1], root=worktree)), context={})


async def test_sealed_manifest_and_metadata_cannot_change(captured):
    pool, _, snapshot = captured
    async with pool.acquire() as conn:
        for sql in (
            "UPDATE connection_snapshots SET context_fingerprint='other' WHERE id=$1",
            "UPDATE connection_snapshots SET sealed_at=NULL WHERE id=$1",
            "UPDATE connection_snapshot_members SET revision=repeat('a',40) WHERE snapshot_id=$1",
            "DELETE FROM connection_snapshot_members WHERE snapshot_id=$1",
            "UPDATE connection_snapshot_files SET path='other' WHERE snapshot_id=$1",
            "DELETE FROM connection_snapshot_files WHERE snapshot_id=$1",
        ):
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(sql, UUID(snapshot.id))


async def seed_decision(pool, inputs, snapshot):
    versions = ResolutionVersions(
        "fake", "model-1", "policy-1", 1, "site-v1", "search-1", "rank-1", "set-1"
    )
    async with pool.acquire() as conn, conn.transaction():
        sites = []
        for repo in inputs:
            raw = await read_source(conn, snapshot.id, repo.repo_id, "code.py")
            sites.append(
                await conn.fetchval(
                    """
                INSERT INTO interaction_sites
                    (snapshot_id, repo_id, path, start_byte, end_byte, content_hash, kind)
                VALUES ($1,$2,'code.py',0,$3,$4,'call') RETURNING id
            """,
                    UUID(snapshot.id),
                    repo.repo_id,
                    len(raw),
                    hashlib.sha256(raw).hexdigest(),
                )
            )
        candidate = await conn.fetchval(
            """
            INSERT INTO connection_candidates
                (snapshot_id, source_site_id, source_repo_id, target_site_id, target_repo_id)
            VALUES ($1,$2,$3,$4,$5) RETURNING id
        """,
            UUID(snapshot.id),
            sites[0],
            inputs[0].repo_id,
            sites[1],
            inputs[1].repo_id,
        )
        decision = await conn.fetchval(
            """
            INSERT INTO connection_decisions
                (snapshot_id,candidate_id,outcome,provider,model_version,policy_version,evidence_fingerprint)
            VALUES ($1,$2,'supported','fake','model-1','policy-1','evidence-1') RETURNING id
        """,
            UUID(snapshot.id),
            candidate,
        )
        raw = await read_source(conn, snapshot.id, inputs[1].repo_id, "config.json")
        evidence = await conn.fetchval(
            """
            INSERT INTO connection_evidence
                (snapshot_id,repo_id,path,start_byte,end_byte,content_hash,excerpt)
            VALUES ($1,$2,'config.json',0,$3,$4,$5) RETURNING id
        """,
            UUID(snapshot.id),
            inputs[1].repo_id,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            raw.decode(),
        )
        await conn.execute(
            "INSERT INTO connection_decision_evidence VALUES ($1,$2,$3,'support')",
            UUID(snapshot.id),
            decision,
            evidence,
        )
        await conn.execute(
            """
            INSERT INTO cross_connections
                (snapshot_id,decision_id,source_site_id,target_site_id,verb_version,verb)
            VALUES ($1,$2,$3,$4,1,'INVOKES')
        """,
            UUID(snapshot.id),
            decision,
            *sites,
        )
        await record_decision_inputs(
            conn, decision, versions, additional_paths=((inputs[1].repo_id, "missing.yml"),)
        )
    return decision, versions


@pytest.mark.parametrize(
    "change", ["source", "target", "config", "delete", "missing", "mode", "unrelated"]
)
async def test_dependency_changes_invalidate_only_affected_decisions(captured, change):
    pool, inputs, snapshot = captured
    decision, versions = await seed_decision(pool, inputs, snapshot)
    root = inputs[0].root if change == "source" else inputs[1].root
    if change in {"source", "target"}:
        (root / "code.py").write_text("call(2)\n")
    elif change == "config":
        (root / "config.json").write_text('{"target":"two"}\n')
    elif change == "delete":
        (root / "config.json").unlink()
    elif change == "mode":
        git(root, "update-index", "--chmod=+x", "code.py")
        (root / "code.py").chmod(0o755)
    else:
        (root / ("missing.yml" if change == "missing" else "unrelated.txt")).write_text("new")
    commit(root)
    newer = await capture_snapshot(pool, inputs, context={"environment": "test"})
    async with pool.acquire() as conn:
        result = await assess_decision_reuse(conn, decision, newer.id, versions)
        assert result.reusable == (change == "unrelated")
        if change != "unrelated":
            assert "source_target_or_evidence_changed" in result.reasons
        assert (await assess_decision_reuse(conn, decision, snapshot.id, versions)).reusable
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM cross_connections WHERE decision_id=$1", decision
            )
            == 1
        )


async def test_versions_context_and_changed_evidence_block_reuse(captured):
    pool, inputs, snapshot = captured
    decision, versions = await seed_decision(pool, inputs, snapshot)
    newer = await capture_snapshot(pool, inputs, context={"environment": "prod"})
    async with pool.acquire() as conn:
        assert (
            "configuration_context_changed"
            in (await assess_decision_reuse(conn, decision, newer.id, versions)).reasons
        )
        for field in (
            "provider",
            "model_version",
            "policy_version",
            "representation_version",
            "retrieval_version",
            "reranker_version",
            "candidate_set_fingerprint",
        ):
            result = await assess_decision_reuse(
                conn, decision, snapshot.id, replace(versions, **{field: "changed"})
            )
            assert "resolution_versions_or_candidates_changed" in result.reasons
        result = await assess_decision_reuse(
            conn, decision, snapshot.id, replace(versions, vocabulary_version=2)
        )
        assert not result.reusable
        await conn.execute(
            "UPDATE connection_decisions SET conditions=$2::jsonb WHERE id=$1",
            decision,
            '{"env":"new"}',
        )
        assert (
            "decision_or_evidence_changed"
            in (await assess_decision_reuse(conn, decision, snapshot.id, versions)).reasons
        )


async def test_evidence_must_match_captured_bytes(captured):
    pool, inputs, snapshot = captured
    raw = b"call(1)\n"
    async with pool.acquire() as conn:
        for end, digest, excerpt in (
            (999, hashlib.sha256(raw).hexdigest(), raw.decode()),
            (len(raw), "wrong", raw.decode()),
            (len(raw), hashlib.sha256(raw).hexdigest(), "invented"),
        ):
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO connection_evidence
                        (snapshot_id,repo_id,path,start_byte,end_byte,content_hash,excerpt)
                    VALUES ($1,$2,'code.py',0,$3,$4,$5)
                """,
                    UUID(snapshot.id),
                    inputs[0].repo_id,
                    end,
                    digest,
                    excerpt,
                )
