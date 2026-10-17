"""Snapshot extraction and candidate recall without a live embedding service."""

from dataclasses import replace
import subprocess
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from core.connections.sites import SiteOptions, extract_file, index_snapshot_sites
from core.connections.retrieval import (
    RetrievalOptions, _fuse_channels, embed_snapshot_sites, retrieve_all_candidates, retrieve_candidates,
)
from core.connections.snapshots import RepositoryInput, capture_snapshot
from core.embedding_identity import EmbeddingIdentity


@pytest.mark.parametrize("path,source", [
    ("code.py", 'def run():\n    local()\n    client.send("invoice.created")\n'),
    ("code.js", 'function run() { local(); client.send("invoice.created"); }'),
    ("code.ts", 'function run(): void { local(); client.send("invoice.created"); }'),
    ("code.go", 'package main\nfunc run() { local(); client.Send("invoice.created") }'),
    ("code.rs", 'fn run() { local(); client.send("invoice.created"); }'),
    ("code.sol", 'contract C { function run() public { local(); client.send("invoice.created"); } }'),
])
def test_all_configured_calls_including_local(path, source):
    raw = source.encode()
    sites = extract_file(path, raw)
    calls = [s for s in sites if s.kind == "call"]
    assert len(calls) >= 2
    assert any(raw[s.start_byte:s.end_byte] == b"local()" for s in calls)
    assert any("invoice.created" in s.terms for s in calls)
    assert any(s.kind == "definition" for s in sites)


def test_generic_passages_and_unicode_offsets():
    raw = ('é' * 90 + '\nroute: invoice.created\n' * 20).encode()
    sites = extract_file("infra/custom.conf", raw, SiteOptions(passage_chars=100, passage_overlap=20))
    assert len(sites) > 2
    for site in sites:
        assert site.kind == "configuration"
        assert raw[site.start_byte:site.end_byte].decode().strip() in site.retrieval_text
    assert extract_file("binary", b"\0bad") == []
    assert extract_file("binary", b"\xff") == []
    calls = [s for s in extract_file("code.py", 'x = "é"\ncall(x)\n'.encode()) if s.kind == "call"]
    assert calls[0].start_byte == len('x = "é"\n'.encode())


def test_option_bounds():
    with pytest.raises(ValueError):
        SiteOptions(passage_overlap=1800)
    with pytest.raises(ValueError):
        RetrievalOptions(candidate_limit=51)
    with pytest.raises(ValueError):
        RetrievalOptions(exact_reserve=31)


def test_exact_reserve_and_passage_dedup_preserve_specific_occurrences():
    def row(key, kind, start, end):
        return dict(id=key, repo_id=2, path="code.py", kind=kind, start_byte=start,
                    end_byte=end, content_hash="content", retrieval_hash=key,
                    retrieval_text=key, metadata="{}", score=1.)
    call = row("call", "call", 10, 20)
    other_call = row("other", "call", 30, 40)
    passage = row("passage", "declaration", 0, 100)
    duplicate = row("duplicate", "declaration", 80, 180)
    exact = row("exact", "configuration", 200, 300)
    channels = {"semantic": [call, other_call, passage, duplicate], "exact": [exact]}
    hits = _fuse_channels(channels, RetrievalOptions(candidate_limit=4, exact_reserve=1))
    assert [h.target_site_id for h in hits] == ["exact", "call", "other", "passage"]
    hits = _fuse_channels(channels, RetrievalOptions(candidate_limit=1, exact_reserve=1))
    assert hits[0].target_site_id == "exact"


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest_asyncio.fixture(loop_scope="session")
async def corpus(pg_pool, tmp_path):
    inputs = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        git(root, "init")
        (root / "code.py").write_text('def run():\n    publish("invoice.created")\n    local()\n')
        (root / "events.conf").write_text('subscribe: invoice.created\n')
        (root / "binary").write_bytes(b"\0\xff")
        (root / "link").symlink_to("code.py")
        git(root, "add", "-A")
        git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
        async with pg_pool.acquire() as conn:
            rid = await conn.fetchval("INSERT INTO repos(name,root_path) VALUES ($1,$2) RETURNING id",
                                      str(uuid4()), str(root))
        inputs.append(RepositoryInput(rid, root))
    snapshot = await capture_snapshot(pg_pool, tuple(inputs), context={})
    try:
        yield pg_pool, inputs, snapshot
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM connection_snapshots WHERE id IN (SELECT snapshot_id FROM connection_snapshot_members WHERE repo_id=ANY($1::bigint[]))", [r.repo_id for r in inputs])
            await conn.execute("DELETE FROM repos WHERE id=ANY($1::bigint[])", [r.repo_id for r in inputs])


class Embedder:
    def __init__(self):
        self.embedding_identity = EmbeddingIdentity(str(uuid4()), "fixture-v1", 3, "site-v1")
        self.inputs = []

    async def embed(self, texts):
        self.inputs.extend(texts)
        # A deliberately poor semantic match tests the independent exact channel.
        return [[-1., 0., 0.] if "subscribe:" in t else [1., 0., 0.] for t in texts]


@pytest.mark.asyncio(loop_scope="session")
async def test_snapshot_inventory_embedding_reuse_and_cross_repo_recall(corpus):
    pool, inputs, snapshot = corpus
    (inputs[0].root / "code.py").write_text("dirty_uncommitted()\n")
    index = await index_snapshot_sites(pool, snapshot.id)
    assert index.stats["calls"] == 4
    assert len(index.stats["skipped"]) == 4
    assert await index_snapshot_sites(pool, snapshot.id) == index
    embedder = Embedder()
    async with pool.acquire() as conn:
        total, unique = await conn.fetchrow("SELECT count(*),count(DISTINCT retrieval_hash) FROM interaction_sites WHERE snapshot_id=$1", snapshot.id)
        source = await conn.fetchrow("SELECT * FROM interaction_sites WHERE snapshot_id=$1 AND repo_id=$2 AND kind='call' AND retrieval_text LIKE 'publish%'", snapshot.id, inputs[0].repo_id)
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM interaction_sites WHERE snapshot_id=$1 AND retrieval_text LIKE '%dirty_uncommitted%')", snapshot.id)
    assert unique < total
    with pytest.raises(ValueError, match="missing embeddings"):
        await retrieve_candidates(pool, snapshot.id, index.index_version, source["id"], embedder.embedding_identity)
    assert await embed_snapshot_sites(pool, snapshot.id, index.index_version, embedder.embed) == unique
    assert await embed_snapshot_sites(pool, snapshot.id, index.index_version, embedder.embed) == 0
    assert len(embedder.inputs) == unique
    hits = await retrieve_candidates(pool, snapshot.id, index.index_version, source["id"], embedder.embedding_identity)
    assert hits and all(h.target_repo_id == inputs[1].repo_id for h in hits)
    event = next(h for h in hits if h.path == "events.conf")
    assert "exact" in event.channels and event.channels["semantic"]["score"] < 0
    run = await retrieve_all_candidates(pool, snapshot.id, index.index_version, embedder.embedding_identity)
    assert run.source_count == 4
    assert await retrieve_all_candidates(pool, snapshot.id, index.index_version, embedder.embedding_identity) == run
    smaller = await retrieve_all_candidates(pool, snapshot.id, index.index_version, embedder.embedding_identity,
                                           RetrievalOptions(candidate_limit=1, exact_reserve=1))
    assert smaller.id != run.id and smaller.candidate_count == 4
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM connection_candidate_matches WHERE run_id=$1", UUID(run.id)) == run.candidate_count
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM connection_decisions WHERE snapshot_id=$1)", snapshot.id)
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM cross_connections WHERE snapshot_id=$1)", snapshot.id)
    with pytest.raises(ValueError, match="representation"):
        await retrieve_all_candidates(pool, snapshot.id, index.index_version,
                                      replace(embedder.embedding_identity, representation_version="chunk-v1"))


@pytest.mark.asyncio(loop_scope="session")
async def test_invalid_vectors_are_not_cached(corpus):
    pool, _, snapshot = corpus
    index = await index_snapshot_sites(pool, snapshot.id)
    identity = EmbeddingIdentity(str(uuid4()), "invalid", 3, "site-v1")
    for bad in ([0., 0., 0.], [float("nan"), 1., 1.], [1.]):
        async def embed(texts):
            return [bad for _ in texts]
        with pytest.raises(ValueError, match="invalid"):
            await embed_snapshot_sites(pool, snapshot.id, index.index_version, embed,
                                       embedding_identity=identity)
    async with pool.acquire() as conn:
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM interaction_embeddings WHERE model_name=$1)", identity.model_name)


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_candidate_sets_and_index_isolation(corpus):
    pool, inputs, _ = corpus
    for path in inputs[1].root.iterdir():
        if path.name != ".git":
            path.unlink()
    git(inputs[1].root, "add", "-A")
    git(inputs[1].root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "empty")
    snapshot = await capture_snapshot(pool, tuple(inputs), context={})
    first = await index_snapshot_sites(pool, snapshot.id)
    second = await index_snapshot_sites(pool, snapshot.id, SiteOptions(context_chars=100))
    assert first.index_version != second.index_version
    embedder = Embedder()
    await embed_snapshot_sites(pool, snapshot.id, first.index_version, embedder.embed)
    run = await retrieve_all_candidates(pool, snapshot.id, first.index_version, embedder.embedding_identity)
    assert run.source_count == 2 and run.candidate_count == 0
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM connection_candidate_sets WHERE run_id=$1 AND candidate_count=0", UUID(run.id)) == 2
        wrong_source = await conn.fetchval("SELECT id FROM interaction_sites WHERE snapshot_id=$1 AND index_version=$2 AND kind='call' LIMIT 1", snapshot.id, second.index_version)
    with pytest.raises(ValueError, match="Source must"):
        await retrieve_candidates(pool, snapshot.id, first.index_version, wrong_source, embedder.embedding_identity)


@pytest.mark.asyncio(loop_scope="session")
async def test_failed_extraction_rolls_back_and_unsealed_rejected(corpus, monkeypatch):
    pool, inputs, snapshot = corpus
    import core.connections.sites as sites_module
    original = sites_module.extract_file
    count = 0

    def fail_later(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("extraction interrupted")
        return original(*args)

    monkeypatch.setattr(sites_module, "extract_file", fail_later)
    with pytest.raises(RuntimeError, match="interrupted"):
        await index_snapshot_sites(pool, snapshot.id)
    async with pool.acquire() as conn:
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM interaction_sites WHERE snapshot_id=$1)", snapshot.id)
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM connection_site_indexes WHERE snapshot_id=$1)", snapshot.id)
        draft = await conn.fetchval("INSERT INTO connection_snapshots(context_fingerprint) VALUES ('draft') RETURNING id")
        await conn.execute("INSERT INTO connection_snapshot_members VALUES ($1,$2,'HEAD')", draft, inputs[0].repo_id)
    embedder = Embedder()
    with pytest.raises(ValueError, match="sealed"):
        await embed_snapshot_sites(pool, draft, "missing", embedder.embed)
    with pytest.raises(ValueError, match="sealed"):
        await index_snapshot_sites(pool, draft)
    assert embedder.inputs == []
