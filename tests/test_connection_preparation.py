"""Reranking provenance and bounded captured-source evidence."""

from dataclasses import replace
import json
from uuid import UUID

import pytest

from core.connections.preparation import (
    PreparationOptions, RerankerIdentity, prepare_candidates, source_window,
)
from core.connections.retrieval import embed_snapshot_sites, retrieve_all_candidates
from core.connections.sites import index_snapshot_sites
from tests.test_connection_retrieval import corpus as _corpus, Embedder

corpus = _corpus


def test_windows_are_byte_bounded_and_utf8_aligned():
    raw = ('é' * 30 + 'invoke("topic")' + '🌊' * 100).encode()
    window = source_window(raw, 60, 75, 100, 50)
    assert window['excerpt'] == raw[window['start_byte']:window['end_byte']].decode()
    assert window['end_byte'] - window['start_byte'] <= 100
    assert not window['site_clipped']
    clipped = source_window(raw, 60, len(raw), 100, 50)
    assert clipped['site_clipped']
    assert clipped['start_byte'] == 60
    with pytest.raises(ValueError):
        PreparationOptions(total_bytes=100)
    with pytest.raises(ValueError):
        PreparationOptions(max_excerpts=1)
    with pytest.raises(ValueError):
        RerankerIdentity('', 'model')


async def retrieved(corpus):
    pool, inputs, snapshot = corpus
    index = await index_snapshot_sites(pool, snapshot.id)
    embedder = Embedder()
    await embed_snapshot_sites(pool, snapshot.id, index.index_version, embedder.embed)
    run = await retrieve_all_candidates(pool, snapshot.id, index.index_version, embedder.embedding_identity)
    return pool, inputs, snapshot, run


@pytest.mark.asyncio(loop_scope='session')
async def test_reranking_packets_reuse_and_provenance(corpus):
    pool, inputs, snapshot, retrieval = await retrieved(corpus)
    calls = []

    async def rerank(query, documents):
        calls.append((query, documents))
        return [10. if 'subscribe:' in d else -2. for d in documents]

    identity = RerankerIdentity('fixture', 'v1')
    options = PreparationOptions(target_limit=1, excerpt_bytes=100, total_bytes=300, max_excerpts=3)
    paths = ((inputs[0].repo_id, 'missing.conf'), (inputs[0].repo_id, 'binary'))
    run = await prepare_candidates(pool, retrieval.id, rerank, identity, options, additional_paths=paths)
    assert run.packet_count == 4 and len(calls) == 4
    assert await prepare_candidates(pool, retrieval.id, rerank, identity, options, additional_paths=paths) == run
    assert len(calls) == 4
    async with pool.acquire() as conn:
        rows = await conn.fetch('SELECT * FROM connection_evidence_packets WHERE preparation_run_id=$1', UUID(run.id))
        for row in rows:
            payload = json.loads(row['payload'])
            assert payload['alternatives'][0]['score'] == 10.
            assert sum(a['selected'] for a in payload['alternatives']) == 1
            assert len(payload['alternatives']) > 1
            assert payload['excerpt_bytes'] <= options.total_bytes
            assert {e['role'] for e in payload['excerpts']} == {'source', 'target'}
            assert {o['reason'] for o in payload['omissions']} == {'missing', 'non_utf8'}
            for excerpt in payload['excerpts']:
                stored = await conn.fetchrow('SELECT * FROM connection_evidence WHERE id=$1', UUID(excerpt['evidence_id']))
                assert stored['excerpt'] == excerpt['excerpt']
                assert len(stored['excerpt'].encode()) <= 100
        assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM connection_decisions WHERE snapshot_id=$1)', snapshot.id)
        assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM cross_connections WHERE snapshot_id=$1)', snapshot.id)
    newer = await prepare_candidates(pool, retrieval.id, rerank, replace(identity, model_version='v2'), options)
    assert newer.id != run.id
    assert len(calls) == 8


@pytest.mark.asyncio(loop_scope='session')
async def test_bad_scorer_rolls_back_entire_preparation(corpus):
    pool, _, snapshot, retrieval = await retrieved(corpus)
    for bad in ([], [float('nan')], [True]):
        count = 0

        async def rerank(query, docs):
            nonlocal count
            count += 1
            return [1.] * len(docs) if count == 1 else bad

        with pytest.raises(ValueError, match='finite score'):
            await prepare_candidates(pool, retrieval.id, rerank, RerankerIdentity('fixture', 'bad'))
        async with pool.acquire() as conn:
            assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM connection_preparation_runs WHERE snapshot_id=$1)', snapshot.id)
            assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM connection_evidence WHERE snapshot_id=$1)', snapshot.id)


@pytest.mark.asyncio(loop_scope='session')
async def test_invalid_inputs_do_not_invoke_reranker(corpus):
    pool, _, _, retrieval = await retrieved(corpus)

    async def forbidden(query, docs):
        pytest.fail('Scorer must not be invoked')

    with pytest.raises(ValueError, match='outside'):
        await prepare_candidates(pool, retrieval.id, forbidden, RerankerIdentity('fixture', 'v1'),
                                 additional_paths=((999999999, 'file'),))
    with pytest.raises(ValueError, match='completed'):
        await prepare_candidates(pool, UUID(int=0), forbidden, RerankerIdentity('fixture', 'v1'))


@pytest.mark.asyncio(loop_scope='session')
async def test_ties_clipping_and_context_budget(corpus):
    pool, inputs, _, retrieval = await retrieved(corpus)
    (inputs[0].root / 'code.py').write_text('uncommitted_change()')

    async def equal(query, docs):
        return [0.] * len(docs)

    options = PreparationOptions(target_limit=1, excerpt_bytes=100, total_bytes=200, max_excerpts=2)
    run = await prepare_candidates(pool, retrieval.id, equal, RerankerIdentity('fixture', 'ties'), options,
                                   additional_paths=((inputs[0].repo_id, 'events.conf'),))
    async with pool.acquire() as conn:
        for row in await conn.fetch('SELECT payload FROM connection_evidence_packets WHERE preparation_run_id=$1', UUID(run.id)):
            payload = json.loads(row['payload'])
            assert [a['retrieval_rank'] for a in payload['alternatives']] == list(range(1, len(payload['alternatives']) + 1))
            assert payload['omissions'][0]['reason'] == 'budget'
            assert 'uncommitted_change' not in str(payload)
            assert len(payload['excerpts']) == 2


@pytest.mark.asyncio(loop_scope='session')
async def test_timeout_rolls_back(corpus):
    import asyncio
    pool, _, snapshot, retrieval = await retrieved(corpus)

    async def stalled(query, docs):
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        await prepare_candidates(pool, retrieval.id, stalled, RerankerIdentity('fixture', 'timeout'),
                                 PreparationOptions(timeout_seconds=0.01))
    async with pool.acquire() as conn:
        assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM connection_preparation_runs WHERE snapshot_id=$1)', snapshot.id)


@pytest.mark.asyncio(loop_scope='session')
async def test_empty_candidates_do_not_call_scorer(corpus):
    from tests.test_connection_retrieval import git
    from core.connections.snapshots import capture_snapshot
    pool, inputs, _ = corpus
    for path in inputs[1].root.iterdir():
        if path.name != '.git':
            path.unlink()
    git(inputs[1].root, 'add', '-A')
    git(inputs[1].root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'empty')
    snapshot = await capture_snapshot(pool, tuple(inputs), context={})
    _, _, _, retrieval = await retrieved((pool, inputs, snapshot))

    async def forbidden(query, docs):
        pytest.fail('No candidates to rerank')

    run = await prepare_candidates(pool, retrieval.id, forbidden, RerankerIdentity('fixture', 'empty'))
    assert run.packet_count == 2
    async with pool.acquire() as conn:
        for row in await conn.fetch('SELECT payload FROM connection_evidence_packets WHERE preparation_run_id=$1', UUID(run.id)):
            payload = json.loads(row['payload'])
            assert payload['empty_candidates'] and payload['alternatives'] == []
            assert len(payload['excerpts']) == 1
