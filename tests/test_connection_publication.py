"""Atomic graph activation, historical reads, and mixed graph traversal."""

from uuid import UUID, uuid4

import asyncpg
import pytest

from cli.connections import parser
from core.connections.assessment import assess_packet
from core.connections.publication import publish_graph
from core.connections.published_queries import read_graph, neighborhood
from tests.test_connection_retrieval import corpus as _corpus
from tests.test_connection_assessment import prepared, FixtureClient, response_for

corpus = _corpus


async def test_publish_activate_and_preserve_history(corpus):
    pool, snapshot, packet = await prepared(corpus)
    decisions = await assess_packet(pool, packet, FixtureClient())
    revision = await publish_graph(pool, snapshot.id, decisions)
    view = await read_graph(pool, snapshot_id=snapshot.id)
    assert view['revision_id'] == revision
    assert len(view['graph']['edges']) == 2
    assert await publish_graph(pool, snapshot.id, tuple(reversed(decisions))) == revision
    empty = await publish_graph(pool, snapshot.id, ())
    assert empty != revision
    assert (await read_graph(pool, snapshot_id=snapshot.id))['graph']['edges'] == []
    assert await read_graph(pool, revision_id=revision) == view
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError, match='immutable'):
            await conn.execute("UPDATE connection_graph_revisions SET graph='{}' WHERE id=$1", UUID(revision))
    assert await publish_graph(pool, snapshot.id, decisions) == revision


async def test_failed_publication_does_not_replace_active(corpus):
    pool, snapshot, packet = await prepared(corpus)
    decisions = await assess_packet(pool, packet, FixtureClient())
    active = await publish_graph(pool, snapshot.id, ())
    async with pool.acquire() as conn:
        await conn.execute("UPDATE connection_assessments SET proposal='{}' WHERE decision_id=$1", UUID(decisions[-1]))
    with pytest.raises(ValueError, match='changed'):
        await publish_graph(pool, snapshot.id, decisions)
    assert (await read_graph(pool, snapshot_id=snapshot.id))['revision_id'] == active
    async with pool.acquire() as conn:
        assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM cross_connections WHERE snapshot_id=$1)', snapshot.id)
    with pytest.raises(ValueError, match='supported'):
        await publish_graph(pool, snapshot.id, [UUID(int=0)])


class ResourceClient(FixtureClient):
    async def decide(self, state, questions):
        anchor = next(key for key in questions['resource_anchor']['criteria'] if key != 'none')
        return response_for(questions, {'topology': 'shared_resource', 'resource_anchor': anchor})


async def test_resource_pair_publication_and_direction(corpus):
    pool, snapshot, packet = await prepared(corpus)
    decisions = await assess_packet(pool, packet, ResourceClient())
    revision = await publish_graph(pool, snapshot.id, decisions[:1])
    graph = (await read_graph(pool, revision_id=revision))['graph']
    assert {e['verb'] for e in graph['edges']} == {'PRODUCES', 'CONSUMES'}
    resource = next(key for key in graph['nodes'] if key.startswith('resource:'))
    assert len({e['source'] for e in graph['edges']}) == 2
    assert {e['target'] for e in graph['edges']} == {resource}
    producer = next(e['source'] for e in graph['edges'] if e['verb'] == 'PRODUCES')
    assert len(neighborhood(graph, producer, hops=2)['nodes']) == 3
    assert len(neighborhood(graph, producer, hops=2, direction='out')['nodes']) == 2
    assert neighborhood(graph, producer, hops=2, max_nodes=1)['truncated']
    async with pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM cross_connections WHERE snapshot_id=$1', snapshot.id) == 2


async def test_parser_capture_and_historical_isolation(corpus):
    from core.extractor import index_repo
    from core.semantic_resolver import resolve_repo
    from tests.test_connection_retrieval import git
    from core.connections.snapshots import capture_snapshot
    from core.connections.sites import index_snapshot_sites
    from core.connections.retrieval import embed_snapshot_sites, retrieve_all_candidates
    from core.connections.preparation import prepare_candidates, RerankerIdentity
    from tests.test_connection_retrieval import Embedder
    pool, inputs, _ = corpus
    root = inputs[0].root
    (root / 'code.py').write_text('def local():\n    pass\ndef run():\n    local()\n')
    git(root, 'add', '-A')
    git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'local call')
    snapshot = await capture_snapshot(pool, tuple(inputs), context={})
    index = await index_snapshot_sites(pool, snapshot.id)
    embedder = Embedder()
    await embed_snapshot_sites(pool, snapshot.id, index.index_version, embedder.embed)
    retrieval = await retrieve_all_candidates(pool, snapshot.id, index.index_version, embedder.embedding_identity)
    async def rerank(query, docs):
        return [10. if d.startswith(('publish(', 'local(')) else 0. for d in docs]
    preparation = await prepare_candidates(pool, retrieval.id, rerank, RerankerIdentity('fixture', 'v1'))
    async with pool.acquire() as conn:
        bid = await conn.fetchval("INSERT INTO branches(repo_id,name) VALUES ($1,'parser') RETURNING id", inputs[0].repo_id)
        packet = await conn.fetchval('''SELECT p.id FROM connection_evidence_packets p JOIN interaction_sites s ON s.id=p.source_site_id
            WHERE p.preparation_run_id=$1 AND s.repo_id=$2 LIMIT 1''', UUID(preparation.id), inputs[0].repo_id)
    await index_repo(pool, inputs[0].repo_id, bid, root)
    await resolve_repo(pool, inputs[0].repo_id, bid)
    decisions = await assess_packet(pool, packet, FixtureClient())
    revision = await publish_graph(pool, snapshot.id, decisions[:1], branch_ids=[bid])
    before = await read_graph(pool, revision_id=revision)
    parser_edges = [e for e in before['graph']['edges'] if e['origin'] == 'parser']
    assert {e['verb'] for e in parser_edges} == {'HAS_CALL', 'CALLS'}
    source = next(e['source'] for e in before['graph']['edges'] if e['origin'] == 'cross_repository')
    assert any(e['source'] == source for e in parser_edges)
    assert any(e['origin'] == 'parser' for e in neighborhood(before['graph'], source, include_parser=True)['edges'])
    assert all(e['origin'] != 'parser' for e in neighborhood(before['graph'], source)['edges'])
    (root / 'code.py').write_text('changed()\n')
    await index_repo(pool, inputs[0].repo_id, bid, root)
    with pytest.raises(ValueError, match='source differs'):
        await publish_graph(pool, snapshot.id, decisions[:1], branch_ids=[bid])
    assert await read_graph(pool, revision_id=revision) == before
    assert (await read_graph(pool, snapshot_id=snapshot.id))['revision_id'] == revision


def test_cli_requires_explicit_membership_and_read_scope():
    assert parser().parse_args(['publish', '--snapshot', 's', '--empty']).empty
    assert parser().parse_args(['publish', '--snapshot', 's', '--decision', 'a', '--decision', 'b']).decision == ['a', 'b']
    for args in (['publish', '--snapshot', 's'], ['show'], ['show', '--snapshot', 's', '--revision', 'r']):
        with pytest.raises(SystemExit):
            parser().parse_args(args)


def test_query_bounds():
    with pytest.raises(ValueError):
        neighborhood({}, 'missing', hops=11)
    with pytest.raises(ValueError, match='Unknown'):
        neighborhood({'nodes': {}}, 'missing')


async def test_cli_pipeline_with_mocked_models(corpus, monkeypatch):
    from contextlib import asynccontextmanager
    import json
    import httpx
    import cli.connections as cli
    from core.connections.jev import JevClient
    from core.embedding_identity import EmbeddingIdentity
    pool, inputs, _ = corpus

    @asynccontextmanager
    async def pool_ctx():
        yield pool

    monkeypatch.setattr(cli, 'pool_ctx', pool_ctx)
    for key, value in {'EMBEDDING_BASE_URL': 'https://example.invalid', 'EMBEDDING_API_KEY': 'fixture',
                       'EMBEDDING_MODEL': f'fixture-cli-{uuid4()}', 'EMBEDDING_DIM': '3', 'AI_GATEWAY_API_KEY': 'fixture'}.items():
        monkeypatch.setenv(key, value)
    closed = []
    class Embedder:
        def __init__(self, config):
            self.embedding_identity = EmbeddingIdentity(config.model, config.model_version, config.dim, config.representation_version)
        async def embed(self, texts):
            return [[1., 0., 0.] for _ in texts]
        async def aclose(self):
            closed.append(True)
    monkeypatch.setattr(cli, 'OpenAICompatibleEmbedder', Embedder)

    def handler(request):
        body = json.loads(request.content)
        overrides = {'relevance': 'relevant'} if 'relevance' in body['questions'] else {}
        return httpx.Response(200, json=response_for(body['questions'], overrides))
    monkeypatch.setattr(cli, 'JevClient', lambda config: JevClient(config, transport=httpx.MockTransport(handler)))

    async def command(*args):
        return await cli.run(cli.parser().parse_args(list(args)))

    captured = await command('capture', '--repo', f'{inputs[0].repo_id}=HEAD', '--repo', f'{inputs[1].repo_id}=HEAD')
    sid = captured['id']
    indexed = await command('index', '--snapshot', sid)
    common = ('--snapshot', sid, '--index', indexed['index_version'])
    assert (await command('embed', *common))['embedded_inputs'] > 0 and closed
    retrieval = await command('retrieve', *common)
    preparation = await command('prepare', '--run', retrieval['id'], '--target-limit', '5')
    decisions = await command('assess', '--packet', preparation['packet_ids'][0])
    async with pool.acquire() as conn:
        supported = await conn.fetchval("SELECT id FROM connection_decisions WHERE id=ANY($1::uuid[]) AND outcome='supported' LIMIT 1",
                                        [UUID(d) for d in decisions['decision_ids']])
        assert await conn.fetchval("SELECT EXISTS(SELECT 1 FROM connection_decisions WHERE id=ANY($1::uuid[]) AND outcome='insufficient_evidence')",
                                   [UUID(d) for d in decisions['decision_ids']])
    published = await command('publish', '--snapshot', sid, '--decision', str(supported))
    view = await command('show', '--revision', published['revision_id'])
    source = view['graph']['edges'][0]['source']
    neighbors = await command('neighbors', '--snapshot', sid, '--node', source)
    assert len(neighbors['graph']['edges']) == 1
    with pytest.raises(ValueError, match='two distinct'):
        await command('capture', '--repo', f'{inputs[0].repo_id}=HEAD')
