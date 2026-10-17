"""Jev wire contracts, abstention policy, and database provenance without live API calls."""

import json
from uuid import UUID

import httpx
import pytest

from core.connections.assessment import assess_packet, build_questions, interpret
from core.connections.jev import JevClient, JevConfig, validate_answers
from core.connections.preparation import prepare_candidates, RerankerIdentity, PreparationOptions
from tests.test_connection_retrieval import corpus as _corpus
from tests.test_connection_preparation import retrieved

corpus = _corpus
MODEL = 'typesafe-ai/jev-1.14.0'


def response_for(questions, overrides=None):
    defaults = {'outcome': 'supported', 'topology': 'forward', 'condition': 'established',
                'direct_verb': 'INVOKES', 'source_verb': 'PRODUCES', 'target_verb': 'CONSUMES',
                'resource_anchor': 'none'}
    defaults.update(overrides or {})
    answers = {}
    for key, question in questions.items():
        selected = defaults.get(key, 'support')
        answers[key] = {'type': 'choice', 'choice': selected, 'confidence': .9,
                        'probabilities': {option: float(option == selected) for option in question['criteria']}}
    return {'model': MODEL, 'answers': answers, 'usage': {'input_tokens': 12, 'output_tokens': 3}}


def minimal_packet():
    return {'excerpts': [dict(evidence_id=k, site_id=site, path='code.py', start_byte=0, end_byte=4)
                         for k, site in [('a', 'source'), ('b', 'target')]]}


@pytest.mark.parametrize('overrides,outcome,reason', [
    ({}, 'supported', None),
    ({'evidence_1': 'context'}, 'insufficient_evidence', 'missing_endpoint_support'),
    ({'evidence_1': 'contradiction'}, 'ambiguous', 'contradictory_evidence'),
    ({'condition': 'unresolved'}, 'insufficient_evidence', 'unresolved_conditions'),
    ({'topology': 'none'}, 'insufficient_evidence', 'missing_interaction_shape_or_verb'),
    ({'outcome': 'unsupported'}, 'unsupported', None),
    ({'outcome': 'ambiguous'}, 'ambiguous', None),
    ({'outcome': 'insufficient_evidence'}, 'insufficient_evidence', None),
])
def test_abstention_policy(overrides, outcome, reason):
    packet = minimal_packet()
    questions = build_questions(packet)
    response = response_for(questions, overrides)
    validate_answers(response, questions, MODEL)
    proposal = interpret(packet, dict(source_site_id='source', target_site_id='target'), response['answers'])
    assert proposal['outcome'] == outcome
    if reason:
        assert reason in proposal['abstention_reasons']
    assert bool(proposal['edges']) == (outcome == 'supported')


def test_reverse_and_resource_verbs():
    packet = minimal_packet()
    candidate = dict(source_site_id='source', target_site_id='target')
    questions = build_questions(packet)
    reverse = interpret(packet, candidate, response_for(questions, {'topology': 'reverse'})['answers'])
    assert reverse['edges'][0]['source_site_id'] == 'target'
    shared = interpret(packet, candidate, response_for(questions, {
        'topology': 'shared_resource', 'resource_anchor': 'a'})['answers'])
    assert [e['verb'] for e in shared['edges']] == ['PRODUCES', 'CONSUMES']
    assert shared['resource_identity'] == 'requires_materialization'
    assert 'EMITS' not in questions['source_verb']['criteria']
    assert 'CONSUMES' not in questions['direct_verb']['criteria']


@pytest.mark.parametrize('mutation', ['model', 'missing', 'unknown', 'nan', 'sum', 'choice'])
def test_reject_malformed_answers(mutation):
    questions = build_questions(minimal_packet())
    result = response_for(questions)
    if mutation == 'model':
        result['model'] = 'other-model'
    elif mutation == 'missing':
        del result['answers']['outcome']
    elif mutation == 'unknown':
        result['answers']['outcome']['choice'] = 'invented'
    elif mutation == 'nan':
        result['answers']['outcome']['confidence'] = float('nan')
    elif mutation == 'sum':
        result['answers']['outcome']['probabilities']['supported'] = .1
    else:
        result['answers']['outcome']['choice'] = 'unsupported'
    with pytest.raises(ValueError):
        validate_answers(result, questions, MODEL)


async def test_gateway_wire_request_and_errors():
    questions = build_questions(minimal_packet())
    def handler(request):
        assert str(request.url) == 'https://ai-gateway.vercel.sh/typesafe/v1/systemone'
        assert request.headers['authorization'] == 'Bearer fixture-secret'
        body = json.loads(request.content)
        assert body == {'model': MODEL, 'state': 'fixture', 'questions': questions}
        return httpx.Response(200, json=response_for(questions))
    config = JevConfig('fixture-secret', retries=0)
    assert 'fixture-secret' not in repr(config)
    client = JevClient(config, transport=httpx.MockTransport(handler))
    assert (await client.decide('fixture', questions))['model'] == MODEL
    broken = JevClient(config, transport=httpx.MockTransport(lambda r: httpx.Response(401, text='secret body')))
    with pytest.raises(RuntimeError, match='HTTP 401') as error:
        await broken.decide('fixture', questions)
    assert 'secret body' not in str(error.value)
    with pytest.raises(ValueError, match='Pin'):
        JevConfig('key', model='typesafe-ai/jev')


class FixtureClient:
    identity = {'provider': 'fixture', 'model_version': MODEL, 'wire_version': 'typesafe-choice-v1'}
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    async def decide(self, state, questions):
        self.calls += 1
        assert json.loads(state)['focus']['candidate_id']
        if self.fail and self.calls == 2:
            raise RuntimeError('fixture failed')
        return response_for(questions)


async def prepared(corpus):
    pool, inputs, snapshot, retrieval = await retrieved(corpus)
    async def rerank(query, docs):
        return [1.] * len(docs)
    run = await prepare_candidates(pool, retrieval.id, rerank, RerankerIdentity('fixture', 'v1'),
                                   PreparationOptions(target_limit=2),
                                   additional_paths=((inputs[0].repo_id, 'missing.conf'),))
    async with pool.acquire() as conn:
        packet = await conn.fetchval('SELECT id FROM connection_evidence_packets WHERE preparation_run_id=$1 LIMIT 1', UUID(run.id))
    return pool, snapshot, packet


async def test_persist_reuse_and_dependencies(corpus):
    pool, snapshot, packet = await prepared(corpus)
    client = FixtureClient()
    ids = await assess_packet(pool, packet, client)
    assert len(ids) == 2 and client.calls == 2
    assert await assess_packet(pool, packet, client) == ids
    assert client.calls == 2
    async with pool.acquire() as conn:
        for did in ids:
            assert await conn.fetchval('SELECT outcome FROM connection_decisions WHERE id=$1', UUID(did)) == 'supported'
            assert await conn.fetchval('SELECT count(*) FROM connection_decision_inputs WHERE decision_id=$1', UUID(did)) == 1
            assert await conn.fetchval("SELECT count(*) FROM connection_decision_dependencies WHERE decision_id=$1 AND path='missing.conf' AND content_hash IS NULL", UUID(did)) == 1
            proposal = json.loads(await conn.fetchval('SELECT proposal FROM connection_assessments WHERE decision_id=$1', UUID(did)))
            assert proposal['edges'][0]['verb'] == 'INVOKES'
        assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM cross_connections WHERE snapshot_id=$1)', snapshot.id)


async def test_failed_packet_is_atomic(corpus):
    pool, snapshot, packet = await prepared(corpus)
    with pytest.raises(RuntimeError, match='fixture failed'):
        await assess_packet(pool, packet, FixtureClient(fail=True))
    async with pool.acquire() as conn:
        assert not await conn.fetchval('SELECT EXISTS(SELECT 1 FROM connection_decisions WHERE snapshot_id=$1)', snapshot.id)
        await conn.execute("UPDATE connection_evidence_packets SET payload=jsonb_set(payload,'{empty_candidates}','true') WHERE id=$1", packet)
    client = FixtureClient()
    with pytest.raises(ValueError, match='fingerprint'):
        await assess_packet(pool, packet, client)
    assert client.calls == 0



async def test_gateway_retries_transient_failure(monkeypatch):
    import core.connections.jev as module
    questions = build_questions(minimal_packet())
    attempts = []
    delays = []

    async def sleep(delay):
        delays.append(delay)

    def handler(request):
        attempts.append(request)
        return httpx.Response(429) if len(attempts) == 1 else httpx.Response(200, json=response_for(questions))

    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    client = JevClient(JevConfig('fixture'), transport=httpx.MockTransport(handler))
    await client.decide('fixture', questions)
    assert len(attempts) == 2 and delays == [1]


async def test_citation_site_mismatch_rejected_before_model(corpus):
    from core.connections.snapshots import fingerprint
    pool, _, pid = await prepared(corpus)
    async with pool.acquire() as conn:
        packet = json.loads(await conn.fetchval('SELECT payload FROM connection_evidence_packets WHERE id=$1', pid))
        packet['excerpts'][0]['site_id'] = str(UUID(int=0))
        await conn.execute('UPDATE connection_evidence_packets SET payload=$2::jsonb,packet_fingerprint=$3 WHERE id=$1',
                           pid, json.dumps(packet), fingerprint(packet))
    client = FixtureClient()
    with pytest.raises(ValueError, match='claimed site'):
        await assess_packet(pool, pid, client)
    assert client.calls == 0


async def test_changed_proposal_invalidates_reuse(corpus):
    from core.connections.dependencies import ResolutionVersions, assess_decision_reuse
    pool, snapshot, packet = await prepared(corpus)
    did = UUID((await assess_packet(pool, packet, FixtureClient()))[0])
    async with pool.acquire() as conn:
        versions = ResolutionVersions(**json.loads(await conn.fetchval(
            'SELECT versions FROM connection_decision_inputs WHERE decision_id=$1', did)))
        assert (await assess_decision_reuse(conn, did, snapshot.id, versions)).reusable
        await conn.execute("UPDATE connection_assessments SET proposal='{}' WHERE decision_id=$1", did)
        assert 'decision_or_evidence_changed' in (await assess_decision_reuse(conn, did, snapshot.id, versions)).reasons
