import json

from core.connections.evidence_context import enrich_endpoint
from core.connections.inference import endpoint, prepare, plaintext
from core.connections.sites import extract_file
from core.connections.assessment import interpret
from tests.test_connection_assessment import minimal_packet, response_for


def select(raw, name):
    site = next(s for s in extract_file('test.py', raw) if s.kind == 'call' and s.metadata['name'] == name)
    return dict(id='site', path='test.py', kind=site.kind, start_byte=site.start_byte,
                end_byte=site.end_byte, metadata=site.metadata)


def test_binding_outside_window_with_alias_and_containment():
    raw = ('def unrelated():\n    from wrong import send as deliver\n\n'
           'def outer():\n    from service import (\n        send as deliver,\n    )\n'
           + '    # padding\n' * 100 + '    deliver(payload)\n').encode()
    site = select(raw, 'deliver')
    old = endpoint(site, raw)
    assert 'from service' not in old['context_before']
    result = enrich_endpoint(old, site, raw)
    assert [b['text'] for b in result['bindings']] == ['from service import (\n        send as deliver,\n    )']
    assert result['containment'][0]['name'] == 'outer'
    for entry in result['bindings'] + result['containment']:
        assert raw[entry['start_byte']:entry['end_byte']].decode() == entry['text']
    state, _, _ = prepare(site, raw, site, raw, enrich=True)
    assert 'Known lexical containment' in plaintext(state)
    assert 'from service import (' in plaintext(state)


def test_passage_is_not_an_identified_endpoint():
    raw = b'def send():\n    pass\n'
    site = dict(id='site', path='test.py', kind='declaration', start_byte=0,
                end_byte=len(raw), metadata={'passage': True})
    state, _, packet = prepare(site, raw, site, raw, enrich=True)
    assert not json.loads(state)['target']['is_endpoint']
    assert 'Retrieved passage (endpoint not identified)' in plaintext(state)
    assert all(e['is_endpoint'] is False for e in packet['excerpts'])
    packet = minimal_packet()
    packet['excerpts'][1]['is_endpoint'] = False
    from core.connections.assessment import build_questions
    candidate = {'source_site_id': packet['excerpts'][0]['site_id'],
                 'target_site_id': packet['excerpts'][1]['site_id']}
    proposal = interpret(packet, candidate, response_for(build_questions(packet))['answers'])
    assert 'endpoint_not_identified' in proposal['abstention_reasons']
    assert not proposal['edges']


def test_binding_budget_does_not_silently_truncate_statement():
    raw = ('from service import send\ndef run():\n    value = "' + 'é' * 1500 + '"\n    send(value)\n').encode()
    site = select(raw, 'send')
    result = enrich_endpoint(endpoint(site, raw), site, raw)
    assert result['bindings_omitted'] == 1
    assert [b['text'] for b in result['bindings']] == ['from service import send']
