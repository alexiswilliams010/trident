import json
from pathlib import Path

from core.connections.assessment import build_questions, interpret
from core.connections.inference import prepare
from tests.test_connection_assessment import minimal_packet, response_for


def test_production_matches_evaluated_question():
    raw = b'send()'
    site = dict(id='site', path='code.py', kind='call', start_byte=0, end_byte=len(raw), metadata={'name': 'send'})
    _, questions, _ = prepare(site, raw, site, raw)
    path = Path(__file__).with_name('direct_indirect_question.json')
    # Freeze the tested question locally rather than relying on live providers.
    assert questions['topology'] == json.loads(path.read_text())
    assert set(questions['topology']['criteria']) == {'forward', 'reverse', 'indirect', 'shared_resource', 'none'}


def test_indirect_is_retained_without_graph_edges():
    from core.connections.questions import relationship_questions
    packet = minimal_packet()
    candidate = {'source_site_id': packet['excerpts'][0]['site_id'], 'target_site_id': packet['excerpts'][1]['site_id']}
    questions = relationship_questions(build_questions(packet))
    proposal = interpret(packet, candidate, response_for(questions, {'topology': 'indirect'})['answers'])
    assert proposal['relationship'] == 'indirect'
    assert proposal['outcome'] == 'insufficient_evidence'
    assert proposal['abstention_reasons'] == ['indirect_relationship']
    assert proposal['edges'] == []
