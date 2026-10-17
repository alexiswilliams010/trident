"""Presentation-only experiment preserves evidence and policy."""
from collections import Counter
import json

from benchmark.evaluations.ordered_plaintext import ordered_plaintext
from core.connections.inference import plaintext


def test_source_order_and_markers_preserve_content():
    endpoint = {'path': 'file.py', 'kind': 'call', 'name': 'send', 'scope': ['caller'],
                'line': 3, 'exact_code': 'send("```")', 'exact_code_clipped': True,
                'context_before': 'before\n', 'context_after': '\nafter'}
    state = json.dumps({'instructions': 'Only exact_code identifies the endpoint.',
                        'source': endpoint, 'target': endpoint,
                        'additional_context': [{'label': 'context_1', 'path': 'config',
                                                'text': 'topic=orders', 'clipped': False}]})
    control, ordered = plaintext(state), ordered_plaintext(state)
    markers = ('>>> BEGIN EXACT ENDPOINT <<<', '>>> END EXACT ENDPOINT <<<')
    assert Counter(control.split('\n')) == Counter(line for line in ordered.split('\n') if line not in markers)
    assert control.split('\n\nSOURCE')[0] == ordered.split('\n\nSOURCE')[0]
    for label in ('SOURCE', 'TARGET'):
        section = ordered.split('\n'+label+'\n')[1].split('\nTARGET\n')[0]
        assert section.index('Preceding context:') < section.index(markers[0]) < section.index('Exact code:')
        assert section.index('Exact code clipped: yes') < section.index(markers[1]) < section.index('Following context:')
    assert ordered.endswith(control.split('\n\nCONTEXT_1')[1])
