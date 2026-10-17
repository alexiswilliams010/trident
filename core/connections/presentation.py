"""Endpoint-first evidence with descriptive labels instead of keys."""
from copy import deepcopy
import json
import re

from .inference import plaintext


def fenced(code):
    fence = '`' * max(3, 1 + max((len(m.group()) for m in re.finditer(r'`+', code)), default=0))
    return fence + '\n' + code + '\n' + fence


def describe(item):
    if item.get('is_endpoint') is False:
        return 'This is a retrieved passage, not an identified endpoint.'
    name = item.get('name')
    if item['kind'] == 'call':
        return f'This expression calls {name}.' if name else 'This endpoint is a call expression.'
    if item['kind'] == 'definition':
        return f'This endpoint is the definition of {name}.' if name else 'This endpoint is a definition.'
    if item['kind'] == 'configuration':
        return 'This endpoint is the selected configuration passage.'
    return 'This endpoint is the selected declaration passage.'


def explanatory_plaintext(state, *, repository_labels=None):
    data = json.loads(state)
    sections = [plaintext(state).split('\n\nSOURCE\n', 1)[0]]
    for label, title in [('source', 'SOURCE'), ('target', 'CANDIDATE TARGET')]:
        item = data[label]
        repo = (repository_labels or {}).get(label, item.get('repository', label + ' repository'))
        explanation = ('This is the exact code location being investigated for a cross-repository interaction.'
                       if label == 'source' else
                       'This is the exact code location proposed as the other endpoint of that interaction. '
                       'Its relationship to the source is not yet established.')
        if item.get('is_endpoint') is False:
            explanation = 'This retrieved passage may contain a relevant endpoint, but no precise endpoint has been identified within it.'
        sections.extend(['', title, explanation, f"{repo} · {item['path']}:{item['line']}",
                         'Retrieved passage:' if item.get('is_endpoint') is False else 'Exact code:',
                         fenced(item['exact_code']), describe(item)])
        if item['exact_code_clipped']:
            sections.append('This exact code excerpt is clipped; part of the selected endpoint is omitted.')
    for label in ('source', 'target'):
        item = data[label]
        sections.extend(['', label.upper() + ' CONTEXT',
                         f'The following code surrounds the {"source" if label == "source" else "candidate target"} endpoint. '
                         'It can explain what the endpoint refers to, including imports, bindings, and configuration.'])
        if 'is_endpoint' in item:
            from .evidence_context import render_context
            sections.extend(render_context(item, fenced))
        elif item['scope']:
            sections.append('The endpoint is inside ' + ', which is inside '.join(reversed(item['scope'])) + '.')
            if label == 'target':
                sections.append('A connection to that enclosing code alone does not establish a direct connection '
                                'to the selected endpoint inside it.')
        for key, title in [('context_before', 'Preceding context'), ('context_after', 'Following context')]:
            sections.extend([title + ':', fenced(item[key])])
    for item in data.get('additional_context', []):
        sections.extend(['', item['label'].upper(), f"Additional evidence from {item['path']}:", fenced(item['text'])])
        if item['clipped']:
            sections.append('This additional excerpt is clipped.')
    return '\n'.join(sections)


def identity(item):
    location = f"{item['path']}:{item['line']}"
    if item.get('is_endpoint') is False:
        return f'retrieved passage at {location} (endpoint not identified)'
    return f"{item['kind']} {item.get('name') or '(unnamed)'} at {location}"


def comparison(state, questions, *, repository_labels=None):
    data = json.loads(state)
    source, target = data['source'], data['target']
    name = source.get('name') or ''
    root_name = re.split(r'[.:]', name)[0]
    bindings = [b['text'] for b in source.get('bindings', [])
                if root_name and root_name in re.findall(r'[A-Za-z_$][\w$]*', b['text'])]
    enclosing = [f"{c['kind']} {c['name']}" + (f" at line {c['line']}" if 'line' in c else '')
                 for c in target.get('containment', [])]
    enclosing_text = ' > '.join(enclosing) or '(no enclosing definition recorded)'
    facts = ['ENDPOINT COMPARISON', 'Source: ' + identity(source),
             'Source binding statements mentioning its called name (not resolved destinations):',
             *(bindings or ['(none selected)']),
             'Selected target: ' + identity(target), 'Target is inside: ' + enclosing_text]
    plain = explanatory_plaintext(state, repository_labels=repository_labels)
    policy, rest = plain.split('\n\nSOURCE\n', 1)
    rendered = policy + '\n\n' + '\n'.join(facts) + '\n\nSOURCE\n' + rest
    updated = deepcopy(questions)
    updated['condition']['instructions'] = (
        f"Does the binding or destination of {identity(source)} identify {identity(target)} itself, "
        f"its enclosing definition ({enclosing_text}), or another destination?")
    updated['condition']['criteria']['exact_target'] = f"The selected target itself: {identity(target)}."
    updated['condition']['criteria']['intermediate'] = (
        f"An enclosing definition ({enclosing_text}) or another intermediate operation, "
        'rather than the selected target itself.')
    return rendered, updated
