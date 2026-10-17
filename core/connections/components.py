"""Deterministic component membership and evidence-preserving graph projection."""

from dataclasses import asdict, dataclass
import json
import re
from uuid import UUID

from .published_queries import read_graph
from .snapshots import fingerprint

PROJECTION_VERSION = 'components-v1'


@dataclass(frozen=True)
class ComponentRule:
    repo_id: int
    key: str
    label: str
    prefix: str = ''

    def __post_init__(self):
        if (type(self.repo_id) is not int or self.repo_id <= 0 or not isinstance(self.key, str)
                or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', self.key)):
            raise ValueError('Component rules require a positive repository ID and a simple key')
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 200:
            raise ValueError('Component labels must contain 1 to 200 characters')
        if (not isinstance(self.prefix, str) or self.prefix.startswith('/') or '\\' in self.prefix
                or (self.prefix != '' and any(part in {'.', '..', ''} for part in self.prefix.split('/')))):
            raise ValueError('Component prefixes must be canonical repository-relative paths')


def project_components(graph, rules=()):
    """Project only published cross-repository interactions; never infer new ones."""
    repos = {m['repo_id'] for m in graph['members']}
    if len(repos) < 2:
        raise ValueError('Component connections require at least two repositories')
    rules = sorted(set(rules), key=lambda r: (r.repo_id, r.prefix, r.key, r.label))
    prefixes, labels = {}, {}
    for rule in rules:
        if rule.repo_id not in repos:
            raise ValueError('Component rule repository is outside the graph')
        prefix_key, component_key = (rule.repo_id, rule.prefix), (rule.repo_id, rule.key)
        if prefix_key in prefixes:
            raise ValueError('A component prefix has conflicting assignments')
        if component_key in labels and labels[component_key] != rule.label:
            raise ValueError('One component key cannot have conflicting labels')
        prefixes[prefix_key], labels[component_key] = rule.key, rule.label
    result = {'version': PROJECTION_VERSION, 'snapshot_id': graph['snapshot_id'],
              'rules': [asdict(r) for r in rules], 'nodes': {}, 'edges': [],
              'memberships': {}, 'unused_rules': [], 'excluded_parser_edges': 0}
    used = set()
    for node_id, node in sorted(graph['nodes'].items()):
        if node_id.startswith('resource:'):
            result['nodes'][node_id] = {'kind': 'resource', 'identity_scope': node['identity_scope']}
            continue
        matches = [r for r in rules if r.repo_id == node['repo_id'] and
                   (not r.prefix or node['path'] == r.prefix or node['path'].startswith(r.prefix + '/'))]
        if matches:
            rule = max(matches, key=lambda r: len(r.prefix))
            used.add(rule)
            component_id = f'component:{rule.repo_id}:named:{rule.key}'
            label, prefix = rule.label, rule.prefix
        else:
            component_id = f"component:{node['repo_id']}:repository"
            label, prefix = f"Repository {node['repo_id']}", None
        result['memberships'][node_id] = {'component_id': component_id, 'matched_prefix': prefix}
        component = result['nodes'].setdefault(component_id, {'kind': 'component', 'repo_id': node['repo_id'],
                                                              'label': label, 'member_count': 0})
        component['member_count'] += 1
    decisions = {d['id']: d for d in graph['decisions']}
    grouped = {}
    for index, edge in enumerate(graph['edges']):
        if edge['origin'] != 'cross_repository':
            result['excluded_parser_edges'] += 1
            continue
        source = result['memberships'][edge['source']]['component_id']
        target = (edge['target'] if edge['target'].startswith('resource:') else
                  result['memberships'][edge['target']]['component_id'])
        if source == target:
            raise ValueError('A cross-repository connection collapsed into one component')
        decision = decisions[edge['decision_id']]
        conditions = decision['conditions']
        key = fingerprint([source, target, edge['verb'], edge['vocabulary_version'], conditions])
        aggregate = grouped.setdefault(key, {'id': key, 'source': source, 'target': target,
            'verb': edge['verb'], 'vocabulary_version': edge['vocabulary_version'], 'conditions': conditions,
            'connection_indices': [], 'decision_ids': []})
        aggregate['connection_indices'].append(index)
        aggregate['decision_ids'].append(edge['decision_id'])
    for key, edge in sorted(grouped.items()):
        edge['decision_ids'] = sorted(set(edge['decision_ids']))
        edge['connection_count'] = len(edge['connection_indices'])
        edge['decision_count'] = len(edge['decision_ids'])
        result['edges'].append(edge)
    result['unused_rules'] = [asdict(r) for r in rules if r not in used]
    return result


async def create_projection(pool, graph_revision_id, rules=()):
    view = await read_graph(pool, revision_id=graph_revision_id)
    projection = project_components(view['graph'], rules)
    projection['graph_revision_id'] = view['revision_id']
    projection['graph_fingerprint'] = view['fingerprint']
    digest = fingerprint(projection)
    async with pool.acquire() as conn, conn.transaction():
        pid = await conn.fetchval('''INSERT INTO connection_component_projections
            (snapshot_id,graph_revision_id,fingerprint,projection) VALUES ($1,$2,$3,$4::jsonb)
            ON CONFLICT(graph_revision_id,fingerprint) DO NOTHING RETURNING id''',
            UUID(projection['snapshot_id']), UUID(view['revision_id']), digest, json.dumps(projection))
        if pid is None:
            pid = await conn.fetchval('''SELECT id FROM connection_component_projections
                WHERE graph_revision_id=$1 AND fingerprint=$2''', UUID(view['revision_id']), digest)
        return str(pid)


async def read_projection(pool, projection_id):
    async with pool.acquire() as conn:
        row = await conn.fetchrow('SELECT * FROM connection_component_projections WHERE id=$1', UUID(str(projection_id)))
        if row is None:
            raise ValueError('Unknown component projection')
        return {'projection_id': str(row['id']), 'fingerprint': row['fingerprint'],
                'projection': json.loads(row['projection'])}


async def explain_component_edge(pool, projection_id, edge_id):
    projection = (await read_projection(pool, projection_id))['projection']
    edge = next((e for e in projection['edges'] if e['id'] == edge_id), None)
    if edge is None:
        raise ValueError('Unknown component edge')
    view = await read_graph(pool, revision_id=projection['graph_revision_id'])
    graph = view['graph']
    connections = [graph['edges'][i] for i in edge['connection_indices']]
    decision_connections = [e for e in graph['edges'] if e.get('decision_id') in edge['decision_ids']]
    node_ids = {e[k] for e in decision_connections for k in ('source', 'target')}
    return {'projection_id': str(projection_id), 'graph_revision_id': view['revision_id'], 'edge': edge,
            'connections': connections, 'decision_connections': decision_connections,
            'nodes': {key: graph['nodes'][key] for key in sorted(node_ids)},
            'decisions': [d for d in graph['decisions'] if d['id'] in edge['decision_ids']]}


def component_neighborhood(projection, component_id, *, hops=1, direction='both', max_nodes=200):
    from .published_queries import neighborhood
    graph = {'nodes': projection['nodes'], 'decisions': [],
             'edges': [{**e, 'origin': 'component_projection'} for e in projection['edges']]}
    result = neighborhood(graph, component_id, hops=hops, direction=direction, max_nodes=max_nodes)
    result.pop('decisions')
    result['graph_revision_id'] = projection.get('graph_revision_id')
    return result
