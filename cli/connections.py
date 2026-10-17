"""Explicit publication and exploration of assessed cross-repository graphs."""

import argparse
import asyncio
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import sys

from core.connections.assessment import assess_packet
from core.connections.jev import JevClient, JevConfig, JevReranker
from core.connections.snapshots import RepositoryInput, capture_snapshot
from core.connections.sites import index_snapshot_sites
from core.connections.retrieval import embed_snapshot_sites, retrieve_all_candidates
from core.connections.preparation import prepare_candidates, PreparationOptions
from core.embedding_identity import EmbeddingIdentity
from core.embedder import EmbedderConfig, OpenAICompatibleEmbedder
from core.connections.publication import publish_graph
from core.connections.published_queries import neighborhood, read_graph
from db.connection import pool_ctx


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture', help='Capture committed source from at least two registered repositories')
    capture.add_argument('--repo', action='append', required=True, metavar='ID=REVISION')
    capture.add_argument('--context', default='{}', help='JSON configuration context')
    index = commands.add_parser('index', help='Extract sites from captured bytes')
    index.add_argument('--snapshot', required=True)
    for name in ('embed', 'retrieve'):
        command = commands.add_parser(name, help=('Embed site inputs (makes paid model calls)' if name == 'embed'
                                                  else 'Retrieve cross-repository candidates from cached vectors'))
        command.add_argument('--snapshot', required=True)
        command.add_argument('--index', required=True)
    prepare = commands.add_parser('prepare', help='Rerank with Jev and assemble evidence (makes paid model calls)')
    prepare.add_argument('--run', required=True, help='Retrieval run ID')
    prepare.add_argument('--target-limit', type=int, choices=range(1, 6), default=5)
    prepare.add_argument('--context-path', action='append', default=[], metavar='REPO_ID=PATH')
    assess = commands.add_parser('assess', help='Assess a prepared packet through Jev (makes paid model calls)')
    assess.add_argument('--packet', required=True)
    publish = commands.add_parser('publish', help='Atomically replace active membership with the selected decisions')
    publish.add_argument('--snapshot', required=True)
    selection = publish.add_mutually_exclusive_group(required=True)
    selection.add_argument('--decision', action='append')
    selection.add_argument('--empty', action='store_true', help='Publish an explicitly empty cross-repository graph')
    publish.add_argument('--parser-branch', type=int, action='append', default=[])
    for name in ('show', 'neighbors'):
        command = commands.add_parser(name)
        selector = command.add_mutually_exclusive_group(required=True)
        selector.add_argument('--snapshot', help='Read the active revision for this snapshot')
        selector.add_argument('--revision', help='Read an immutable historical revision')
        if name == 'neighbors':
            command.add_argument('--node', required=True, help='Node key from show, such as site:<UUID>')
            command.add_argument('--hops', type=int, default=1)
            command.add_argument('--direction', choices=('both', 'out', 'in'), default='both')
            command.add_argument('--include-parser', action='store_true')
            command.add_argument('--max-nodes', type=int, default=200)
    return root


async def run(args):
    async with pool_ctx() as pool:
        if args.command == 'capture':
            repositories = []
            for spec in args.repo:
                repo, separator, revision = spec.partition('=')
                if not separator or not revision:
                    raise ValueError('Repository syntax is ID=REVISION')
                async with pool.acquire() as conn:
                    root = await conn.fetchval('SELECT root_path FROM repos WHERE id=$1', int(repo))
                if not root:
                    raise ValueError('Repository must have a registered checkout root')
                repositories.append(RepositoryInput(int(repo), Path(root), revision))
            context = json.loads(args.context)
            if not isinstance(context, dict):
                raise ValueError('Context must be a JSON object')
            return asdict(await capture_snapshot(pool, tuple(repositories), context=context))
        if args.command == 'index':
            return asdict(await index_snapshot_sites(pool, args.snapshot))
        if args.command == 'embed':
            embedder = OpenAICompatibleEmbedder(replace(EmbedderConfig.from_env(), representation_version='site-v1'))
            try:
                count = await embed_snapshot_sites(pool, args.snapshot, args.index, embedder.embed)
            finally:
                await embedder.aclose()
            return {'embedded_inputs': count}
        if args.command == 'retrieve':
            model = os.environ.get('EMBEDDING_MODEL', '')
            identity = EmbeddingIdentity(model, os.environ.get('EMBEDDING_MODEL_VERSION') or model,
                                         int(os.environ.get('EMBEDDING_DIM', '4096')), 'site-v1')
            return asdict(await retrieve_all_candidates(pool, args.snapshot, args.index, identity))
        if args.command == 'prepare':
            reranker = JevReranker(JevClient(JevConfig.from_env()))
            paths = []
            for spec in args.context_path:
                repo, separator, path = spec.partition('=')
                if not separator or not path:
                    raise ValueError('Context path syntax is REPO_ID=PATH')
                paths.append((int(repo), path))
            run = await prepare_candidates(pool, args.run, reranker.rerank, reranker.identity,
                PreparationOptions(target_limit=args.target_limit, timeout_seconds=180), additional_paths=tuple(paths))
            async with pool.acquire() as conn:
                from uuid import UUID
                packets = await conn.fetch('SELECT id FROM connection_evidence_packets WHERE preparation_run_id=$1 ORDER BY id', UUID(run.id))
            return {**asdict(run), 'packet_ids': [str(p['id']) for p in packets]}
        if args.command == 'assess':
            return {'decision_ids': await assess_packet(pool, args.packet, JevClient(JevConfig.from_env()))}
        if args.command == 'publish':
            return {'revision_id': await publish_graph(pool, args.snapshot, args.decision or (), branch_ids=args.parser_branch)}
        result = await read_graph(pool, snapshot_id=args.snapshot, revision_id=args.revision)
        if args.command == 'neighbors':
            result['graph'] = neighborhood(result['graph'], args.node, hops=args.hops,
                direction=args.direction, include_parser=args.include_parser, max_nodes=args.max_nodes)
        return result


def main():
    args = parser().parse_args()
    try:
        result = asyncio.run(run(args))
    except (ValueError, RuntimeError) as error:
        print(f'connections: {error}', file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
