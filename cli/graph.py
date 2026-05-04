"""CLI: graph exploration commands for agentic workflows.

    python -m cli.graph --repo-name myrepo callers-of withdraw
    python -m cli.graph --repo-name myrepo ancestors withdraw --json
    python -m cli.graph --repo-name myrepo paths withdraw deposit
    python -m cli.graph --repo-name myrepo entrypoints --kind function
    python -m cli.graph --repo-name myrepo source withdraw --json

All queries run with a configurable statement timeout (default 120s).
Pass --json for structured output suitable for LLM agent consumption.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from cli._output import (
    def_to_json,
    emit,
    format_def,
    format_import,
    format_inheritance_node,
    import_to_json,
    inheritance_node_to_json,
)
from cli._repo import resolve_repo_id, resolve_repo_ids
from core.graph import (
    ancestors,
    callers_of,
    callees_of,
    entrypoint_paths,
    entrypoints,
    file_dependents,
    file_imports,
    get_source,
    inheritance_tree,
    paths_between,
    reachable_from,
    resolve_definitions,
)
from db.connection import pool_ctx


async def _resolve_repo_ids(pool, args: argparse.Namespace) -> list[int]:
    if args.repos:
        names = [s.strip() for s in args.repos.split(",") if s.strip()]
        return await resolve_repo_ids(pool, names)
    repo_id = await resolve_repo_id(pool, name=args.repo_name, create=False)
    return [repo_id]


def _repo_label(args: argparse.Namespace) -> str:
    return args.repos if args.repos else args.repo_name


# ────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ────────────────────────────────────────────────────────────────────


async def _cmd_resolve(pool, rids, args) -> int:
    defs = await resolve_definitions(
        pool, rids, args.name, kind=args.kind, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "resolve", "target": args.name, "repo": _repo_label(args),
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        print(f'definitions matching "{args.name}" ({len(defs)} results):')
        for d in defs:
            print(format_def(d))
    return 0


async def _cmd_callers_of(pool, rids, args) -> int:
    defs = await callers_of(
        pool, rids, args.name, confidence=args.confidence, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "callers-of", "target": args.name, "repo": _repo_label(args),
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        print(f'callers of "{args.name}" ({len(defs)} results):')
        for d in defs:
            print(format_def(d))
    return 0


async def _cmd_callees_of(pool, rids, args) -> int:
    defs = await callees_of(
        pool, rids, args.name, confidence=args.confidence, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "callees-of", "target": args.name, "repo": _repo_label(args),
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        print(f'callees of "{args.name}" ({len(defs)} results):')
        for d in defs:
            print(format_def(d))
    return 0


async def _cmd_ancestors(pool, rids, args) -> int:
    defs = await ancestors(
        pool, rids, args.name,
        max_depth=args.max_depth, confidence=args.confidence, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "ancestors", "target": args.name, "repo": _repo_label(args),
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        print(f'ancestors of "{args.name}" ({len(defs)} results):')
        for d in defs:
            print(format_def(d))
    return 0


async def _cmd_reachable(pool, rids, args) -> int:
    defs = await reachable_from(
        pool, rids, args.name,
        max_depth=args.max_depth, confidence=args.confidence, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "reachable", "target": args.name, "repo": _repo_label(args),
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        print(f'reachable from "{args.name}" ({len(defs)} results):')
        for d in defs:
            print(format_def(d))
    return 0


async def _cmd_paths(pool, rids, args) -> int:
    result = await paths_between(
        pool, rids, args.source_name, args.target_name,
        max_depth=args.max_depth, max_paths=args.max_paths, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "paths", "source": args.source_name, "target": args.target_name,
              "repo": _repo_label(args),
              "paths": [[def_to_json(d) for d in path] for path in result]}, True)
    else:
        print(f'paths from "{args.source_name}" to "{args.target_name}" ({len(result)} paths):')
        for i, path in enumerate(result, 1):
            names = " → ".join(d.qualified_name for d in path)
            print(f"  [{i}] {names}")
    return 0


async def _cmd_entrypoints(pool, rids, args) -> int:
    defs = await entrypoints(
        pool, rids, kind=args.kind, file_path=args.file,
        include_internal=args.include_internal, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "entrypoints", "repo": _repo_label(args), "target": args.kind or "*",
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        print(f"entrypoints ({len(defs)} results):")
        for d in defs:
            print(format_def(d))
    return 0


async def _cmd_entrypoint_paths(pool, rids, args) -> int:
    result = await entrypoint_paths(
        pool, rids, args.name,
        max_depth=args.max_depth, max_paths=args.max_paths, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "entrypoint-paths", "target": args.name, "repo": _repo_label(args),
              "paths": [[def_to_json(d) for d in path] for path in result]}, True)
    else:
        print(f'entrypoint paths to "{args.name}" ({len(result)} paths):')
        for i, path in enumerate(result, 1):
            names = " → ".join(d.qualified_name for d in path)
            print(f"  [{i}] {names}")
    return 0


async def _cmd_source(pool, rids, args) -> int:
    defs = await get_source(pool, rids, args.name, kind=args.kind, timeout_s=args.timeout)
    if args.json:
        emit({"command": "source", "target": args.name, "repo": _repo_label(args),
              "results": [def_to_json(d) for d in defs]}, True)
    else:
        for d in defs:
            print(f"── {d.qualified_name} ({d.kind})  {d.file_path}:{d.start_line}-{d.end_line} ──")
            print(d.source or "(no source)")
            print()
    return 0


async def _cmd_imports(pool, rids, args) -> int:
    imps = await file_imports(
        pool, rids, file_path=args.file, dep_class=args.dep_class, timeout_s=args.timeout,
    )
    if args.json:
        emit({"command": "imports", "repo": _repo_label(args), "file": args.file,
              "results": [import_to_json(i) for i in imps]}, True)
    else:
        label = f' for "{args.file}"' if args.file else ""
        print(f"imports{label} ({len(imps)} results):")
        for i in imps:
            print(format_import(i))
    return 0


async def _cmd_dependents(pool, rids, args) -> int:
    imps = await file_dependents(pool, rids, args.file_path, timeout_s=args.timeout)
    if args.json:
        emit({"command": "dependents", "repo": _repo_label(args), "file": args.file_path,
              "results": [import_to_json(i) for i in imps]}, True)
    else:
        print(f'files importing "{args.file_path}" ({len(imps)} results):')
        for i in imps:
            print(format_import(i))
    return 0


async def _cmd_inheritance(pool, rids, args) -> int:
    nodes = await inheritance_tree(pool, rids, args.name, timeout_s=args.timeout)
    if args.json:
        emit({"command": "inheritance", "target": args.name, "repo": _repo_label(args),
              "results": [inheritance_node_to_json(n) for n in nodes]}, True)
    else:
        print(f'inheritance tree for "{args.name}" ({len(nodes)} types):')
        for n in nodes:
            print(format_inheritance_node(n))
    return 0


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> int:
    async with pool_ctx(args.dsn) as pool:
        rids = await _resolve_repo_ids(pool, args)
        return await args.func(pool, rids, args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="trident — graph exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo_group = parser.add_mutually_exclusive_group(required=True)
    repo_group.add_argument("--repo-name", type=str, help="Repo name (must already be indexed)")
    repo_group.add_argument("--repos", type=str, help="Comma-separated repo names")
    parser.add_argument("--dsn", type=str, default=None)

    # Common flags shared by all subcommands via parents=
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Emit JSON output")
    common.add_argument("--confidence", type=str, default=None,
                        choices=["certain", "inferred", "uncertain"],
                        help="Filter call edges by confidence level")
    common.add_argument("--timeout", type=int, default=120,
                        help="Query timeout in seconds (default: 120)")

    subs = parser.add_subparsers(dest="command", required=True)

    # resolve
    p = subs.add_parser("resolve", parents=[common], help="Find definitions by name")
    p.add_argument("name", type=str)
    p.add_argument("--kind", type=str, default=None)
    p.set_defaults(func=_cmd_resolve)

    # callers-of
    p = subs.add_parser("callers-of", parents=[common], help="Direct callers of a function")
    p.add_argument("name", type=str)
    p.set_defaults(func=_cmd_callers_of)

    # callees-of
    p = subs.add_parser("callees-of", parents=[common], help="Direct callees of a function")
    p.add_argument("name", type=str)
    p.set_defaults(func=_cmd_callees_of)

    # ancestors
    p = subs.add_parser("ancestors", parents=[common],
                        help="Transitive callers (upward call-graph slice)")
    p.add_argument("name", type=str)
    p.add_argument("--max-depth", type=int, default=None, help="Optional depth limit")
    p.set_defaults(func=_cmd_ancestors)

    # reachable
    p = subs.add_parser("reachable", parents=[common], help="Transitive callees (blast radius)")
    p.add_argument("name", type=str)
    p.add_argument("--max-depth", type=int, default=None, help="Optional depth limit")
    p.set_defaults(func=_cmd_reachable)

    # paths
    p = subs.add_parser("paths", parents=[common],
                        help="All simple call paths between two definitions")
    p.add_argument("source_name", type=str)
    p.add_argument("target_name", type=str)
    p.add_argument("--max-depth", type=int, default=None, help="Optional depth limit")
    p.add_argument("--max-paths", type=int, default=50, help="Max paths to return (default: 50)")
    p.set_defaults(func=_cmd_paths)

    # entrypoints
    p = subs.add_parser("entrypoints", parents=[common],
                        help="Functions with no internal callers")
    p.add_argument("--kind", type=str, default=None, help="Filter by kind (function, method, ...)")
    p.add_argument("--file", type=str, default=None,
                   help="Filter by file path (substring match)")
    p.add_argument("--include-internal", action="store_true",
                   help="Include _prefixed, internal/private, and interface functions")
    p.set_defaults(func=_cmd_entrypoints)

    # entrypoint-paths
    p = subs.add_parser("entrypoint-paths", parents=[common],
                        help="Paths from entrypoints to a target")
    p.add_argument("name", type=str)
    p.add_argument("--max-depth", type=int, default=None, help="Optional depth limit")
    p.add_argument("--max-paths", type=int, default=50, help="Max paths to return (default: 50)")
    p.set_defaults(func=_cmd_entrypoint_paths)

    # source
    p = subs.add_parser("source", parents=[common],
                        help="Retrieve source code of definitions")
    p.add_argument("name", type=str)
    p.add_argument("--kind", type=str, default=None)
    p.set_defaults(func=_cmd_source)

    # imports
    p = subs.add_parser("imports", parents=[common], help="List import statements")
    p.add_argument("--file", type=str, default=None, help="Filter by file path")
    p.add_argument("--dep-class", type=str, default=None,
                   choices=["intra_repo", "external", "unresolved"])
    p.set_defaults(func=_cmd_imports)

    # dependents
    p = subs.add_parser("dependents", parents=[common],
                        help="Files that import a given file")
    p.add_argument("file_path", type=str)
    p.set_defaults(func=_cmd_dependents)

    # inheritance
    p = subs.add_parser("inheritance", parents=[common],
                        help="Inheritance hierarchy for a class")
    p.add_argument("name", type=str)
    p.set_defaults(func=_cmd_inheritance)

    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
