"""CLI: index a repository (Tier 1 extractor + Tier 2 semantic resolver).

Usage:
    python -m cli.index <repo_path> --repo-id 1 [--dsn postgresql://...] [--init-schema]
                        [--no-resolve]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.extractor import index_repo
from core.semantic_resolver import resolve_repo
from db.connection import apply_migrations, pool_ctx


async def _run(
    repo_path: Path,
    repo_id: int,
    dsn: str | None,
    init_schema: bool,
    do_resolve: bool,
) -> int:
    async with pool_ctx(dsn) as pool:
        if init_schema:
            async with pool.acquire() as conn:
                applied = await apply_migrations(conn)
                if applied:
                    print(f"Applied migrations: {', '.join(applied)}")

        extract_result = await index_repo(pool, repo_id, repo_path)
        print(
            f"[Tier 1] Indexed {len(extract_result.indexed)} files "
            f"({extract_result.total_nodes} nodes); skipped {len(extract_result.skipped)} unchanged"
        )

        if do_resolve:
            # Resolve only the files we just (re)indexed; unchanged files keep
            # their existing semantic rows.
            file_ids = [r.file_id for r in extract_result.indexed]
            if file_ids:
                resolve_results = await resolve_repo(pool, repo_id, only_file_ids=file_ids)
                tot_defs = sum(r.n_definitions for r in resolve_results)
                tot_refs = sum(r.n_references for r in resolve_results)
                tot_calls = sum(r.n_call_edges for r in resolve_results)
                tot_da = sum(r.n_data_access for r in resolve_results)
                print(
                    f"[Tier 2] {len(resolve_results)} files: "
                    f"{tot_defs} definitions, {tot_refs} references, "
                    f"{tot_calls} call_edges, {tot_da} data_access"
                )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsgrep — index a repo (Tier 1 + Tier 2)")
    parser.add_argument("repo_path", type=Path, help="Path to the repo to index")
    parser.add_argument("--repo-id", type=int, required=True)
    parser.add_argument("--dsn", type=str, default=None, help="Postgres DSN (defaults to DATABASE_URL env)")
    parser.add_argument("--init-schema", action="store_true", help="Apply migrations before indexing (idempotent)")
    parser.add_argument("--no-resolve", action="store_true", help="Skip Tier 2 semantic resolution")
    args = parser.parse_args(argv)

    if not args.repo_path.is_dir():
        print(f"error: {args.repo_path} is not a directory", file=sys.stderr)
        return 2

    return asyncio.run(_run(args.repo_path, args.repo_id, args.dsn, args.init_schema, not args.no_resolve))


if __name__ == "__main__":
    raise SystemExit(main())
