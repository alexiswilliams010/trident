"""CLI: index a repository (Tier 1).

Usage:
    python -m cli.index <repo_path> --repo-id 1 [--dsn postgresql://...]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.extractor import index_repo
from db.connection import apply_migrations, pool_ctx


async def _run(repo_path: Path, repo_id: int, dsn: str | None, init_schema: bool) -> int:
    async with pool_ctx(dsn) as pool:
        if init_schema:
            async with pool.acquire() as conn:
                applied = await apply_migrations(conn)
                if applied:
                    print(f"Applied migrations: {', '.join(applied)}")
        result = await index_repo(pool, repo_id, repo_path)

    print(f"Indexed {len(result.indexed)} files ({result.total_nodes} nodes)")
    if result.skipped:
        print(f"Skipped {len(result.skipped)} unchanged files")
    for r in result.indexed[:10]:
        print(f"  {r.language:8s}  {r.node_count:6d} nodes  {r.rel_path}")
    if len(result.indexed) > 10:
        print(f"  ... and {len(result.indexed) - 10} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsgrep — index a repo (Tier 1)")
    parser.add_argument("repo_path", type=Path, help="Path to the repo to index")
    parser.add_argument("--repo-id", type=int, required=True, help="Logical repo id")
    parser.add_argument("--dsn", type=str, default=None, help="Postgres DSN (defaults to DATABASE_URL env)")
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Apply db/schema.sql before indexing (idempotent)",
    )
    args = parser.parse_args(argv)

    if not args.repo_path.is_dir():
        print(f"error: {args.repo_path} is not a directory", file=sys.stderr)
        return 2

    return asyncio.run(_run(args.repo_path, args.repo_id, args.dsn, args.init_schema))


if __name__ == "__main__":
    raise SystemExit(main())
