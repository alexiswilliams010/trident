"""CLI: manage branches within an indexed DB.

Subcommands:
    list                                   — show branches grouped by repo
    set-default REPO_NAME BRANCH_NAME      — re-flag a repo's default branch
    drop REPO_NAME BRANCH_NAME             — delete a non-default branch (+ cascade)
    gc                                     — reclaim orphan file_versions and chunk_embeddings

Branches are scoped per repo (UNIQUE on `repo_id, name`), so two repos can
each own a `main` or a `feature/x` without colliding. Branch deletion
cascades through every per-branch table (branch_files, references,
call_edges, data_access, inherits_edges, overrides_edges, imports,
external_dependencies, chunks). Shared content (file_versions, nodes,
definitions, chunk_embeddings) survives the drop and is reclaimed by `gc`
when no remaining branch references it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from db.connection import pool_ctx


# ────────────────────────────────────────────────────────────────────
# Subcommands
# ────────────────────────────────────────────────────────────────────


async def _cmd_list(pool, args: argparse.Namespace) -> int:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT r.name AS repo, b.name AS branch, b.is_default,
                   COALESCE(bf_counts.n, 0) AS files
            FROM branches b
            JOIN repos r ON r.id = b.repo_id
            LEFT JOIN (
                SELECT branch_id, COUNT(*) AS n FROM branch_files GROUP BY branch_id
            ) bf_counts ON bf_counts.branch_id = b.id
            ORDER BY r.name, b.is_default DESC, b.name
            """
        )
    if not rows:
        print("(no branches)")
        return 0
    width_repo = max(len(r["repo"]) for r in rows)
    width_branch = max(len(r["branch"]) for r in rows)
    print(f"{'repo':<{width_repo}}  {'branch':<{width_branch}}  default  files")
    print("-" * (width_repo + width_branch + 22))
    for r in rows:
        flag = "  *    " if r["is_default"] else "       "
        print(f"{r['repo']:<{width_repo}}  {r['branch']:<{width_branch}}  {flag}  {r['files']}")
    return 0


async def _cmd_set_default(pool, args: argparse.Namespace) -> int:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT b.id FROM branches b
                JOIN repos r ON r.id = b.repo_id
                WHERE r.name = $1 AND b.name = $2
                """,
                args.repo_name, args.branch,
            )
            if row is None:
                print(f"error: no branch {args.branch!r} on repo {args.repo_name!r}", file=sys.stderr)
                return 2
            target_id = row["id"]
            await conn.execute(
                """
                UPDATE branches SET is_default = FALSE
                WHERE repo_id = (SELECT id FROM repos WHERE name = $1)
                """,
                args.repo_name,
            )
            await conn.execute(
                "UPDATE branches SET is_default = TRUE WHERE id = $1",
                target_id,
            )
    print(f"set {args.repo_name}:{args.branch} as default branch")
    return 0


async def _cmd_drop(pool, args: argparse.Namespace) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT b.id, b.is_default FROM branches b
            JOIN repos r ON r.id = b.repo_id
            WHERE r.name = $1 AND b.name = $2
            """,
            args.repo_name, args.branch,
        )
        if row is None:
            print(f"error: no branch {args.branch!r} on repo {args.repo_name!r}", file=sys.stderr)
            return 2
        if row["is_default"]:
            print(
                f"error: refusing to drop the default branch {args.repo_name}:{args.branch}. "
                "Run `make branch-set-default REPO_NAME=... BRANCH=other` first.",
                file=sys.stderr,
            )
            return 2
        # Single statement; cascades through all per-branch tables.
        await conn.execute("DELETE FROM branches WHERE id = $1", row["id"])
    print(
        f"dropped {args.repo_name}:{args.branch}. "
        "Shared content (file_versions, definitions, chunk_embeddings) is preserved; "
        "run `make gc` to reclaim any rows no other branch references."
    )
    return 0


async def _cmd_gc(pool, args: argparse.Namespace) -> int:
    async with pool.acquire() as conn:
        # Two passes — first orphan file_versions, then orphan chunk_embeddings.
        # No transaction wrapper: GC may take a while on large DBs and we don't
        # want a single long-running transaction holding locks on shared tables.
        fv_status = await conn.execute(
            """
            DELETE FROM file_versions
            WHERE id NOT IN (SELECT DISTINCT file_version_id FROM branch_files)
            """,
        )
        ce_status = await conn.execute(
            """
            DELETE FROM chunk_embeddings
            WHERE content_hash NOT IN (SELECT DISTINCT content_hash FROM chunks)
            """,
        )
    print(f"file_versions: {fv_status}")
    print(f"chunk_embeddings: {ce_status}")
    return 0


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> int:
    async with pool_ctx(args.dsn) as pool:
        return await args.func(pool, args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="trident — branch management")
    parser.add_argument("--dsn", type=str, default=None)
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("list", help="List branches across all repos")
    p.set_defaults(func=_cmd_list)

    p = subs.add_parser("set-default", help="Mark a branch as the repo's default")
    p.add_argument("--repo-name", type=str, required=True)
    p.add_argument("--branch", type=str, required=True)
    p.set_defaults(func=_cmd_set_default)

    p = subs.add_parser("drop", help="Delete a non-default branch (cascades)")
    p.add_argument("--repo-name", type=str, required=True)
    p.add_argument("--branch", type=str, required=True)
    p.set_defaults(func=_cmd_drop)

    p = subs.add_parser("gc", help="Reclaim orphan file_versions and chunk_embeddings")
    p.set_defaults(func=_cmd_gc)

    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
