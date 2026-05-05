"""Resolve a CLI `--repo-name` (and optional `--branch`) to the integer ids
used internally.

Public API at the CLI is the human-readable `name`; everything below the CLI
threads `(repo_id, branch_id)` because that's what the schema's foreign keys
and WHERE clauses use. These helpers bridge the two.

Branches are scoped per-repo: two repos can each own a `main` or `feature/x`
without colliding. When a query omits `--branch`, the repo's designated
default branch (the row with `is_default=TRUE`) is used.
"""

from __future__ import annotations

import sys

import asyncpg


async def resolve_repo_id(
    pool: asyncpg.Pool,
    *,
    name: str,
    root_path: str | None = None,
    create: bool = False,
) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM repos WHERE name = $1", name)
        if row is not None:
            return row["id"]
        if not create:
            print(
                f"error: no repo named {name!r} in this database. "
                "Run `make index REPO_NAME=... REPO_PATH=...` first, "
                "or check `SELECT name FROM repos` for the right name.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        row = await conn.fetchrow(
            "INSERT INTO repos (name, root_path) VALUES ($1, $2) RETURNING id",
            name, root_path,
        )
        return row["id"]


async def resolve_repo_ids(pool: asyncpg.Pool, names: list[str]) -> list[int]:
    """Bulk version: take a list of repo names and return their IDs in the
    same order. Errors out (SystemExit 2) on any name that isn't indexed,
    so a typo can't silently scope a cross-repo query to fewer repos than
    the caller asked for."""
    if not names:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, id FROM repos WHERE name = ANY($1::text[])", names,
        )
    found = {r["name"]: r["id"] for r in rows}
    missing = [n for n in names if n not in found]
    if missing:
        print(
            f"error: no repo(s) named {missing!r} in this database. "
            "Check `SELECT name FROM repos` for the right names.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return [found[n] for n in names]


DEFAULT_BRANCH_NAME = "main"


async def resolve_branch_id(
    pool: asyncpg.Pool,
    *,
    repo_id: int,
    name: str | None = None,
    create: bool = False,
    repo_name_for_errors: str = "",
) -> int:
    """Resolve a branch name to its branch_id within a single repo.

    name=None → return the row with is_default=TRUE for this repo_id.
    create=True with name=None → upsert DEFAULT_BRANCH_NAME ('main') as the
                                  default branch (marks is_default=TRUE if
                                  this is the first branch for the repo).
    create=True with name='foo' → upsert; if this is the first branch, mark
                                   it as default.
    create=False with a missing name → SystemExit(2) with a clear message.
    """
    target_name = name or DEFAULT_BRANCH_NAME

    async with pool.acquire() as conn:
        if name is None:
            row = await conn.fetchrow(
                "SELECT id FROM branches WHERE repo_id=$1 AND is_default",
                repo_id,
            )
            if row is not None:
                return row["id"]
            if not create:
                print(
                    f"error: repo {repo_name_for_errors or repo_id!r} has no "
                    "default branch. Run `make index ... BRANCH=name` first, "
                    "or check `SELECT name FROM branches WHERE repo_id=...`.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
        else:
            row = await conn.fetchrow(
                "SELECT id FROM branches WHERE repo_id=$1 AND name=$2",
                repo_id, name,
            )
            if row is not None:
                return row["id"]
            if not create:
                print(
                    f"error: repo {repo_name_for_errors or repo_id!r} has no "
                    f"branch named {name!r}. Available branches: "
                    "`SELECT name FROM branches WHERE repo_id=...`.",
                    file=sys.stderr,
                )
                raise SystemExit(2)

        # Create. First branch in a repo becomes the default automatically.
        async with conn.transaction():
            has_any = await conn.fetchval(
                "SELECT 1 FROM branches WHERE repo_id=$1 LIMIT 1",
                repo_id,
            )
            is_default = not has_any
            row = await conn.fetchrow(
                """
                INSERT INTO branches (repo_id, name, is_default)
                VALUES ($1, $2, $3)
                ON CONFLICT (repo_id, name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                repo_id, target_name, is_default,
            )
        return row["id"]


async def resolve_repo_and_branch(
    pool: asyncpg.Pool,
    *,
    repo_name: str,
    branch_name: str | None = None,
    create: bool = False,
    root_path: str | None = None,
) -> tuple[int, int]:
    """Combined helper used at every CLI entry point that targets a single
    (repo, branch). Returns (repo_id, branch_id)."""
    repo_id = await resolve_repo_id(
        pool, name=repo_name, root_path=root_path, create=create,
    )
    branch_id = await resolve_branch_id(
        pool, repo_id=repo_id, name=branch_name, create=create,
        repo_name_for_errors=repo_name,
    )
    return (repo_id, branch_id)


async def resolve_repo_branch_pairs(
    pool: asyncpg.Pool,
    pairs: list[tuple[str, str | None]],
) -> list[tuple[int, int]]:
    """Bulk version for cross-repo queries with `--repos a:main,b:feature/x,c`
    syntax. Each entry is (repo_name, branch_name|None); a None branch means
    "this repo's default branch". Errors out on any unknown repo/branch."""
    if not pairs:
        return []
    repo_names = [p[0] for p in pairs]
    repo_ids = await resolve_repo_ids(pool, repo_names)
    out: list[tuple[int, int]] = []
    for (repo_name, branch_name), repo_id in zip(pairs, repo_ids):
        branch_id = await resolve_branch_id(
            pool, repo_id=repo_id, name=branch_name, create=False,
            repo_name_for_errors=repo_name,
        )
        out.append((repo_id, branch_id))
    return out
