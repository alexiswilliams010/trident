"""Resolve a CLI `--repo-name` to the integer `repo_id` used internally.

Public API at the CLI is the human-readable `name`; everything below the CLI
threads `repo_id` (BIGINT) because that's what the schema's foreign keys and
WHERE clauses use. This helper bridges the two.

Indexing creates the row on first use (`create=True`); read-only commands
look it up and error out if missing (`create=False`) so a typo doesn't
silently scope a query to nothing.
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
