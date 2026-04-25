"""Tier 1 extractor: Tree-sitter parse → `files` + `nodes` tables.

Per Architecture §4: language-agnostic. Walks the CST depth-first, assigns
sequential ids client-side, and inserts rows via `asyncpg.copy_records_to_table`.
Incremental: skipped if the content hash matches the existing `files` row.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from db.connection import reserve_node_ids
from .file_walker import DiscoveredFile, WalkConfig, walk_repo
from .grammar_meta import LANGUAGES

NODE_COLUMNS = (
    "id",
    "file_id",
    "node_type",
    "is_named",
    "start_byte",
    "end_byte",
    "start_row",
    "start_col",
    "end_row",
    "end_col",
    "text",
    "parent_id",
    "child_index",
)


@dataclass
class IndexFileResult:
    file_id: int
    rel_path: str
    language: str
    node_count: int
    skipped: bool = False  # True if hash matched existing row


@dataclass
class IndexRepoResult:
    indexed: list[IndexFileResult]
    skipped: list[IndexFileResult]

    @property
    def total_nodes(self) -> int:
        return sum(r.node_count for r in self.indexed)


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _walk_tree(root) -> list[tuple]:
    """DFS preorder. Returns list of (ts_node, parent_walk_index, child_index).

    parent_walk_index is None for the root, otherwise the position of the
    parent in this returned list. Caller maps walk_index → db id by adding
    a base id reserved from the sequence.
    """
    results: list[tuple] = []
    # Stack: (ts_node, parent_walk_index, child_index_among_siblings)
    stack: list[tuple] = [(root, None, 0)]
    while stack:
        ts, parent_wi, child_idx = stack.pop()
        my_wi = len(results)
        results.append((ts, parent_wi, child_idx))
        # Push children in reverse so leftmost is processed next.
        n = ts.child_count
        for i in range(n - 1, -1, -1):
            stack.append((ts.children[i], my_wi, i))
    return results


async def index_file(
    conn: asyncpg.Connection,
    repo_id: int,
    discovered: DiscoveredFile,
    *,
    from_dependency: bool = False,
) -> IndexFileResult:
    """Parse one file and write its `files` row + all `nodes` rows.

    If the existing `files` row for (repo_id, rel_path) has a matching
    content_hash, the file is skipped and IndexFileResult.skipped=True.
    """
    raw = discovered.path.read_bytes()
    content_hash = _hash_bytes(raw)

    existing = await conn.fetchrow(
        "SELECT id, content_hash FROM files WHERE repo_id=$1 AND path=$2",
        repo_id,
        discovered.rel_path,
    )
    if existing is not None and existing["content_hash"] == content_hash:
        return IndexFileResult(
            file_id=existing["id"],
            rel_path=discovered.rel_path,
            language=discovered.language,
            node_count=0,
            skipped=True,
        )

    # Hash changed (or never indexed) — drop old rows.
    if existing is not None:
        await conn.execute("DELETE FROM files WHERE id=$1", existing["id"])

    text = raw.decode("utf-8", errors="replace")

    # Insert files row first.
    file_id = await conn.fetchval(
        """
        INSERT INTO files (repo_id, path, language, content_hash, raw_content, from_dependency)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        repo_id,
        discovered.rel_path,
        discovered.language,
        content_hash,
        text,
        from_dependency,
    )

    # Parse with the language's grammar.
    parser = LANGUAGES[discovered.language].parser()
    tree = parser.parse(raw)

    walk = _walk_tree(tree.root_node)
    if not walk:
        return IndexFileResult(
            file_id=file_id,
            rel_path=discovered.rel_path,
            language=discovered.language,
            node_count=0,
        )

    # Reserve a contiguous block of ids for this file's nodes.
    first_id = await reserve_node_ids(conn, len(walk))

    records: list[tuple] = []
    for wi, (ts, parent_wi, child_idx) in enumerate(walk):
        is_leaf = ts.child_count == 0
        node_text: str | None = None
        if is_leaf:
            try:
                node_text = ts.text.decode("utf-8", errors="replace")
            except Exception:
                node_text = None
        parent_id = (first_id + parent_wi) if parent_wi is not None else None
        records.append(
            (
                first_id + wi,
                file_id,
                ts.type,
                ts.is_named,
                ts.start_byte,
                ts.end_byte,
                ts.start_point[0],
                ts.start_point[1],
                ts.end_point[0],
                ts.end_point[1],
                node_text,
                parent_id,
                child_idx,
            )
        )

    await conn.copy_records_to_table("nodes", records=records, columns=NODE_COLUMNS)

    return IndexFileResult(
        file_id=file_id,
        rel_path=discovered.rel_path,
        language=discovered.language,
        node_count=len(records),
    )


async def index_repo(
    pool: asyncpg.Pool,
    repo_id: int,
    repo_root: str | Path,
    *,
    walk_config: WalkConfig | None = None,
) -> IndexRepoResult:
    """Walk a repo and index every supported file. Single connection / single transaction per file."""
    cfg = walk_config or WalkConfig.with_defaults(repo_root)
    indexed: list[IndexFileResult] = []
    skipped: list[IndexFileResult] = []
    async with pool.acquire() as conn:
        for discovered in walk_repo(cfg):
            async with conn.transaction():
                result = await index_file(conn, repo_id, discovered)
            if result.skipped:
                skipped.append(result)
            else:
                indexed.append(result)
    return IndexRepoResult(indexed=indexed, skipped=skipped)


def index_repo_sync(repo_id: int, repo_root: str | Path, dsn: str | None = None) -> IndexRepoResult:
    """Convenience sync wrapper for CLI use."""
    from db.connection import pool_ctx

    async def _run() -> IndexRepoResult:
        async with pool_ctx(dsn) as pool:
            return await index_repo(pool, repo_id, repo_root)

    return asyncio.run(_run())
