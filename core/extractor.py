"""Tier 1 extractor: Tree-sitter parse → `file_versions` + `nodes` + `branch_files`.

Per Architecture §4: language-agnostic. Walks the CST depth-first, assigns
sequential ids client-side, and inserts rows via `asyncpg.copy_records_to_table`.

Branch-aware indexing model:

  • file_versions are content-keyed (UNIQUE on `repo_id, content_hash`). Two
    branches with identical content for some path share one row, so the
    parse output (nodes, definitions) is shared too — no tree-sitter
    re-parse across branches with shared content.

  • branch_files maps `(branch_id, path) → file_version_id`. The "view" of
    a branch is its set of branch_files rows.

  Indexing a file under a branch:
    1. Hash the bytes on disk.
    2. If a file_version with that (repo_id, content_hash) exists, skip
       parsing entirely and just upsert branch_files. This is the cheap
       path that fires for nearly every file when re-indexing a branch
       that diverges from main by only a few files.
    3. Otherwise parse, INSERT file_versions, COPY nodes tied to the new
       file_version_id, then upsert branch_files.

Size-cache shortcut: branch_files.size stores each file's byte length, and
`branch_dirs` stores a per-directory SHA-256 over its sorted manifest
(forming a Merkle tree rooted at the branch). On re-index the walker
compares `os.stat().st_size` against the cached size; on a match we trust
the cached content_hash and skip read+SHA-256 entirely. Size is chosen
over mtime because it's the only filesystem-metadata signal that's reliable
across platforms. Same-size content edits slip past — pass `--rehash` to
bypass the cache when that matters.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from db.connection import reserve_node_ids
from .file_walker import DiscoveredFile, WalkConfig, walk_repo_tree
from .grammar_meta import LANGUAGES

NODE_COLUMNS = (
    "id",
    "file_version_id",
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
    file_version_id: int
    rel_path: str
    language: str
    node_count: int
    skipped: bool = False  # True if (branch_id, path) already mapped to this file_version


@dataclass
class IndexRepoResult:
    indexed: list[IndexFileResult]
    skipped: list[IndexFileResult]
    deleted: list[str]

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
    branch_id: int,
    discovered: DiscoveredFile,
    raw: bytes,
    content_hash: str,
    *,
    from_dependency: bool = False,
) -> IndexFileResult:
    """Cold-path: parse one file and write file_versions + nodes + branch_files.

    Caller (index_repo) has already hashed the bytes and confirmed no
    matching file_versions row exists for this content. Hash-hit and
    mapping-only-update cases are handled by index_repo's bulk UNNEST.
    """
    text = raw.decode("utf-8", errors="replace")

    file_version_id = await conn.fetchval(
        """
        INSERT INTO file_versions (repo_id, content_hash, language, raw_content)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        repo_id,
        content_hash,
        discovered.language,
        text,
    )

    # Parse with the language's grammar. Pass the file extension so multi-grammar
    # languages (TypeScript: .ts vs .tsx) pick the right sub-grammar.
    parser = LANGUAGES[discovered.language].parser(Path(discovered.rel_path).suffix.lower())
    tree = parser.parse(raw)

    walk = _walk_tree(tree.root_node)

    if walk:
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
                    file_version_id,
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
        node_count = len(records)
    else:
        node_count = 0

    await conn.execute(
        """
        INSERT INTO branch_files (branch_id, path, file_version_id, from_dependency, size)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (branch_id, path) DO UPDATE
          SET file_version_id = EXCLUDED.file_version_id,
              from_dependency = EXCLUDED.from_dependency,
              size = EXCLUDED.size
        """,
        branch_id, discovered.rel_path, file_version_id, from_dependency, discovered.size,
    )

    return IndexFileResult(
        file_version_id=file_version_id,
        rel_path=discovered.rel_path,
        language=discovered.language,
        node_count=node_count,
    )


async def index_repo(
    pool: asyncpg.Pool,
    repo_id: int,
    branch_id: int,
    repo_root: str | Path,
    *,
    walk_config: WalkConfig | None = None,
    force_rehash: bool = False,
) -> IndexRepoResult:
    """Walk a repo and index every supported file under (repo_id, branch_id).

    Re-uses cached `(size, content_hash)` from `branch_files` so files whose
    byte size hasn't changed skip read+SHA-256 entirely. The walker also
    builds per-directory Merkle hashes which are persisted to `branch_dirs`.

    The per-file loop only writes for truly new content (cold path: parse +
    COPY nodes) and for hash-hit files whose `branch_files` mapping or
    cached size is missing or stale (bulk UNNEST upsert). Files whose hash
    and mapping are already correct cost zero DB round-trips beyond the
    initial prefetch.

    The branch_files prune at the end removes mappings for user files no
    longer reached by the walker (deletes, moves, newly excluded) without
    touching file_versions — file_versions are reclaimed only by `make gc`
    when no branch references them.
    """
    cfg = walk_config or WalkConfig.with_defaults(repo_root)
    indexed: list[IndexFileResult] = []
    skipped: list[IndexFileResult] = []

    # Prefetch the branch's current (path, fv_id, size, content_hash) so the
    # walker's leaf-hash provider can skip read+SHA-256 when size matches.
    # JOIN file_versions to pull content_hash in the same round-trip — FK
    # guarantees every branch_files row has a matching file_versions row.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE repos SET root_path=$2 WHERE id=$1",
            repo_id,
            str(Path(repo_root).resolve()),
        )
        bf_rows = await conn.fetch(
            """
            SELECT bf.path, bf.file_version_id, bf.size, fv.content_hash
            FROM branch_files bf
            JOIN file_versions fv ON fv.id = bf.file_version_id
            WHERE bf.branch_id = $1
            """,
            branch_id,
        )

    cached_fv_by_path: dict[str, int] = {}
    cached_size_by_path: dict[str, int] = {}
    cached_hash_by_path: dict[str, str] = {}
    for r in bf_rows:
        cached_fv_by_path[r["path"]] = r["file_version_id"]
        if r["size"] is not None:
            cached_size_by_path[r["path"]] = r["size"]
            cached_hash_by_path[r["path"]] = r["content_hash"]

    # Provider closure: returns the file's content hash, reading from disk
    # only on cache miss. Cache-hit files contribute to the walk for tree
    # purposes but never get read or retained in memory.
    raw_by_path: dict[str, bytes] = {}
    hash_by_path: dict[str, str] = {}

    def leaf_hash_provider(rel_path: str, size: int, abs_path: Path) -> str:
        if not force_rehash:
            cached_h = cached_hash_by_path.get(rel_path)
            if cached_h is not None and cached_size_by_path.get(rel_path) == size:
                hash_by_path[rel_path] = cached_h
                return cached_h
        raw = abs_path.read_bytes()
        h = _hash_bytes(raw)
        raw_by_path[rel_path] = raw
        hash_by_path[rel_path] = h
        return h

    files, dir_hashes = walk_repo_tree(cfg, leaf_hash_provider)
    walked_paths = [f.rel_path for f in files]

    async with pool.acquire() as conn:
        # For fresh hashes (cache miss) we may need to learn the file_version_id.
        # Cache-hit hashes already know theirs from the prefetch — no lookup.
        fresh_hashes = list({hash_by_path[p] for p in raw_by_path.keys()})
        fv_by_hash: dict[str, int] = {}
        if fresh_hashes:
            fv_rows = await conn.fetch(
                "SELECT id, content_hash FROM file_versions "
                "WHERE repo_id=$1 AND content_hash = ANY($2::text[])",
                repo_id, fresh_hashes,
            )
            fv_by_hash = {r["content_hash"]: r["id"] for r in fv_rows}

        # Partition: unchanged, rebind (bulk upsert), new (cold path).
        # Cold path is deduped by content_hash so identical-content paths
        # don't race to INSERT the same file_versions row.
        cold_by_hash: dict[str, tuple[DiscoveredFile, bytes]] = {}
        cold_extra: dict[str, list[DiscoveredFile]] = {}
        rebinds: list[tuple[str, int, str, int]] = []  # (rel_path, fv_id, language, size)
        for f in files:
            h = hash_by_path[f.rel_path]
            if f.rel_path in raw_by_path:
                # Cache miss: we read the file. fv may or may not exist.
                fv_id = fv_by_hash.get(h)
                if fv_id is None:
                    if h in cold_by_hash:
                        cold_extra.setdefault(h, []).append(f)
                    else:
                        cold_by_hash[h] = (f, raw_by_path[f.rel_path])
                    continue
                if (cached_fv_by_path.get(f.rel_path) == fv_id
                        and cached_size_by_path.get(f.rel_path) == f.size):
                    skipped.append(IndexFileResult(
                        file_version_id=fv_id,
                        rel_path=f.rel_path,
                        language=f.language,
                        node_count=0,
                        skipped=True,
                    ))
                else:
                    # Mapping changed, size needs backfill, or both.
                    rebinds.append((f.rel_path, fv_id, f.language, f.size))
            else:
                # Cache hit: trust prefetched fv_id; truly skipped.
                fv_id = cached_fv_by_path[f.rel_path]
                skipped.append(IndexFileResult(
                    file_version_id=fv_id,
                    rel_path=f.rel_path,
                    language=f.language,
                    node_count=0,
                    skipped=True,
                ))

        if rebinds:
            await conn.execute(
                """
                INSERT INTO branch_files (branch_id, path, file_version_id, from_dependency, size)
                SELECT $1, * FROM UNNEST($2::text[], $3::bigint[], $4::boolean[], $5::bigint[])
                ON CONFLICT (branch_id, path) DO UPDATE
                  SET file_version_id = EXCLUDED.file_version_id,
                      from_dependency = EXCLUDED.from_dependency,
                      size = EXCLUDED.size
                """,
                branch_id,
                [r[0] for r in rebinds],
                [r[1] for r in rebinds],
                [False] * len(rebinds),
                [r[3] for r in rebinds],
            )
            for rel_path, fv_id, lang, _size in rebinds:
                indexed.append(IndexFileResult(
                    file_version_id=fv_id,
                    rel_path=rel_path,
                    language=lang,
                    node_count=0,
                    skipped=False,
                ))

        # Prune branch_files mappings for user files no longer reached by the
        # walker (deletes, moves, newly excluded). Dependency mappings are
        # kept — they're populated by Phase 3, not the walker, so absence
        # here is not evidence of removal. file_versions are not deleted
        # here; orphans are reclaimed by `make gc`.
        deleted_rows = await conn.fetch(
            """
            DELETE FROM branch_files
            WHERE branch_id = $1
              AND from_dependency = FALSE
              AND path <> ALL($2::text[])
            RETURNING path
            """,
            branch_id,
            walked_paths,
        )
        deleted = [r["path"] for r in deleted_rows]

        # Persist directory Merkle hashes. Bulk upsert all current dirs,
        # then prune rows for directories no longer in the walk (deleted
        # subtrees, newly excluded paths).
        dir_paths = list(dir_hashes.keys())
        dir_tree_hashes = [dir_hashes[p] for p in dir_paths]
        await conn.execute(
            """
            INSERT INTO branch_dirs (branch_id, path, tree_hash)
            SELECT $1, * FROM UNNEST($2::text[], $3::text[])
            ON CONFLICT (branch_id, path) DO UPDATE
              SET tree_hash = EXCLUDED.tree_hash
            """,
            branch_id, dir_paths, dir_tree_hashes,
        )
        await conn.execute(
            "DELETE FROM branch_dirs WHERE branch_id=$1 AND path <> ALL($2::text[])",
            branch_id, dir_paths,
        )

    # Cold path: parse + insert each new file on its own connection so
    # tree-sitter work and DB I/O overlap across files.
    if cold_by_hash:
        sem = asyncio.Semaphore(int(os.environ.get("INDEX_CONCURRENCY", "4")))

        async def _cold(discovered: DiscoveredFile, raw: bytes, content_hash: str) -> IndexFileResult:
            async with sem, pool.acquire() as c, c.transaction():
                return await index_file(c, repo_id, branch_id, discovered, raw, content_hash)

        cold_items = list(cold_by_hash.items())  # [(hash, (discovered, raw)), ...]
        cold_results = await asyncio.gather(
            *[_cold(d, r, h) for h, (d, r) in cold_items]
        )
        indexed.extend(cold_results)

        # Map extra paths (same hash as a cold task) onto the minted fv_id.
        if cold_extra:
            hash_to_fv = {h: res.file_version_id for (h, _), res in zip(cold_items, cold_results)}
            extra_paths: list[str] = []
            extra_fvs: list[int] = []
            extra_sizes: list[int] = []
            for h, dups in cold_extra.items():
                fv_id = hash_to_fv[h]
                for d in dups:
                    extra_paths.append(d.rel_path)
                    extra_fvs.append(fv_id)
                    extra_sizes.append(d.size)
                    indexed.append(IndexFileResult(
                        file_version_id=fv_id,
                        rel_path=d.rel_path,
                        language=d.language,
                        node_count=0,
                    ))
            async with pool.acquire() as c:
                await c.execute(
                    """
                    INSERT INTO branch_files (branch_id, path, file_version_id, from_dependency, size)
                    SELECT $1, * FROM UNNEST($2::text[], $3::bigint[], $4::boolean[], $5::bigint[])
                    ON CONFLICT (branch_id, path) DO UPDATE
                      SET file_version_id = EXCLUDED.file_version_id,
                          from_dependency = EXCLUDED.from_dependency,
                          size = EXCLUDED.size
                    """,
                    branch_id, extra_paths, extra_fvs, [False] * len(extra_paths), extra_sizes,
                )

    return IndexRepoResult(indexed=indexed, skipped=skipped, deleted=deleted)


def index_repo_sync(
    repo_id: int,
    branch_id: int,
    repo_root: str | Path,
    dsn: str | None = None,
) -> IndexRepoResult:
    """Convenience sync wrapper for CLI use."""
    from db.connection import pool_ctx

    async def _run() -> IndexRepoResult:
        async with pool_ctx(dsn) as pool:
            return await index_repo(pool, repo_id, branch_id, repo_root)

    return asyncio.run(_run())
