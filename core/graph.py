"""Graph traversal queries over the semantic code graph.

Pure SQL against Tier 2 tables (definitions, call_edges, inherits_edges,
overrides_edges, imports, files, nodes). Returns definition metadata, not
chunks — complementary to core/retrieval.py which is embedding-focused.

All queries run inside a transaction with a configurable statement timeout
(default 120s). No internal depth limits are imposed; the timeout is the
only guardrail. An optional max_depth can be passed to recursive traversals.
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg


@dataclass
class DefInfo:
    def_id: int
    name: str
    qualified_name: str
    kind: str
    file_path: str
    file_id: int
    start_line: int
    end_line: int
    visibility: str | None
    repo_id: int
    source: str | None = None
    depth: int | None = None


def _norm_repo_ids(x: int | list[int]) -> list[int]:
    if isinstance(x, int):
        return [x]
    return list(x)


def _row_to_def(r: asyncpg.Record) -> DefInfo:
    return DefInfo(
        def_id=r["def_id"],
        name=r["name"],
        qualified_name=r["qualified_name"],
        kind=r["kind"],
        file_path=r["file_path"],
        file_id=r["file_id"],
        start_line=r["start_row"],
        end_line=r["end_row"],
        visibility=r.get("visibility"),
        repo_id=r["repo_id"],
    )


_DEF_COLS = """\
d.id AS def_id, d.name, d.qualified_name, d.kind, d.visibility,
f.path AS file_path, f.id AS file_id, f.repo_id,
n.start_row, n.end_row, n.start_byte, n.end_byte"""


async def _set_timeout(conn: asyncpg.Connection, timeout_s: int) -> None:
    await conn.execute(f"SET LOCAL statement_timeout = '{timeout_s}s'")


# ────────────────────────────────────────────────────────────────────
# Resolve definitions by name
# ────────────────────────────────────────────────────────────────────


async def resolve_definitions(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    kind: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    rids = _norm_repo_ids(repo_ids)
    kind_clause = "AND d.kind = $3" if kind else ""
    params: list = [rids, name]
    if kind:
        params.append(kind)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT {_DEF_COLS}
                FROM definitions d
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                WHERE f.repo_id = ANY($1::bigint[])
                  AND (d.name = $2 OR d.qualified_name = $2
                       OR d.qualified_name LIKE '%%.' || $2)
                  {kind_clause}
                ORDER BY d.qualified_name
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


async def _resolve_def_ids(
    conn: asyncpg.Connection,
    repo_ids: list[int],
    name: str,
) -> list[int]:
    rows = await conn.fetch(
        """
        SELECT d.id FROM definitions d
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = ANY($1::bigint[])
          AND (d.name = $2 OR d.qualified_name = $2
               OR d.qualified_name LIKE '%.' || $2)
        """,
        repo_ids, name,
    )
    return [r["id"] for r in rows]


# ────────────────────────────────────────────────────────────────────
# Direct neighbors (1-hop)
# ────────────────────────────────────────────────────────────────────


async def callers_of(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    rids = _norm_repo_ids(repo_ids)
    conf_clause = "AND ce.confidence = $3" if confidence else ""
    params: list = [rids, name]
    if confidence:
        params.append(confidence)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM call_edges ce
                JOIN definitions target ON target.id = ce.callee_def_id
                JOIN files tf ON tf.id = target.file_id
                JOIN definitions d ON d.id = ce.caller_def_id
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                WHERE tf.repo_id = ANY($1::bigint[])
                  AND (target.name = $2 OR target.qualified_name = $2
                       OR target.qualified_name LIKE '%%.' || $2)
                  {conf_clause}
                ORDER BY d.id
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


async def callees_of(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    rids = _norm_repo_ids(repo_ids)
    conf_clause = "AND ce.confidence = $3" if confidence else ""
    params: list = [rids, name]
    if confidence:
        params.append(confidence)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM call_edges ce
                JOIN definitions caller ON caller.id = ce.caller_def_id
                JOIN files cf ON cf.id = caller.file_id
                JOIN definitions d ON d.id = ce.callee_def_id
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                WHERE cf.repo_id = ANY($1::bigint[])
                  AND (caller.name = $2 OR caller.qualified_name = $2
                       OR caller.qualified_name LIKE '%%.' || $2)
                  {conf_clause}
                ORDER BY d.id
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Transitive traversals (recursive CTE)
# ────────────────────────────────────────────────────────────────────


async def ancestors(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    max_depth: int | None = None,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Transitive callers — upward call-graph slice."""
    rids = _norm_repo_ids(repo_ids)
    depth_clause = f"AND anc.depth < {max_depth}" if max_depth is not None else ""
    conf_clause = "AND ce.confidence = $2" if confidence else ""
    params: list = [rids]
    if confidence:
        params.append(confidence)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            seed_ids = await _resolve_def_ids(conn, rids, name)
            if not seed_ids:
                return []
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE anc AS (
                    SELECT ce.caller_def_id AS def_id, 1 AS depth
                    FROM call_edges ce
                    WHERE ce.callee_def_id = ANY($1::bigint[])
                      AND ce.caller_def_id IS NOT NULL
                      {conf_clause.replace('$2', '$' + str(len(params) + 1)) if confidence else ''}
                    UNION
                    SELECT ce.caller_def_id, anc.depth + 1
                    FROM call_edges ce
                    JOIN anc ON anc.def_id = ce.callee_def_id
                    WHERE ce.caller_def_id IS NOT NULL
                      AND ce.caller_def_id != ALL($1::bigint[])
                      {depth_clause}
                      {conf_clause.replace('$2', '$' + str(len(params) + 1)) if confidence else ''}
                )
                SELECT DISTINCT ON (d.id) {_DEF_COLS}, anc.depth
                FROM anc
                JOIN definitions d ON d.id = anc.def_id
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                ORDER BY d.id, anc.depth
                """,
                seed_ids, *params[1:],
            )
    result = []
    for r in rows:
        d = _row_to_def(r)
        d.depth = r["depth"]
        result.append(d)
    return result


async def reachable_from(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    max_depth: int | None = None,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Transitive callees — downward call-graph slice (blast radius)."""
    rids = _norm_repo_ids(repo_ids)
    depth_clause = f"AND reach.depth < {max_depth}" if max_depth is not None else ""
    conf_clause = "AND ce.confidence = $2" if confidence else ""
    params: list = [rids]
    if confidence:
        params.append(confidence)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            seed_ids = await _resolve_def_ids(conn, rids, name)
            if not seed_ids:
                return []
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE reach AS (
                    SELECT ce.callee_def_id AS def_id, 1 AS depth
                    FROM call_edges ce
                    WHERE ce.caller_def_id = ANY($1::bigint[])
                      AND ce.callee_def_id IS NOT NULL
                      {conf_clause.replace('$2', '$' + str(len(params) + 1)) if confidence else ''}
                    UNION
                    SELECT ce.callee_def_id, reach.depth + 1
                    FROM call_edges ce
                    JOIN reach ON reach.def_id = ce.caller_def_id
                    WHERE ce.callee_def_id IS NOT NULL
                      AND ce.callee_def_id != ALL($1::bigint[])
                      {depth_clause}
                      {conf_clause.replace('$2', '$' + str(len(params) + 1)) if confidence else ''}
                )
                SELECT DISTINCT ON (d.id) {_DEF_COLS}, reach.depth
                FROM reach
                JOIN definitions d ON d.id = reach.def_id
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                ORDER BY d.id, reach.depth
                """,
                seed_ids, *params[1:],
            )
    result = []
    for r in rows:
        d = _row_to_def(r)
        d.depth = r["depth"]
        result.append(d)
    return result


# ────────────────────────────────────────────────────────────────────
# Path enumeration
# ────────────────────────────────────────────────────────────────────


async def paths_between(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    source_name: str,
    target_name: str,
    *,
    max_depth: int | None = None,
    max_paths: int = 50,
    timeout_s: int = 120,
) -> list[list[DefInfo]]:
    """All simple call paths between two definitions."""
    rids = _norm_repo_ids(repo_ids)
    depth_clause = (
        f"AND array_length(p.path, 1) < {max_depth}"
        if max_depth is not None else ""
    )
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            src_ids = await _resolve_def_ids(conn, rids, source_name)
            dst_ids = await _resolve_def_ids(conn, rids, target_name)
            if not src_ids or not dst_ids:
                return []
            path_rows = await conn.fetch(
                f"""
                WITH RECURSIVE paths AS (
                    SELECT ARRAY[ce.caller_def_id, ce.callee_def_id] AS path
                    FROM call_edges ce
                    WHERE ce.caller_def_id = ANY($1::bigint[])
                      AND ce.callee_def_id IS NOT NULL
                    UNION ALL
                    SELECT p.path || ce.callee_def_id
                    FROM paths p
                    JOIN call_edges ce ON ce.caller_def_id = p.path[array_length(p.path, 1)]
                    WHERE ce.callee_def_id IS NOT NULL
                      AND NOT p.path @> ARRAY[ce.callee_def_id]
                      {depth_clause}
                )
                SELECT path FROM paths
                WHERE path[array_length(path, 1)] = ANY($2::bigint[])
                LIMIT $3
                """,
                src_ids, dst_ids, max_paths,
            )
            if not path_rows:
                return []
            all_ids: set[int] = set()
            for pr in path_rows:
                all_ids.update(pr["path"])
            def_rows = await conn.fetch(
                f"""
                SELECT {_DEF_COLS}
                FROM definitions d
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                WHERE d.id = ANY($1::bigint[])
                """,
                list(all_ids),
            )
    defs_by_id = {r["def_id"]: _row_to_def(r) for r in def_rows}
    result: list[list[DefInfo]] = []
    for pr in path_rows:
        path = [defs_by_id[did] for did in pr["path"] if did in defs_by_id]
        if path:
            result.append(path)
    return result


# ────────────────────────────────────────────────────────────────────
# Entrypoints
# ────────────────────────────────────────────────────────────────────


async def entrypoints(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    *,
    kind: str | None = None,
    file_path: str | None = None,
    include_internal: bool = False,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Functions/methods with no internal callers and no override relationships.

    By default excludes constructors, internal/private visibility, and
    interface file stubs. Pass --kind constructor to include constructors,
    or include_internal=True to include internal/private functions.
    """
    rids = _norm_repo_ids(repo_ids)
    clauses: list[str] = []
    params: list = [rids]
    idx = 2
    if kind:
        clauses.append(f"AND d.kind = ${idx}")
        params.append(kind)
        idx += 1
    else:
        clauses.append("AND d.kind IN ('function', 'method')")
    if file_path:
        clauses.append(f"AND f.path LIKE '%%' || ${idx}")
        params.append(file_path)
        idx += 1
    if not include_internal:
        clauses.append("AND (d.visibility IS NULL OR d.visibility NOT IN ('internal', 'private'))")
        clauses.append("AND f.path NOT LIKE '%%/interfaces/%%'")
        clauses.append("AND f.path NOT LIKE '%%/interface/%%'")
    extra = "\n                  ".join(clauses)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT {_DEF_COLS}
                FROM definitions d
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                LEFT JOIN call_edges ce ON ce.callee_def_id = d.id
                LEFT JOIN overrides_edges oe ON oe.child_def_id = d.id
                WHERE f.repo_id = ANY($1::bigint[])
                  AND ce.id IS NULL
                  AND oe.id IS NULL
                  {extra}
                ORDER BY f.path, d.qualified_name
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


async def entrypoint_paths(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    target_name: str,
    *,
    max_depth: int | None = None,
    max_paths: int = 50,
    timeout_s: int = 120,
) -> list[list[DefInfo]]:
    """Call paths from entrypoints to a target definition."""
    anc = await ancestors(pool, repo_ids, target_name, max_depth=max_depth, timeout_s=timeout_s)
    ep = await entrypoints(pool, repo_ids, timeout_s=timeout_s)
    ep_ids = {e.def_id for e in ep}
    entry_ancestors = [a for a in anc if a.def_id in ep_ids]
    if not entry_ancestors:
        return []
    result: list[list[DefInfo]] = []
    for ea in entry_ancestors:
        p = await paths_between(
            pool, repo_ids, ea.qualified_name, target_name,
            max_depth=max_depth, max_paths=max(1, max_paths // len(entry_ancestors)),
            timeout_s=timeout_s,
        )
        result.extend(p)
        if len(result) >= max_paths:
            break
    return result[:max_paths]


# ────────────────────────────────────────────────────────────────────
# Source extraction
# ────────────────────────────────────────────────────────────────────


async def get_source(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    kind: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Resolve definitions and populate source from raw file content."""
    rids = _norm_repo_ids(repo_ids)
    kind_clause = "AND d.kind = $3" if kind else ""
    params: list = [rids, name]
    if kind:
        params.append(kind)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT {_DEF_COLS}, f.raw_content
                FROM definitions d
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                WHERE f.repo_id = ANY($1::bigint[])
                  AND (d.name = $2 OR d.qualified_name = $2
                       OR d.qualified_name LIKE '%%.' || $2)
                  {kind_clause}
                ORDER BY d.qualified_name
                """,
                *params,
            )
    result = []
    for r in rows:
        d = _row_to_def(r)
        raw = r["raw_content"] or ""
        d.source = raw[r["start_byte"]:r["end_byte"]]
        result.append(d)
    return result


# ────────────────────────────────────────────────────────────────────
# Import queries
# ────────────────────────────────────────────────────────────────────


@dataclass
class ImportInfo:
    import_path: str
    imported_names: list[str]
    dep_class: str
    resolved_file: str | None
    file_path: str
    file_id: int


async def file_imports(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    *,
    file_path: str | None = None,
    dep_class: str | None = None,
    timeout_s: int = 120,
) -> list[ImportInfo]:
    rids = _norm_repo_ids(repo_ids)
    clauses: list[str] = []
    params: list = [rids]
    idx = 2
    if file_path:
        clauses.append(f"AND f.path = ${idx}")
        params.append(file_path)
        idx += 1
    if dep_class:
        clauses.append(f"AND i.dep_class = ${idx}")
        params.append(dep_class)
        idx += 1
    extra = " ".join(clauses)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT i.import_path, i.imported_names, i.dep_class,
                       rf.path AS resolved_file,
                       f.path AS file_path, f.id AS file_id
                FROM imports i
                JOIN files f ON f.id = i.file_id
                LEFT JOIN files rf ON rf.id = i.resolved_file_id
                WHERE f.repo_id = ANY($1::bigint[])
                  {extra}
                ORDER BY f.path, i.import_path
                """,
                *params,
            )
    return [
        ImportInfo(
            import_path=r["import_path"],
            imported_names=list(r["imported_names"] or []),
            dep_class=r["dep_class"],
            resolved_file=r["resolved_file"],
            file_path=r["file_path"],
            file_id=r["file_id"],
        )
        for r in rows
    ]


async def file_dependents(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    file_path: str,
    *,
    timeout_s: int = 120,
) -> list[ImportInfo]:
    """Files that import a given file (reverse import lookup)."""
    rids = _norm_repo_ids(repo_ids)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                """
                SELECT i.import_path, i.imported_names, i.dep_class,
                       target.path AS resolved_file,
                       f.path AS file_path, f.id AS file_id
                FROM imports i
                JOIN files f ON f.id = i.file_id
                JOIN files target ON target.id = i.resolved_file_id
                WHERE f.repo_id = ANY($1::bigint[])
                  AND target.path = $2
                ORDER BY f.path
                """,
                rids, file_path,
            )
    return [
        ImportInfo(
            import_path=r["import_path"],
            imported_names=list(r["imported_names"] or []),
            dep_class=r["dep_class"],
            resolved_file=r["resolved_file"],
            file_path=r["file_path"],
            file_id=r["file_id"],
        )
        for r in rows
    ]


# ────────────────────────────────────────────────────────────────────
# Inheritance tree
# ────────────────────────────────────────────────────────────────────


@dataclass
class InheritanceNode:
    def_info: DefInfo
    bases: list[str]
    children: list[str]


async def inheritance_tree(
    pool: asyncpg.Pool,
    repo_ids: int | list[int],
    name: str,
    *,
    timeout_s: int = 120,
) -> list[InheritanceNode]:
    """Full inheritance hierarchy (up and down) from a named class."""
    rids = _norm_repo_ids(repo_ids)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            seed_ids = await _resolve_def_ids(conn, rids, name)
            if not seed_ids:
                return []
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE tree AS (
                    -- seed: both sides of edges touching the seed
                    SELECT child_def_id AS def_id FROM inherits_edges
                    WHERE child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[])
                    UNION
                    SELECT base_def_id FROM inherits_edges
                    WHERE (child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[]))
                      AND base_def_id IS NOT NULL
                ),
                walk_up AS (
                    SELECT def_id FROM tree
                    UNION
                    SELECT ie.base_def_id
                    FROM inherits_edges ie
                    JOIN walk_up w ON w.def_id = ie.child_def_id
                    WHERE ie.base_def_id IS NOT NULL
                ),
                walk_down AS (
                    SELECT def_id FROM tree
                    UNION
                    SELECT ie.child_def_id
                    FROM inherits_edges ie
                    JOIN walk_down w ON w.def_id = ie.base_def_id
                ),
                all_ids AS (
                    SELECT def_id FROM walk_up
                    UNION
                    SELECT def_id FROM walk_down
                )
                SELECT DISTINCT {_DEF_COLS}
                FROM all_ids a
                JOIN definitions d ON d.id = a.def_id
                JOIN nodes n ON n.id = d.node_id
                JOIN files f ON f.id = d.file_id
                ORDER BY d.qualified_name
                """,
                seed_ids,
            )
            all_ids = [r["def_id"] for r in rows]
            if not all_ids:
                all_ids = seed_ids
                rows = await conn.fetch(
                    f"""
                    SELECT {_DEF_COLS}
                    FROM definitions d
                    JOIN nodes n ON n.id = d.node_id
                    JOIN files f ON f.id = d.file_id
                    WHERE d.id = ANY($1::bigint[])
                    """,
                    all_ids,
                )
            edge_rows = await conn.fetch(
                """
                SELECT child_def_id, base_def_id, base_name
                FROM inherits_edges
                WHERE child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[])
                """,
                all_ids,
            )

    defs_by_id = {r["def_id"]: _row_to_def(r) for r in rows}
    bases_map: dict[int, list[str]] = {}
    children_map: dict[int, list[str]] = {}
    for er in edge_rows:
        child_id = er["child_def_id"]
        base_id = er["base_def_id"]
        base_name = er["base_name"]
        child_def = defs_by_id.get(child_id)
        base_def = defs_by_id.get(base_id) if base_id else None
        bases_map.setdefault(child_id, []).append(
            base_def.qualified_name if base_def else base_name
        )
        if base_id and base_id in defs_by_id and child_def:
            children_map.setdefault(base_id, []).append(child_def.qualified_name)

    return [
        InheritanceNode(
            def_info=defs_by_id[did],
            bases=bases_map.get(did, []),
            children=children_map.get(did, []),
        )
        for did in defs_by_id
    ]
