"""Graph traversal queries over the semantic code graph.

Pure SQL against Tier 2 tables (definitions, call_edges, inherits_edges,
overrides_edges, imports, file_versions, branch_files, nodes). Returns
definition metadata, not chunks — complementary to core/retrieval.py which
is embedding-focused.

Branch model: every public function takes a list of `branch_ids`. Definitions
are content-shared so a single def can be visible in multiple branches;
results are de-duped by def_id. Edge tables (call_edges, inherits_edges,
overrides_edges, imports) are per-branch and filtered by branch_id, so a
walk's results reflect the union of those branches' resolution context.

All queries run inside a transaction with a configurable statement timeout
(default 120s).
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
    file_version_id: int
    start_line: int
    end_line: int
    visibility: str | None
    repo_id: int
    source: str | None = None
    depth: int | None = None


def _norm_branch_ids(x: int | list[int]) -> list[int]:
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
        file_version_id=r["file_version_id"],
        start_line=r["start_row"],
        end_line=r["end_row"],
        visibility=r.get("visibility"),
        repo_id=r["repo_id"],
    )


# Common SELECT-fragment for definitions joined with their nodes and the path
# they have in the current branch set. `bf.path` is the per-branch path
# mapping; using `DISTINCT ON (d.id)` and ordering by branch_id keeps a
# single representative row per def even when the same def is visible in
# multiple of the requested branches.
_DEF_COLS = """\
d.id AS def_id, d.name, d.qualified_name, d.kind, d.visibility,
bf.path AS file_path, fv.id AS file_version_id, b.repo_id,
n.start_row, n.end_row, n.start_byte, n.end_byte"""

# Standard JOIN suffix that reaches branch_files + branches + file_versions
# from a definitions row. Caller adds the WHERE on bf.branch_id.
_DEF_JOINS = """\
JOIN nodes n ON n.id = d.node_id
JOIN file_versions fv ON fv.id = d.file_version_id
JOIN branch_files bf ON bf.file_version_id = fv.id
JOIN branches b ON b.id = bf.branch_id"""


async def _set_timeout(conn: asyncpg.Connection, timeout_s: int) -> None:
    await conn.execute(f"SET LOCAL statement_timeout = '{timeout_s}s'")


# ────────────────────────────────────────────────────────────────────
# Resolve definitions by name
# ────────────────────────────────────────────────────────────────────


async def resolve_definitions(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    kind: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    bids = _norm_branch_ids(branch_ids)
    kind_clause = "AND d.kind = $3" if kind else ""
    params: list = [bids, name]
    if kind:
        params.append(kind)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM definitions d
                {_DEF_JOINS}
                WHERE bf.branch_id = ANY($1::bigint[])
                  AND (d.name = $2 OR d.qualified_name = $2
                       OR d.qualified_name LIKE '%%.' || $2)
                  {kind_clause}
                ORDER BY d.id, bf.branch_id
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


async def _resolve_def_ids(
    conn: asyncpg.Connection,
    branch_ids: list[int],
    name: str,
) -> list[int]:
    rows = await conn.fetch(
        """
        SELECT DISTINCT d.id FROM definitions d
        JOIN branch_files bf ON bf.file_version_id = d.file_version_id
        WHERE bf.branch_id = ANY($1::bigint[])
          AND (d.name = $2 OR d.qualified_name = $2
               OR d.qualified_name LIKE '%.' || $2)
        """,
        branch_ids, name,
    )
    return [r["id"] for r in rows]


# ────────────────────────────────────────────────────────────────────
# Direct neighbors (1-hop)
# ────────────────────────────────────────────────────────────────────


async def callers_of(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    bids = _norm_branch_ids(branch_ids)
    conf_clause = "AND ce.confidence = $3" if confidence else ""
    params: list = [bids, name]
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
                JOIN definitions d ON d.id = ce.caller_def_id
                {_DEF_JOINS}
                WHERE ce.branch_id = ANY($1::bigint[])
                  AND bf.branch_id = ANY($1::bigint[])
                  AND (target.name = $2 OR target.qualified_name = $2
                       OR target.qualified_name LIKE '%%.' || $2)
                  {conf_clause}
                ORDER BY d.id, bf.branch_id
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


async def callees_of(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    bids = _norm_branch_ids(branch_ids)
    conf_clause = "AND ce.confidence = $3" if confidence else ""
    params: list = [bids, name]
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
                JOIN definitions d ON d.id = ce.callee_def_id
                {_DEF_JOINS}
                WHERE ce.branch_id = ANY($1::bigint[])
                  AND bf.branch_id = ANY($1::bigint[])
                  AND (caller.name = $2 OR caller.qualified_name = $2
                       OR caller.qualified_name LIKE '%%.' || $2)
                  {conf_clause}
                ORDER BY d.id, bf.branch_id
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Transitive traversals (recursive CTE)
# ────────────────────────────────────────────────────────────────────


async def ancestors(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    max_depth: int | None = None,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Transitive callers — upward call-graph slice."""
    bids = _norm_branch_ids(branch_ids)
    depth_clause = f"AND anc.depth < {max_depth}" if max_depth is not None else ""
    conf_clause_param_idx = 3
    conf_clause = f"AND ce.confidence = ${conf_clause_param_idx}" if confidence else ""
    params: list = [None, bids]  # placeholder; filled in below
    if confidence:
        params.append(confidence)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            seed_ids = await _resolve_def_ids(conn, bids, name)
            if not seed_ids:
                return []
            params[0] = seed_ids
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE anc AS (
                    SELECT ce.caller_def_id AS def_id, 1 AS depth
                    FROM call_edges ce
                    WHERE ce.callee_def_id = ANY($1::bigint[])
                      AND ce.caller_def_id IS NOT NULL
                      AND ce.branch_id = ANY($2::bigint[])
                      {conf_clause}
                    UNION
                    SELECT ce.caller_def_id, anc.depth + 1
                    FROM call_edges ce
                    JOIN anc ON anc.def_id = ce.callee_def_id
                    WHERE ce.caller_def_id IS NOT NULL
                      AND ce.caller_def_id != ALL($1::bigint[])
                      AND ce.branch_id = ANY($2::bigint[])
                      {depth_clause}
                      {conf_clause}
                )
                SELECT DISTINCT ON (d.id) {_DEF_COLS}, anc.depth
                FROM anc
                JOIN definitions d ON d.id = anc.def_id
                {_DEF_JOINS}
                WHERE bf.branch_id = ANY($2::bigint[])
                ORDER BY d.id, anc.depth, bf.branch_id
                """,
                *params,
            )
    result = []
    for r in rows:
        d = _row_to_def(r)
        d.depth = r["depth"]
        result.append(d)
    return result


async def reachable_from(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    max_depth: int | None = None,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Transitive callees — downward call-graph slice (blast radius)."""
    bids = _norm_branch_ids(branch_ids)
    depth_clause = f"AND reach.depth < {max_depth}" if max_depth is not None else ""
    conf_clause_param_idx = 3
    conf_clause = f"AND ce.confidence = ${conf_clause_param_idx}" if confidence else ""
    params: list = [None, bids]
    if confidence:
        params.append(confidence)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            seed_ids = await _resolve_def_ids(conn, bids, name)
            if not seed_ids:
                return []
            params[0] = seed_ids
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE reach AS (
                    SELECT ce.callee_def_id AS def_id, 1 AS depth
                    FROM call_edges ce
                    WHERE ce.caller_def_id = ANY($1::bigint[])
                      AND ce.callee_def_id IS NOT NULL
                      AND ce.branch_id = ANY($2::bigint[])
                      {conf_clause}
                    UNION
                    SELECT ce.callee_def_id, reach.depth + 1
                    FROM call_edges ce
                    JOIN reach ON reach.def_id = ce.caller_def_id
                    WHERE ce.callee_def_id IS NOT NULL
                      AND ce.callee_def_id != ALL($1::bigint[])
                      AND ce.branch_id = ANY($2::bigint[])
                      {depth_clause}
                      {conf_clause}
                )
                SELECT DISTINCT ON (d.id) {_DEF_COLS}, reach.depth
                FROM reach
                JOIN definitions d ON d.id = reach.def_id
                {_DEF_JOINS}
                WHERE bf.branch_id = ANY($2::bigint[])
                ORDER BY d.id, reach.depth, bf.branch_id
                """,
                *params,
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
    branch_ids: int | list[int],
    source_name: str,
    target_name: str,
    *,
    max_depth: int | None = None,
    max_paths: int = 50,
    timeout_s: int = 120,
) -> list[list[DefInfo]]:
    """All simple call paths between two definitions, scoped to branch set."""
    bids = _norm_branch_ids(branch_ids)
    depth_clause = (
        f"AND array_length(p.path, 1) < {max_depth}"
        if max_depth is not None else ""
    )
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            src_ids = await _resolve_def_ids(conn, bids, source_name)
            dst_ids = await _resolve_def_ids(conn, bids, target_name)
            if not src_ids or not dst_ids:
                return []
            path_rows = await conn.fetch(
                f"""
                WITH RECURSIVE paths AS (
                    SELECT ARRAY[ce.caller_def_id, ce.callee_def_id] AS path
                    FROM call_edges ce
                    WHERE ce.caller_def_id = ANY($1::bigint[])
                      AND ce.callee_def_id IS NOT NULL
                      AND ce.branch_id = ANY($4::bigint[])
                    UNION ALL
                    SELECT p.path || ce.callee_def_id
                    FROM paths p
                    JOIN call_edges ce ON ce.caller_def_id = p.path[array_length(p.path, 1)]
                    WHERE ce.callee_def_id IS NOT NULL
                      AND NOT p.path @> ARRAY[ce.callee_def_id]
                      AND ce.branch_id = ANY($4::bigint[])
                      {depth_clause}
                )
                SELECT path FROM paths
                WHERE path[array_length(path, 1)] = ANY($2::bigint[])
                LIMIT $3
                """,
                src_ids, dst_ids, max_paths, bids,
            )
            if not path_rows:
                return []
            all_ids: set[int] = set()
            for pr in path_rows:
                all_ids.update(pr["path"])
            def_rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM definitions d
                {_DEF_JOINS}
                WHERE d.id = ANY($1::bigint[])
                  AND bf.branch_id = ANY($2::bigint[])
                ORDER BY d.id, bf.branch_id
                """,
                list(all_ids), bids,
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
    branch_ids: int | list[int],
    *,
    kind: str | None = None,
    file_path: str | None = None,
    include_internal: bool = False,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Functions/methods with no internal callers and no override relationships,
    visible in the given branches."""
    bids = _norm_branch_ids(branch_ids)
    clauses: list[str] = []
    params: list = [bids]
    idx = 2
    if kind:
        clauses.append(f"AND d.kind = ${idx}")
        params.append(kind)
        idx += 1
    else:
        clauses.append("AND d.kind IN ('function', 'method')")
    if file_path:
        clauses.append(f"AND bf.path LIKE '%%' || ${idx}")
        params.append(file_path)
        idx += 1
    if not include_internal:
        clauses.append("AND (d.visibility IS NULL OR d.visibility NOT IN ('internal', 'private'))")
        clauses.append("AND bf.path NOT LIKE '%%/interfaces/%%'")
        clauses.append("AND bf.path NOT LIKE '%%/interface/%%'")
    extra = "\n                  ".join(clauses)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM definitions d
                {_DEF_JOINS}
                LEFT JOIN call_edges ce
                    ON ce.callee_def_id = d.id AND ce.branch_id = ANY($1::bigint[])
                LEFT JOIN overrides_edges oe
                    ON oe.child_def_id = d.id AND oe.branch_id = ANY($1::bigint[])
                WHERE bf.branch_id = ANY($1::bigint[])
                  AND ce.id IS NULL
                  AND oe.id IS NULL
                  {extra}
                ORDER BY d.id, bf.branch_id
                """,
                *params,
            )
    return [_row_to_def(r) for r in rows]


async def entrypoint_paths(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    target_name: str,
    *,
    max_depth: int | None = None,
    max_paths: int = 50,
    timeout_s: int = 120,
) -> list[list[DefInfo]]:
    """Call paths from entrypoints to a target definition."""
    anc = await ancestors(pool, branch_ids, target_name, max_depth=max_depth, timeout_s=timeout_s)
    ep = await entrypoints(pool, branch_ids, timeout_s=timeout_s)
    ep_ids = {e.def_id for e in ep}
    entry_ancestors = [a for a in anc if a.def_id in ep_ids]
    if not entry_ancestors:
        return []
    result: list[list[DefInfo]] = []
    for ea in entry_ancestors:
        p = await paths_between(
            pool, branch_ids, ea.qualified_name, target_name,
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
    branch_ids: int | list[int],
    name: str,
    *,
    kind: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Resolve definitions and populate source from raw file content."""
    bids = _norm_branch_ids(branch_ids)
    kind_clause = "AND d.kind = $3" if kind else ""
    params: list = [bids, name]
    if kind:
        params.append(kind)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}, fv.raw_content
                FROM definitions d
                {_DEF_JOINS}
                WHERE bf.branch_id = ANY($1::bigint[])
                  AND (d.name = $2 OR d.qualified_name = $2
                       OR d.qualified_name LIKE '%%.' || $2)
                  {kind_clause}
                ORDER BY d.id, bf.branch_id
                """,
                *params,
            )
    result = []
    for r in rows:
        d = _row_to_def(r)
        raw = (r["raw_content"] or "").encode("utf-8")
        d.source = raw[r["start_byte"]:r["end_byte"]].decode("utf-8", errors="replace")
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
    file_version_id: int


async def file_imports(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    *,
    file_path: str | None = None,
    dep_class: str | None = None,
    timeout_s: int = 120,
) -> list[ImportInfo]:
    bids = _norm_branch_ids(branch_ids)
    clauses: list[str] = []
    params: list = [bids]
    idx = 2
    if file_path:
        clauses.append(f"AND bf.path = ${idx}")
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
                       rbf.path AS resolved_file,
                       bf.path AS file_path, fv.id AS file_version_id
                FROM imports i
                JOIN file_versions fv ON fv.id = i.file_version_id
                JOIN branch_files bf
                    ON bf.file_version_id = fv.id AND bf.branch_id = i.branch_id
                LEFT JOIN branch_files rbf
                    ON rbf.file_version_id = i.resolved_file_version_id
                       AND rbf.branch_id = i.branch_id
                WHERE i.branch_id = ANY($1::bigint[])
                  {extra}
                ORDER BY bf.path, i.import_path
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
            file_version_id=r["file_version_id"],
        )
        for r in rows
    ]


async def file_dependents(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    file_path: str,
    *,
    timeout_s: int = 120,
) -> list[ImportInfo]:
    """Files that import a given file (reverse import lookup)."""
    bids = _norm_branch_ids(branch_ids)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            rows = await conn.fetch(
                """
                SELECT i.import_path, i.imported_names, i.dep_class,
                       tbf.path AS resolved_file,
                       bf.path AS file_path, fv.id AS file_version_id
                FROM imports i
                JOIN file_versions fv ON fv.id = i.file_version_id
                JOIN branch_files bf
                    ON bf.file_version_id = fv.id AND bf.branch_id = i.branch_id
                JOIN branch_files tbf
                    ON tbf.file_version_id = i.resolved_file_version_id
                       AND tbf.branch_id = i.branch_id
                WHERE i.branch_id = ANY($1::bigint[])
                  AND tbf.path = $2
                ORDER BY bf.path
                """,
                bids, file_path,
            )
    return [
        ImportInfo(
            import_path=r["import_path"],
            imported_names=list(r["imported_names"] or []),
            dep_class=r["dep_class"],
            resolved_file=r["resolved_file"],
            file_path=r["file_path"],
            file_version_id=r["file_version_id"],
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
    branch_ids: int | list[int],
    name: str,
    *,
    timeout_s: int = 120,
) -> list[InheritanceNode]:
    """Full inheritance hierarchy (up and down) from a named class, scoped to
    the given branches."""
    bids = _norm_branch_ids(branch_ids)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            seed_ids = await _resolve_def_ids(conn, bids, name)
            if not seed_ids:
                return []
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE tree AS (
                    SELECT child_def_id AS def_id FROM inherits_edges
                    WHERE branch_id = ANY($2::bigint[])
                      AND (child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[]))
                    UNION
                    SELECT base_def_id FROM inherits_edges
                    WHERE branch_id = ANY($2::bigint[])
                      AND (child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[]))
                      AND base_def_id IS NOT NULL
                ),
                walk_up AS (
                    SELECT def_id FROM tree
                    UNION
                    SELECT ie.base_def_id
                    FROM inherits_edges ie
                    JOIN walk_up w ON w.def_id = ie.child_def_id
                    WHERE ie.base_def_id IS NOT NULL
                      AND ie.branch_id = ANY($2::bigint[])
                ),
                walk_down AS (
                    SELECT def_id FROM tree
                    UNION
                    SELECT ie.child_def_id
                    FROM inherits_edges ie
                    JOIN walk_down w ON w.def_id = ie.base_def_id
                    WHERE ie.branch_id = ANY($2::bigint[])
                ),
                all_ids AS (
                    SELECT def_id FROM walk_up
                    UNION
                    SELECT def_id FROM walk_down
                )
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM all_ids a
                JOIN definitions d ON d.id = a.def_id
                {_DEF_JOINS}
                WHERE bf.branch_id = ANY($2::bigint[])
                ORDER BY d.id, bf.branch_id
                """,
                seed_ids, bids,
            )
            all_ids = [r["def_id"] for r in rows]
            if not all_ids:
                all_ids = seed_ids
                rows = await conn.fetch(
                    f"""
                    SELECT DISTINCT ON (d.id) {_DEF_COLS}
                    FROM definitions d
                    {_DEF_JOINS}
                    WHERE d.id = ANY($1::bigint[])
                      AND bf.branch_id = ANY($2::bigint[])
                    ORDER BY d.id, bf.branch_id
                    """,
                    all_ids, bids,
                )
            edge_rows = await conn.fetch(
                """
                SELECT child_def_id, base_def_id, base_name
                FROM inherits_edges
                WHERE branch_id = ANY($2::bigint[])
                  AND (child_def_id = ANY($1::bigint[]) OR base_def_id = ANY($1::bigint[]))
                """,
                all_ids, bids,
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


# ────────────────────────────────────────────────────────────────────
# Reachability & data-flow queries
# ────────────────────────────────────────────────────────────────────


async def is_reachable(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    source_name: str,
    target_name: str,
    *,
    max_depth: int | None = None,
    confidence: str | None = None,
    timeout_s: int = 60,
) -> bool:
    """Boolean: does *any* call path exist from source to target?

    Cheaper than `paths_between` when only a yes/no answer is needed — the
    recursive CTE stops at the first hit via `LIMIT 1`.
    """
    bids = _norm_branch_ids(branch_ids)
    depth_clause = (
        f"AND array_length(p.path, 1) < {max_depth}"
        if max_depth is not None else ""
    )
    conf_clause = "AND ce.confidence = $4" if confidence else ""
    seed_conf_clause = "AND ce.confidence = $4" if confidence else ""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            src_ids = await _resolve_def_ids(conn, bids, source_name)
            dst_ids = await _resolve_def_ids(conn, bids, target_name)
            if not src_ids or not dst_ids:
                return False
            params: list = [src_ids, dst_ids, bids]
            if confidence:
                params.append(confidence)
            row = await conn.fetchrow(
                f"""
                WITH RECURSIVE paths AS (
                    SELECT ARRAY[ce.caller_def_id, ce.callee_def_id] AS path
                    FROM call_edges ce
                    WHERE ce.caller_def_id = ANY($1::bigint[])
                      AND ce.callee_def_id IS NOT NULL
                      AND ce.branch_id = ANY($3::bigint[])
                      {seed_conf_clause}
                    UNION ALL
                    SELECT p.path || ce.callee_def_id
                    FROM paths p
                    JOIN call_edges ce ON ce.caller_def_id = p.path[array_length(p.path, 1)]
                    WHERE ce.callee_def_id IS NOT NULL
                      AND NOT p.path @> ARRAY[ce.callee_def_id]
                      AND ce.branch_id = ANY($3::bigint[])
                      {depth_clause}
                      {conf_clause}
                )
                SELECT 1 FROM paths
                WHERE path[array_length(path, 1)] = ANY($2::bigint[])
                LIMIT 1
                """,
                *params,
            )
    return row is not None


# ────────────────────────────────────────────────────────────────────
# Data-access queries (over the data_access table)
# ────────────────────────────────────────────────────────────────────


async def writers_of(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Definitions that write to (or read+write) the named target.

    The target is typically a state variable; `name` is resolved by the same
    rules as `callers_of` (name, qualified_name, or `*.name` suffix). Returns
    a deduped DefInfo list of the writing functions.
    """
    return await _data_accessors(pool, branch_ids, name, ("write", "readwrite"), timeout_s)


async def readers_of(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    *,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Definitions that read (or read+write) the named target."""
    return await _data_accessors(pool, branch_ids, name, ("read", "readwrite"), timeout_s)


async def _data_accessors(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    name: str,
    access_types: tuple[str, ...],
    timeout_s: int,
) -> list[DefInfo]:
    bids = _norm_branch_ids(branch_ids)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            target_ids = await _resolve_def_ids(conn, bids, name)
            if not target_ids:
                return []
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM data_access da
                JOIN definitions d ON d.id = da.accessor_def_id
                {_DEF_JOINS}
                WHERE da.target_def_id = ANY($1::bigint[])
                  AND da.access_type = ANY($2::text[])
                  AND da.branch_id = ANY($3::bigint[])
                  AND bf.branch_id = ANY($3::bigint[])
                ORDER BY d.id, bf.branch_id
                """,
                target_ids, list(access_types), bids,
            )
    return [_row_to_def(r) for r in rows]


# ────────────────────────────────────────────────────────────────────
# Taint paths (call_edges ∪ data_access) with sanitizer exclusion
# ────────────────────────────────────────────────────────────────────


async def taint_paths(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    source_name: str,
    sink_name: str,
    *,
    sanitizer_names: list[str] | None = None,
    max_depth: int | None = None,
    max_paths: int = 50,
    timeout_s: int = 120,
) -> list[list[DefInfo]]:
    """Paths from source to sink through the UNION of call edges and data-access
    edges, excluding any path that touches a sanitizer definition.

    The walk treats data-access as a regular graph edge: A writes V means
    A → V; B reads V means V → B. So a flow "A writes storage S, B reads S,
    B calls sink" is captured as A → S → B → sink.

    Sanitizer exclusion is applied at the recursive step — paths containing
    a sanitizer def_id are pruned before they're extended further. Returns up
    to `max_paths` simple (acyclic) paths.
    """
    bids = _norm_branch_ids(branch_ids)
    depth_clause = (
        f"AND array_length(p.path, 1) < {max_depth}"
        if max_depth is not None else ""
    )
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _set_timeout(conn, timeout_s)
            src_ids = await _resolve_def_ids(conn, bids, source_name)
            dst_ids = await _resolve_def_ids(conn, bids, sink_name)
            if not src_ids or not dst_ids:
                return []
            sanitizer_ids: list[int] = []
            for sname in sanitizer_names or []:
                sanitizer_ids.extend(await _resolve_def_ids(conn, bids, sname))
            path_rows = await conn.fetch(
                f"""
                WITH RECURSIVE
                edges AS (
                    -- call edges as (from -> to)
                    SELECT caller_def_id AS src, callee_def_id AS dst
                    FROM call_edges
                    WHERE branch_id = ANY($4::bigint[])
                      AND caller_def_id IS NOT NULL
                      AND callee_def_id IS NOT NULL
                    UNION ALL
                    -- data writes: accessor writes -> target
                    SELECT accessor_def_id AS src, target_def_id AS dst
                    FROM data_access
                    WHERE branch_id = ANY($4::bigint[])
                      AND access_type IN ('write', 'readwrite')
                    UNION ALL
                    -- data reads: target -> accessor that reads it
                    SELECT target_def_id AS src, accessor_def_id AS dst
                    FROM data_access
                    WHERE branch_id = ANY($4::bigint[])
                      AND access_type IN ('read', 'readwrite')
                ),
                paths AS (
                    SELECT ARRAY[e.src, e.dst] AS path
                    FROM edges e
                    WHERE e.src = ANY($1::bigint[])
                      AND NOT (e.dst = ANY($5::bigint[]))
                    UNION ALL
                    SELECT p.path || e.dst
                    FROM paths p
                    JOIN edges e ON e.src = p.path[array_length(p.path, 1)]
                    WHERE NOT p.path @> ARRAY[e.dst]
                      AND NOT (e.dst = ANY($5::bigint[]))
                      {depth_clause}
                )
                SELECT path FROM paths
                WHERE path[array_length(path, 1)] = ANY($2::bigint[])
                LIMIT $3
                """,
                src_ids, dst_ids, max_paths, bids, sanitizer_ids,
            )
            if not path_rows:
                return []
            all_ids: set[int] = set()
            for pr in path_rows:
                all_ids.update(pr["path"])
            def_rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (d.id) {_DEF_COLS}
                FROM definitions d
                {_DEF_JOINS}
                WHERE d.id = ANY($1::bigint[])
                  AND bf.branch_id = ANY($2::bigint[])
                ORDER BY d.id, bf.branch_id
                """,
                list(all_ids), bids,
            )
    defs_by_id = {r["def_id"]: _row_to_def(r) for r in def_rows}
    result: list[list[DefInfo]] = []
    for pr in path_rows:
        path = [defs_by_id[did] for did in pr["path"] if did in defs_by_id]
        if path:
            result.append(path)
    return result


# ────────────────────────────────────────────────────────────────────
# Entry-point filter
# ────────────────────────────────────────────────────────────────────


async def entrypoints_reaching(
    pool: asyncpg.Pool,
    branch_ids: int | list[int],
    target_name: str,
    *,
    max_depth: int | None = None,
    confidence: str | None = None,
    timeout_s: int = 120,
) -> list[DefInfo]:
    """Entrypoints from which `target_name` is reachable via the call graph.

    Useful for "who can drain my funds" / "who can trigger this sensitive
    sink" investigations. Computed as `ancestors(target) ∩ entrypoints()`.
    """
    anc = await ancestors(
        pool, branch_ids, target_name,
        max_depth=max_depth, confidence=confidence, timeout_s=timeout_s,
    )
    if not anc:
        return []
    ep = await entrypoints(pool, branch_ids, timeout_s=timeout_s)
    ep_ids = {e.def_id for e in ep}
    return [a for a in anc if a.def_id in ep_ids]
