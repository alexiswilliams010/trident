"""Phase 3: heuristic cross-file import resolution (Architecture §5.3.5).

Pipeline (per branch — branches are the unit of cross-file resolution):
    1. extract_imports         — walk each file_version's CST, insert rows into `imports`
    2. build_branch_index      — file_index, name_index, package_index, path_index
    3. resolve_imports         — language-specific path math + external classification
    4. link_cross_file         — UPDATE `"references"` and `call_edges` where the
                                 imported name now resolves cross-file

Branch-aware model: file_versions / nodes / definitions are content-shared
across branches, but cross-file resolution outputs (imports, refs/call_edges
target_def_id fills, inherits_resolutions, overrides_edges, external_deps)
are per-branch because they depend on the set of files visible in the branch.
Every row this module writes is stamped with `branch_id`.

The Deno resolver sandbox (Phase 6+) replaces the language-specific path math
with native resolvers; the table shapes do not change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import asyncpg

from .config_loader import LanguageConfig, load_language_config
from .grammar_meta import LANGUAGES
from .languages import ImportEntry, ResolvedImport, get_handler


# ────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────


def _text(ts_node) -> str:
    return ts_node.text.decode("utf-8", errors="replace")


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _dfs(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for i in range(n.child_count - 1, -1, -1):
            stack.append(n.children[i])


# ────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────
# ImportEntry / ResolvedImport live in core.languages.base — they're the
# contract between handlers and this resolver. Re-exported here only for
# backwards-compatibility with existing callers.


@dataclass
class BranchIndex:
    """In-memory indexes built from the branch's `branch_files` + `definitions` tables.

    Every dict is keyed by file_version_id (since branches are mappings from
    paths to file_version_ids). Two repos / branches won't collide because
    file_version_ids are globally unique and we only load ones reachable from
    this branch.

    Per-language state (Python's dotted-module index, Go's package index,
    Rust's crate/module map, tsconfig.json paths, etc.) lives in
    `lang_state[<language>]` as a handler-defined dataclass. See
    `core/languages/<lang>/index.py` for each language's state shape.
    """

    branch_id: int
    repo_id: int
    file_index: dict[str, int]                          # rel_path → file_version_id (intra-repo only)
    name_index: dict[str, list[tuple[int, int]]]        # def_name → [(file_version_id, def_id), ...]
    qualified_to_def: dict[tuple[int, str], int]        # (file_version_id, def_name) → def_id
    files_by_id: dict[int, str]                         # file_version_id → rel_path
    file_languages: dict[int, str]                      # file_version_id → language
    lang_state: dict[str, Any] = field(default_factory=dict)


# Backwards-compatible alias for the old name; some test code may reference it.
RepoIndex = BranchIndex


@dataclass
class ResolutionStats:
    by_class: dict[str, int] = field(default_factory=dict)  # 'intra_repo' / 'external' / 'unresolved'
    cross_file_refs_resolved: int = 0
    cross_file_calls_resolved: int = 0
    cross_file_inherits_resolved: int = 0
    overrides_inserted: int = 0
    unresolved_paths: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# CST extraction (per language)
# ────────────────────────────────────────────────────────────────────
# Per-language extractors live under core/languages/<lang>/imports.py;
# dispatch goes through `get_handler(lang).extract_imports(...)` in the
# resolve_branch_imports loop below.


# ────────────────────────────────────────────────────────────────────
# Indexing
# ────────────────────────────────────────────────────────────────────




async def build_branch_index(
    conn: asyncpg.Connection, repo_id: int, branch_id: int,
) -> BranchIndex:
    """Build an in-memory index of the branch's intra-repo files + their defs.

    All file lookups are scoped to `branch_files` for `branch_id` to ensure
    cross-file resolution sees only the file_versions visible in the branch.
    Definitions are content-shared, but we only load the ones whose
    file_version is mapped by this branch.

    Per-language state is delegated to each handler via `init_state`,
    `index_file`, and `finalize_index` — this function stays language-agnostic.
    """
    files = await conn.fetch(
        """
        SELECT fv.id, bf.path, fv.language
        FROM branch_files bf
        JOIN file_versions fv ON fv.id = bf.file_version_id
        WHERE bf.branch_id = $1 AND bf.from_dependency = FALSE
        """,
        branch_id,
    )
    defs = await conn.fetch(
        """
        SELECT d.id, d.file_version_id, d.name
        FROM definitions d
        JOIN branch_files bf ON bf.file_version_id = d.file_version_id
        WHERE bf.branch_id = $1 AND bf.from_dependency = FALSE
        """,
        branch_id,
    )
    repo_row = await conn.fetchrow(
        "SELECT root_path FROM repos WHERE id=$1", repo_id,
    )
    repo_root = Path(repo_row["root_path"]) if repo_row and repo_row["root_path"] else None

    file_index: dict[str, int] = {}
    files_by_id: dict[int, str] = {}
    file_languages: dict[int, str] = {}
    lang_state: dict[str, Any] = {}
    for f in files:
        fvid = f["id"]
        rel = f["path"]
        lang = f["language"]
        file_index[rel] = fvid
        files_by_id[fvid] = rel
        file_languages[fvid] = lang
        try:
            handler = get_handler(lang)
        except KeyError:
            continue
        if lang not in lang_state:
            lang_state[lang] = handler.init_state()
        handler.index_file(fvid, rel, lang_state[lang])

    for lang, state in lang_state.items():
        get_handler(lang).finalize_index(repo_root, state)

    name_index: dict[str, list[tuple[int, int]]] = {}
    qualified_to_def: dict[tuple[int, str], int] = {}
    for d in defs:
        name_index.setdefault(d["name"], []).append((d["file_version_id"], d["id"]))
        qualified_to_def[(d["file_version_id"], d["name"])] = d["id"]

    return BranchIndex(
        branch_id=branch_id,
        repo_id=repo_id,
        file_index=file_index,
        name_index=name_index,
        qualified_to_def=qualified_to_def,
        files_by_id=files_by_id,
        file_languages=file_languages,
        lang_state=lang_state,
    )


# Backwards-compatible alias for the old name.
build_repo_index = build_branch_index


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────


async def _clear_imports_for_branch(conn: asyncpg.Connection, branch_id: int) -> None:
    """Drop all imports + external_dependencies rows for a branch. Called at
    the start of each cross-file resolution pass so re-runs are idempotent."""
    await conn.execute("DELETE FROM imports WHERE branch_id=$1", branch_id)
    await conn.execute("DELETE FROM external_dependencies WHERE branch_id=$1", branch_id)


async def _ensure_external_dep(
    conn: asyncpg.Connection,
    branch_id: int,
    package_name: str,
    language: str,
) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO external_dependencies (branch_id, package_name, language)
        VALUES ($1, $2, $3)
        ON CONFLICT (branch_id, package_name, language) DO UPDATE SET package_name = EXCLUDED.package_name
        RETURNING id
        """,
        branch_id,
        package_name,
        language,
    )
    return row["id"]


async def _insert_imports(
    conn: asyncpg.Connection,
    branch_id: int,
    resolved: Iterable[ResolvedImport],
) -> None:
    rows: list[tuple] = []
    ext_cache: dict[tuple[str, str], int] = {}
    for r in resolved:
        ext_id: int | None = None
        if r.dep_class == "external" and r.package_name:
            key = (r.package_name, r.entry.language)
            ext_id = ext_cache.get(key)
            if ext_id is None:
                ext_id = await _ensure_external_dep(conn, branch_id, r.package_name, r.entry.language)
                ext_cache[key] = ext_id
        rows.append(
            (
                branch_id,
                r.entry.file_version_id,
                r.entry.node_id,
                r.entry.import_path,
                r.resolved_file_version_id,
                r.entry.imported_names,
                r.dep_class,
                ext_id,
            )
        )
    if not rows:
        return
    await conn.executemany(
        """
        INSERT INTO imports (branch_id, file_version_id, node_id, import_path, resolved_file_version_id, imported_names, dep_class, external_dep_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        rows,
    )


# ────────────────────────────────────────────────────────────────────
# Cross-file linking
# ────────────────────────────────────────────────────────────────────


async def _link_cross_file(
    conn: asyncpg.Connection,
    branch_id: int,
    idx: BranchIndex,
) -> tuple[int, int, dict[int, set[int]]]:
    """Two-tier cross-file linking, scoped to one branch:

    Tier A (certain) — direct imports: each name in `imports.imported_names`
    is matched against the resolved target file's definitions. Hit → set
    target_def_id, mark call_edges.confidence='certain'.

    Tier B (inferred) — imported-file fuzzy fallback: for any name still
    unresolved in an importing file, look across the file's set of
    intra-repo-imported files for a UNIQUE definition with that name. Hit →
    same set updates but with confidence='inferred'. This handles patterns
    like `token.transfer(...)` where `transfer` lives in an imported file
    but is not itself an `imported_name`.

    Returns (refs_updated, call_edges_updated, imported_files_by) where
    imported_files_by is keyed by importer file_version_id and gives the set
    of intra-repo-imported file_version_ids visible to that importer in this
    branch.
    """
    intra_imports = await conn.fetch(
        """
        SELECT i.file_version_id, i.resolved_file_version_id, i.imported_names
        FROM imports i
        WHERE i.branch_id=$1 AND i.dep_class='intra_repo' AND i.resolved_file_version_id IS NOT NULL
        """,
        branch_id,
    )

    # importer_file_version_id → set(imported_file_version_ids)
    imported_files_by: dict[int, set[int]] = {}

    # Implicit-import seeding (e.g. Go: every file in a package implicitly
    # imports its peers). Each handler that needs this overrides
    # `seed_implicit_imports`; the default is a no-op.
    for lang_name in idx.lang_state:
        seeded = get_handler(lang_name).seed_implicit_imports(idx)
        for importer, targets in seeded.items():
            imported_files_by.setdefault(importer, set()).update(targets)

    # importer_file_version_id → name → target_def_id (Tier A direct hits)
    direct_by: dict[int, dict[str, int]] = {}
    for row in intra_imports:
        importer = row["file_version_id"]
        target_file = row["resolved_file_version_id"]
        # Expand the import's effective target set. Most languages: just the
        # target file. Go: every sibling .go file in the imported package, so
        # Tier-B fuzzy matching can match symbols defined in any peer.
        target_lang = idx.file_languages.get(target_file)
        if target_lang is not None:
            target_files = get_handler(target_lang).expand_import_target(target_file, idx)
        else:
            target_files = {target_file}
        imported_files_by.setdefault(importer, set()).update(target_files)
        for name in (row["imported_names"] or []):
            target_def_id = idx.qualified_to_def.get((target_file, name))
            if target_def_id is not None:
                direct_by.setdefault(importer, {})[name] = target_def_id

    refs_updated = 0
    calls_updated = 0

    # ── Tier A: direct imports (certain) ──
    for importer, name_to_def in direct_by.items():
        if not name_to_def:
            continue
        names = list(name_to_def.keys())
        defs = [name_to_def[n] for n in names]
        refs_updated += _affected_rows(await conn.execute(
            """
            UPDATE "references" AS r
            SET target_def_id = u.target_def_id,
                resolution_confidence = 0.7
            FROM UNNEST($3::text[], $4::bigint[]) AS u(name, target_def_id)
            WHERE r.branch_id = $1 AND r.file_version_id = $2
              AND r.target_def_id IS NULL AND r.name = u.name
            """,
            branch_id, importer, names, defs,
        ))
        calls_updated += _affected_rows(await conn.execute(
            """
            UPDATE call_edges AS ce
            SET callee_def_id = u.target_def_id, confidence = 'certain'
            FROM UNNEST($3::text[], $4::bigint[]) AS u(name, target_def_id)
            WHERE ce.branch_id = $1 AND ce.callee_def_id IS NULL AND ce.callee_name = u.name
              AND ce.callsite_node_id IN (SELECT id FROM nodes WHERE file_version_id = $2)
            """,
            branch_id, importer, names, defs,
        ))

    # ── Tier B: fuzzy match in imported files (inferred) ──
    for importer, imported_file_ids in imported_files_by.items():
        if not imported_file_ids:
            continue
        # Gather still-unresolved names in this file (within this branch).
        ref_rows = await conn.fetch(
            'SELECT DISTINCT name FROM "references" '
            "WHERE branch_id=$1 AND file_version_id=$2 AND target_def_id IS NULL",
            branch_id, importer,
        )
        call_rows = await conn.fetch(
            "SELECT DISTINCT ce.callee_name "
            "FROM call_edges ce "
            "WHERE ce.branch_id=$1 AND ce.callee_def_id IS NULL AND ce.callee_name IS NOT NULL "
            "  AND ce.callsite_node_id IN (SELECT id FROM nodes WHERE file_version_id=$2)",
            branch_id, importer,
        )
        unresolved_names = {r["name"] for r in ref_rows} | {r["callee_name"] for r in call_rows}

        fuzzy_pairs: dict[str, int] = {}
        for name in unresolved_names:
            cands = idx.name_index.get(name, [])
            in_imported = [(fvid, did) for fvid, did in cands if fvid in imported_file_ids]
            if len(in_imported) == 1:
                fuzzy_pairs[name] = in_imported[0][1]

        if not fuzzy_pairs:
            continue
        names = list(fuzzy_pairs.keys())
        defs = [fuzzy_pairs[n] for n in names]
        refs_updated += _affected_rows(await conn.execute(
            """
            UPDATE "references" AS r
            SET target_def_id = u.target_def_id,
                resolution_confidence = 0.5
            FROM UNNEST($3::text[], $4::bigint[]) AS u(name, target_def_id)
            WHERE r.branch_id = $1 AND r.file_version_id = $2
              AND r.target_def_id IS NULL AND r.name = u.name
            """,
            branch_id, importer, names, defs,
        ))
        calls_updated += _affected_rows(await conn.execute(
            """
            UPDATE call_edges AS ce
            SET callee_def_id = u.target_def_id, confidence = 'inferred'
            FROM UNNEST($3::text[], $4::bigint[]) AS u(name, target_def_id)
            WHERE ce.branch_id = $1 AND ce.callee_def_id IS NULL AND ce.callee_name = u.name
              AND ce.callsite_node_id IN (SELECT id FROM nodes WHERE file_version_id = $2)
            """,
            branch_id, importer, names, defs,
        ))

    return refs_updated, calls_updated, imported_files_by


async def _link_cross_file_inheritance(
    conn: asyncpg.Connection,
    branch_id: int,
    idx: BranchIndex,
    imported_files_by: dict[int, set[int]],
) -> int:
    """Resolve `inherits_edges.base_def_id` for rows where the base definition
    lives in a different file. Mirrors the call/reference linker:

      Tier A — direct hit on an inheritance-eligible def in an imported file: certain.
      Tier B — multiple candidates: pick first, mark inferred.

    The name lookup is kind-filtered to {contract, interface, class}. The
    repo-wide `name_index` would otherwise also surface synthetic module defs
    (file `Policy.sol` produces a module def named `Policy`, same as the
    contract), and `is Policy` would ambiguously match both.

    Returns the number of edges newly resolved.
    """
    pending = await conn.fetch(
        """
        SELECT ie.id, ie.child_def_id, ie.base_name, d.file_version_id AS child_file_version_id
        FROM inherits_edges ie
        JOIN definitions d ON d.id = ie.child_def_id
        WHERE ie.branch_id = $1 AND ie.base_def_id IS NULL
        """,
        branch_id,
    )
    if not pending:
        return 0

    target_rows = await conn.fetch(
        """
        SELECT d.id, d.file_version_id, d.name
        FROM definitions d
        JOIN branch_files bf ON bf.file_version_id = d.file_version_id
        WHERE bf.branch_id = $1 AND d.kind IN ('contract', 'interface', 'class', 'type')
        """,
        branch_id,
    )
    targets_by_name: dict[str, list[tuple[int, int]]] = {}
    for r in target_rows:
        targets_by_name.setdefault(r["name"], []).append((r["file_version_id"], r["id"]))

    updates: list[tuple[int, int, str]] = []  # (edge_id, base_def_id, confidence)
    for r in pending:
        name = r["base_name"]
        child_file = r["child_file_version_id"]
        imported = imported_files_by.get(child_file, set())
        cands = targets_by_name.get(name, [])
        in_imported = [(fvid, did) for fvid, did in cands if fvid in imported]
        if len(in_imported) == 1:
            updates.append((r["id"], in_imported[0][1], "certain"))
        elif len(in_imported) > 1:
            updates.append((r["id"], in_imported[0][1], "inferred"))

    if not updates:
        return 0
    await conn.executemany(
        "UPDATE inherits_edges SET base_def_id=$2, confidence=$3 WHERE id=$1",
        updates,
    )
    return len(updates)


async def _generate_overrides(
    conn: asyncpg.Connection,
    branch_id: int,
) -> int:
    """For each resolved (child_class, base_class) inheritance pair in this
    branch, find method/function/modifier defs inside the child whose `name`
    matches one in the base (or any transitive ancestor — closest match wins).
    Insert into overrides_edges tagged with branch_id. Pre-clears the branch's
    overrides so re-runs are idempotent.
    """
    # Drop any pre-existing rows scoped to this branch.
    await conn.execute(
        "DELETE FROM overrides_edges WHERE branch_id = $1",
        branch_id,
    )

    inh_resolved = await conn.fetch(
        """
        SELECT ie.child_def_id AS child_class, ie.base_def_id AS base_class
        FROM inherits_edges ie
        WHERE ie.branch_id = $1 AND ie.base_def_id IS NOT NULL
        ORDER BY ie.child_def_id, ie.ord
        """,
        branch_id,
    )
    if not inh_resolved:
        return 0

    # child_class → ordered list of direct bases (preserve declaration order).
    direct_bases: dict[int, list[int]] = {}
    for r in inh_resolved:
        direct_bases.setdefault(r["child_class"], []).append(r["base_class"])

    # Per-class methods: scope_id → [(name, kind, def_id), ...].
    # Definitions are content-shared, so we only restrict to those whose
    # file_version is mapped by this branch.
    method_rows = await conn.fetch(
        """
        SELECT d.id, d.name, d.kind, d.scope_id
        FROM definitions d
        JOIN branch_files bf ON bf.file_version_id = d.file_version_id
        WHERE bf.branch_id = $1
          AND d.kind IN ('function', 'method', 'modifier', 'constructor')
          AND d.scope_id IS NOT NULL
        """,
        branch_id,
    )
    methods_by_class: dict[int, list[asyncpg.Record]] = {}
    for m in method_rows:
        methods_by_class.setdefault(m["scope_id"], []).append(m)

    def _ancestors_in_order(cls: int) -> list[int]:
        """Linearised ancestor list (BFS over direct_bases). Closest first."""
        seen: set[int] = set()
        out: list[int] = []
        frontier = list(direct_bases.get(cls, []))
        while frontier:
            nxt: list[int] = []
            for a in frontier:
                if a in seen:
                    continue
                seen.add(a)
                out.append(a)
                nxt.extend(direct_bases.get(a, []))
            frontier = nxt
        return out

    pairs: list[tuple[int, int]] = []
    for child_class in direct_bases:
        ancestors = _ancestors_in_order(child_class)
        if not ancestors:
            continue
        child_methods = methods_by_class.get(child_class, [])
        for cm in child_methods:
            # Walk ancestors closest-first; first matching name+kind wins.
            for anc in ancestors:
                anc_methods = methods_by_class.get(anc, [])
                match = next(
                    (am for am in anc_methods if am["name"] == cm["name"] and am["kind"] == cm["kind"]),
                    None,
                )
                if match is not None:
                    pairs.append((cm["id"], match["id"]))
                    break

    if not pairs:
        return 0
    await conn.executemany(
        "INSERT INTO overrides_edges (branch_id, child_def_id, base_def_id) VALUES ($1, $2, $3) "
        "ON CONFLICT (branch_id, child_def_id, base_def_id) DO NOTHING",
        [(branch_id, c, b) for c, b in pairs],
    )
    return len(pairs)


def _affected_rows(execute_status: str) -> int:
    """asyncpg `execute()` returns a status string like 'UPDATE 5'."""
    parts = execute_status.split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


# ────────────────────────────────────────────────────────────────────
# Top-level driver
# ────────────────────────────────────────────────────────────────────


async def resolve_branch_imports(
    pool: asyncpg.Pool, repo_id: int, branch_id: int,
) -> ResolutionStats:
    stats = ResolutionStats()
    configs: dict[str, LanguageConfig] = {}

    async with pool.acquire() as conn:
        files = await conn.fetch(
            """
            SELECT fv.id, bf.path, fv.language, fv.raw_content
            FROM branch_files bf
            JOIN file_versions fv ON fv.id = bf.file_version_id
            WHERE bf.branch_id = $1 AND bf.from_dependency = FALSE
            """,
            branch_id,
        )

        await _clear_imports_for_branch(conn, branch_id)

        all_entries: list[ImportEntry] = []
        for f in files:
            lang = f["language"]
            if lang not in configs:
                try:
                    configs[lang] = load_language_config(lang)
                except FileNotFoundError:
                    configs[lang] = None  # type: ignore
            cfg = configs.get(lang)
            if cfg is None or cfg.imports is None or not cfg.imports.node_types:
                continue
            try:
                handler = get_handler(lang)
            except KeyError:
                continue

            source = (f["raw_content"] or "").encode("utf-8")
            parser = LANGUAGES[lang].parser(PurePosixPath(f["path"]).suffix.lower())
            tree = parser.parse(source)

            # Pair ts_nodes to DB ids via the same DFS preorder used by Tier 1.
            ts_walk = list(_dfs(tree.root_node))
            db_ids = await conn.fetch(
                "SELECT id FROM nodes WHERE file_version_id=$1 ORDER BY id",
                f["id"],
            )
            if len(ts_walk) != len(db_ids):
                # CST size mismatch — skip this file rather than corrupt the imports table.
                continue
            db_id_for = {ts.id: db_ids[i]["id"] for i, ts in enumerate(ts_walk)}

            entries = handler.extract_imports(f["id"], f["path"], tree.root_node, db_id_for)
            all_entries.extend(entries)

        idx = await build_branch_index(conn, repo_id, branch_id)

        resolved: list[ResolvedImport] = []
        for entry in all_entries:
            cfg = configs.get(entry.language)
            assert cfg is not None
            resolved.append(get_handler(entry.language).resolve(entry, idx, cfg))
            stats.by_class[resolved[-1].dep_class] = stats.by_class.get(resolved[-1].dep_class, 0) + 1
            if resolved[-1].dep_class == "unresolved":
                stats.unresolved_paths.append(f"{entry.source_rel_path}: {entry.import_path}")

        await _insert_imports(conn, branch_id, resolved)
        refs_updated, calls_updated, imported_files_by = await _link_cross_file(conn, branch_id, idx)
        stats.cross_file_refs_resolved = refs_updated
        stats.cross_file_calls_resolved = calls_updated

        # Inheritance: cross-file linking + override generation. Both are
        # branch-scoped (semantic_resolver wrote the rows with intra-file
        # base_def_id where possible; we fill in the rest).
        stats.cross_file_inherits_resolved = await _link_cross_file_inheritance(
            conn, branch_id, idx, imported_files_by,
        )
        stats.overrides_inserted = await _generate_overrides(conn, branch_id)

    return stats


# Backwards-compatible alias for the old name (callers passing only repo_id
# would now also need branch_id; keep the old name as a typo trap).
resolve_repo_imports = resolve_branch_imports


def resolve_branch_imports_sync(
    repo_id: int, branch_id: int, dsn: str | None = None,
) -> ResolutionStats:
    from db.connection import pool_ctx

    async def _run() -> ResolutionStats:
        async with pool_ctx(dsn) as pool:
            return await resolve_branch_imports(pool, repo_id, branch_id)

    return asyncio.run(_run())


# Backwards-compatible alias.
resolve_repo_imports_sync = resolve_branch_imports_sync
