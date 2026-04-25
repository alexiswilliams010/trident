"""Phase 3: heuristic cross-file import resolution (Architecture §5.3.5).

Pipeline (per repo):
    1. extract_imports         — walk each file's CST, insert rows into `imports`
    2. build_repo_index        — file_index, name_index, package_index, path_index
    3. resolve_imports         — language-specific path math + external classification
    4. link_cross_file         — UPDATE `"references"` and `call_edges` where the
                                 imported name now resolves cross-file

The Deno resolver sandbox (Phase 6+) replaces the language-specific path math
with native resolvers; the table shapes do not change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable

import asyncpg

from .config_loader import LanguageConfig, load_language_config
from .grammar_meta import LANGUAGES


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


@dataclass
class ImportEntry:
    """One import statement worth of info, before resolution."""

    file_id: int                # importer file
    node_id: int                # DB id of the import node
    language: str
    source_rel_path: str        # importer's repo-relative path
    import_path: str            # raw text path: "mypackage.utils", "./Token.sol", "..", "@oz/..."
    imported_names: list[str]   # specific symbols imported (e.g. ["helper", "double"])
    is_relative: bool           # Python: starts with "." ; Solidity: starts with "./" or "../"
    dot_count: int = 0          # Python: leading dots in `from . import …`


@dataclass
class ResolvedImport:
    entry: ImportEntry
    dep_class: str              # 'intra_repo' | 'external' | 'unresolved'
    resolved_file_id: int | None = None
    package_name: str | None = None
    external_dep_id: int | None = None


@dataclass
class RepoIndex:
    """In-memory indexes built from the repo's `files` + `definitions` tables."""

    file_index: dict[str, int]                          # rel_path → file_id (intra-repo only)
    name_index: dict[str, list[tuple[int, int]]]        # def_name → [(file_id, def_id), ...]
    qualified_to_def: dict[tuple[int, str], int]        # (file_id, def_name) → def_id
    package_index_python: dict[str, int]                # dotted module path → file_id
    files_by_id: dict[int, str]                         # file_id → rel_path
    file_languages: dict[int, str]                      # file_id → language
    repo_id: int


@dataclass
class ResolutionStats:
    by_class: dict[str, int] = field(default_factory=dict)  # 'intra_repo' / 'external' / 'unresolved'
    cross_file_refs_resolved: int = 0
    cross_file_calls_resolved: int = 0
    unresolved_paths: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# CST extraction (per language)
# ────────────────────────────────────────────────────────────────────


def _extract_imports_python(file_id: int, source_rel_path: str, ts_root, db_id_for) -> list[ImportEntry]:
    """Pull import statements from a Python CST."""
    out: list[ImportEntry] = []
    for ts in _dfs(ts_root):
        if ts.type == "import_statement":
            # `import a, b.c` — each `name` field is a dotted_name we treat as one import.
            for i in range(ts.child_count):
                if ts.field_name_for_child(i) != "name":
                    continue
                dotted = _text(ts.children[i])
                out.append(
                    ImportEntry(
                        file_id=file_id,
                        node_id=db_id_for[ts.id],
                        language="python",
                        source_rel_path=source_rel_path,
                        import_path=dotted,
                        imported_names=[dotted.split(".")[-1]],
                        is_relative=False,
                    )
                )
        elif ts.type == "import_from_statement":
            mod_node = ts.child_by_field_name("module_name")
            if mod_node is None:
                continue
            is_relative = mod_node.type == "relative_import"
            dot_count = 0
            tail = ""
            if is_relative:
                # `relative_import` -> [import_prefix, dotted_name?]
                # `import_prefix` text is one or more dots: ".", "..", "...".
                for c in mod_node.children:
                    if c.type == "import_prefix":
                        dot_count += _text(c).count(".")
                    elif c.type == "dotted_name":
                        tail = _text(c)
                import_path = "." * dot_count + tail
            else:
                import_path = _text(mod_node)

            names: list[str] = []
            for i in range(ts.child_count):
                if ts.field_name_for_child(i) == "name":
                    names.append(_text(ts.children[i]))
            out.append(
                ImportEntry(
                    file_id=file_id,
                    node_id=db_id_for[ts.id],
                    language="python",
                    source_rel_path=source_rel_path,
                    import_path=import_path,
                    imported_names=names,
                    is_relative=is_relative,
                    dot_count=dot_count,
                )
            )
    return out


def _extract_imports_solidity(file_id: int, source_rel_path: str, ts_root, db_id_for) -> list[ImportEntry]:
    """Pull import_directive nodes from a Solidity CST."""
    out: list[ImportEntry] = []
    for ts in _dfs(ts_root):
        if ts.type != "import_directive":
            continue
        src_node = ts.child_by_field_name("source")
        if src_node is None:
            continue
        raw_path = _strip_quotes(_text(src_node))
        names: list[str] = []
        # `import {Foo, Bar} from "..."` — `import_name` field on each named item.
        for i in range(ts.child_count):
            if ts.field_name_for_child(i) == "import_name":
                names.append(_text(ts.children[i]))
        is_relative = raw_path.startswith("./") or raw_path.startswith("../")
        out.append(
            ImportEntry(
                file_id=file_id,
                node_id=db_id_for[ts.id],
                language="solidity",
                source_rel_path=source_rel_path,
                import_path=raw_path,
                imported_names=names,
                is_relative=is_relative,
            )
        )
    return out


_EXTRACTORS = {
    "python": _extract_imports_python,
    "solidity": _extract_imports_solidity,
}


# ────────────────────────────────────────────────────────────────────
# Indexing
# ────────────────────────────────────────────────────────────────────


def _python_dotted_for(rel_path: str) -> str | None:
    """`mypackage/utils.py` → `mypackage.utils`. `mypackage/__init__.py` → `mypackage`."""
    if not rel_path.endswith(".py"):
        return None
    no_ext = rel_path[:-3]
    parts = no_ext.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


async def build_repo_index(conn: asyncpg.Connection, repo_id: int) -> RepoIndex:
    files = await conn.fetch(
        "SELECT id, path, language FROM files "
        "WHERE repo_id=$1 AND from_dependency=FALSE",
        repo_id,
    )
    defs = await conn.fetch(
        """
        SELECT d.id, d.file_id, d.name
        FROM definitions d JOIN files f ON f.id=d.file_id
        WHERE f.repo_id=$1 AND f.from_dependency=FALSE
        """,
        repo_id,
    )

    file_index: dict[str, int] = {}
    files_by_id: dict[int, str] = {}
    file_languages: dict[int, str] = {}
    pkg_index: dict[str, int] = {}
    for f in files:
        fid = f["id"]
        file_index[f["path"]] = fid
        files_by_id[fid] = f["path"]
        file_languages[fid] = f["language"]
        if f["language"] == "python":
            dotted = _python_dotted_for(f["path"])
            if dotted:
                pkg_index[dotted] = fid

    name_index: dict[str, list[tuple[int, int]]] = {}
    qualified_to_def: dict[tuple[int, str], int] = {}
    for d in defs:
        name_index.setdefault(d["name"], []).append((d["file_id"], d["id"]))
        qualified_to_def[(d["file_id"], d["name"])] = d["id"]

    return RepoIndex(
        file_index=file_index,
        name_index=name_index,
        qualified_to_def=qualified_to_def,
        package_index_python=pkg_index,
        files_by_id=files_by_id,
        file_languages=file_languages,
        repo_id=repo_id,
    )


# ────────────────────────────────────────────────────────────────────
# Resolution (per language)
# ────────────────────────────────────────────────────────────────────


def _resolve_python(entry: ImportEntry, idx: RepoIndex) -> ResolvedImport:
    if entry.is_relative:
        # `from .foo import x` from a/b/main.py → a/b/foo
        # `from ..foo import x` from a/b/main.py → a/foo
        src_dir_parts = list(PurePosixPath(entry.source_rel_path).parts[:-1])
        # Python: 1 dot = current package, 2 dots = parent, etc.
        ascend = entry.dot_count - 1
        if ascend > len(src_dir_parts):
            return ResolvedImport(entry, dep_class="unresolved")
        base_parts = src_dir_parts[: len(src_dir_parts) - ascend]
        tail = entry.import_path.lstrip(".")
        tail_parts = tail.split(".") if tail else []

        # Try: as a module file `<base>/<tail>.py`
        if tail_parts:
            cand = "/".join(base_parts + tail_parts) + ".py"
            if cand in idx.file_index:
                return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.file_index[cand])
            # Try as package __init__.py
            cand = "/".join(base_parts + tail_parts) + "/__init__.py"
            if cand in idx.file_index:
                return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.file_index[cand])
            # Try: each imported name is itself a sibling module (`from . import siblings`)
            # falls through if tail was given but didn't resolve — leave unresolved.
        else:
            # `from . import name` — each imported name is a sibling module.
            # Resolve the FIRST name to populate resolved_file_id (full multi-name handled below).
            for name in entry.imported_names:
                cand = "/".join(base_parts + [name]) + ".py"
                if cand in idx.file_index:
                    return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.file_index[cand])
                cand = "/".join(base_parts + [name]) + "/__init__.py"
                if cand in idx.file_index:
                    return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.file_index[cand])
        return ResolvedImport(entry, "unresolved")

    # Absolute import: try the full dotted path + parent dotted prefixes.
    parts = entry.import_path.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in idx.package_index_python:
            return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.package_index_python[cand])
        parts.pop()

    # Top-level segment isn't local → external (e.g., `import requests`).
    top = entry.import_path.split(".")[0]
    return ResolvedImport(entry, "external", package_name=top)


def _resolve_solidity(entry: ImportEntry, idx: RepoIndex, cfg: LanguageConfig) -> ResolvedImport:
    raw = entry.import_path
    external_prefixes = cfg.imports.external_prefixes if cfg.imports else ()
    dep_dirs = set(cfg.dependency_paths or ())

    # Scoped/prefixed packages (e.g. "@openzeppelin/...") are external.
    for pref in external_prefixes:
        if raw.startswith(pref):
            parts = raw.lstrip("@").split("/")
            pkg = "@" + "/".join(parts[:2]) if raw.startswith("@") and len(parts) >= 2 else parts[0]
            return ResolvedImport(entry, "external", package_name=pkg)

    if entry.is_relative:
        src_dir = PurePosixPath(entry.source_rel_path).parent
        target = _normalize_relative_posix((src_dir / raw).as_posix())
        if target in idx.file_index:
            return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.file_index[target])
        return ResolvedImport(entry, "unresolved")

    # Bare repo-relative path that names a real file (rare but valid).
    if raw in idx.file_index:
        return ResolvedImport(entry, "intra_repo", resolved_file_id=idx.file_index[raw])

    # Path begins with a configured dependency dir ("lib/forge-std/src/Test.sol")
    # → external, with package = the segment immediately after the dep dir.
    parts = raw.split("/")
    if parts and parts[0] in dep_dirs:
        pkg = parts[1] if len(parts) > 1 else parts[0]
        return ResolvedImport(entry, "external", package_name=pkg)

    # No remapping context: a multi-segment name is most likely a Foundry/Hardhat
    # remapping target (e.g. "forge-std/Test.sol"). Best-effort: tag external with
    # package = first segment. Phase 7 (Deno resolver) gets the precise answer.
    if "/" in raw:
        return ResolvedImport(entry, "external", package_name=parts[0])

    return ResolvedImport(entry, "unresolved")


def _normalize_relative_posix(path: str) -> str:
    """`a/b/../c/./d` → `a/c/d`, without touching the filesystem."""
    parts: list[str] = []
    for seg in path.split("/"):
        if seg == "" or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def _resolve_one(entry: ImportEntry, idx: RepoIndex, cfg: LanguageConfig) -> ResolvedImport:
    if entry.language == "python":
        return _resolve_python(entry, idx)
    if entry.language == "solidity":
        return _resolve_solidity(entry, idx, cfg)
    return ResolvedImport(entry, "unresolved")


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────


async def _clear_imports_for_repo(conn: asyncpg.Connection, repo_id: int) -> None:
    await conn.execute(
        "DELETE FROM imports WHERE file_id IN (SELECT id FROM files WHERE repo_id=$1)",
        repo_id,
    )
    await conn.execute("DELETE FROM external_dependencies WHERE repo_id=$1", repo_id)


async def _ensure_external_dep(
    conn: asyncpg.Connection,
    repo_id: int,
    package_name: str,
    language: str,
) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO external_dependencies (repo_id, package_name, language)
        VALUES ($1, $2, $3)
        ON CONFLICT (repo_id, package_name, language) DO UPDATE SET package_name = EXCLUDED.package_name
        RETURNING id
        """,
        repo_id,
        package_name,
        language,
    )
    return row["id"]


async def _insert_imports(
    conn: asyncpg.Connection,
    repo_id: int,
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
                ext_id = await _ensure_external_dep(conn, repo_id, r.package_name, r.entry.language)
                ext_cache[key] = ext_id
        rows.append(
            (
                r.entry.file_id,
                r.entry.node_id,
                r.entry.import_path,
                r.resolved_file_id,
                r.entry.imported_names,
                r.dep_class,
                ext_id,
            )
        )
    if not rows:
        return
    await conn.executemany(
        """
        INSERT INTO imports (file_id, node_id, import_path, resolved_file_id, imported_names, dep_class, external_dep_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        rows,
    )


# ────────────────────────────────────────────────────────────────────
# Cross-file linking
# ────────────────────────────────────────────────────────────────────


async def _link_cross_file(
    conn: asyncpg.Connection,
    repo_id: int,
    idx: RepoIndex,
) -> tuple[int, int]:
    """Two-tier cross-file linking:

    Tier A (certain) — direct imports: each name in `imports.imported_names`
    is matched against the resolved target file's definitions. Hit → set
    target_def_id, mark call_edges.confidence='certain'.

    Tier B (inferred) — imported-file fuzzy fallback: for any name still
    unresolved in an importing file, look across the file's set of
    intra-repo-imported files for a UNIQUE definition with that name. Hit →
    same set updates but with confidence='inferred'. This handles patterns
    like `token.transfer(...)` where `transfer` lives in an imported file
    but is not itself an `imported_name`.

    Returns (refs_updated, call_edges_updated) totals across both tiers.
    """
    intra_imports = await conn.fetch(
        """
        SELECT i.file_id, i.resolved_file_id, i.imported_names
        FROM imports i JOIN files f ON f.id=i.file_id
        WHERE f.repo_id=$1 AND i.dep_class='intra_repo' AND i.resolved_file_id IS NOT NULL
        """,
        repo_id,
    )

    # importer_file_id → set(imported_file_ids)
    imported_files_by: dict[int, set[int]] = {}
    # importer_file_id → name → target_def_id (Tier A direct hits)
    direct_by: dict[int, dict[str, int]] = {}
    for row in intra_imports:
        importer = row["file_id"]
        target_file = row["resolved_file_id"]
        imported_files_by.setdefault(importer, set()).add(target_file)
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
            FROM UNNEST($2::text[], $3::bigint[]) AS u(name, target_def_id)
            WHERE r.file_id = $1 AND r.target_def_id IS NULL AND r.name = u.name
            """,
            importer, names, defs,
        ))
        calls_updated += _affected_rows(await conn.execute(
            """
            UPDATE call_edges AS ce
            SET callee_def_id = u.target_def_id, confidence = 'certain'
            FROM UNNEST($2::text[], $3::bigint[]) AS u(name, target_def_id)
            WHERE ce.callee_def_id IS NULL AND ce.callee_name = u.name
              AND ce.callsite_node_id IN (SELECT id FROM nodes WHERE file_id = $1)
            """,
            importer, names, defs,
        ))

    # ── Tier B: fuzzy match in imported files (inferred) ──
    for importer, imported_file_ids in imported_files_by.items():
        if not imported_file_ids:
            continue
        # Gather still-unresolved names in this file.
        ref_rows = await conn.fetch(
            'SELECT DISTINCT name FROM "references" '
            "WHERE file_id=$1 AND target_def_id IS NULL",
            importer,
        )
        call_rows = await conn.fetch(
            "SELECT DISTINCT ce.callee_name "
            "FROM call_edges ce "
            "WHERE ce.callee_def_id IS NULL AND ce.callee_name IS NOT NULL "
            "  AND ce.callsite_node_id IN (SELECT id FROM nodes WHERE file_id=$1)",
            importer,
        )
        unresolved_names = {r["name"] for r in ref_rows} | {r["callee_name"] for r in call_rows}

        fuzzy_pairs: dict[str, int] = {}
        for name in unresolved_names:
            cands = idx.name_index.get(name, [])
            in_imported = [(fid, did) for fid, did in cands if fid in imported_file_ids]
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
            FROM UNNEST($2::text[], $3::bigint[]) AS u(name, target_def_id)
            WHERE r.file_id = $1 AND r.target_def_id IS NULL AND r.name = u.name
            """,
            importer, names, defs,
        ))
        calls_updated += _affected_rows(await conn.execute(
            """
            UPDATE call_edges AS ce
            SET callee_def_id = u.target_def_id, confidence = 'inferred'
            FROM UNNEST($2::text[], $3::bigint[]) AS u(name, target_def_id)
            WHERE ce.callee_def_id IS NULL AND ce.callee_name = u.name
              AND ce.callsite_node_id IN (SELECT id FROM nodes WHERE file_id = $1)
            """,
            importer, names, defs,
        ))

    return refs_updated, calls_updated


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


async def resolve_repo_imports(pool: asyncpg.Pool, repo_id: int) -> ResolutionStats:
    stats = ResolutionStats()
    configs: dict[str, LanguageConfig] = {}

    async with pool.acquire() as conn:
        files = await conn.fetch(
            "SELECT id, path, language, raw_content FROM files "
            "WHERE repo_id=$1 AND from_dependency=FALSE",
            repo_id,
        )

        await _clear_imports_for_repo(conn, repo_id)

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
            extractor = _EXTRACTORS.get(lang)
            if extractor is None:
                continue

            source = (f["raw_content"] or "").encode("utf-8")
            parser = LANGUAGES[lang].parser()
            tree = parser.parse(source)

            # Pair ts_nodes to DB ids via the same DFS preorder used by Tier 1.
            ts_walk = list(_dfs(tree.root_node))
            db_ids = await conn.fetch(
                "SELECT id FROM nodes WHERE file_id=$1 ORDER BY id",
                f["id"],
            )
            if len(ts_walk) != len(db_ids):
                # CST size mismatch — skip this file rather than corrupt the imports table.
                continue
            db_id_for = {ts.id: db_ids[i]["id"] for i, ts in enumerate(ts_walk)}

            entries = extractor(f["id"], f["path"], tree.root_node, db_id_for)
            all_entries.extend(entries)

        idx = await build_repo_index(conn, repo_id)

        resolved: list[ResolvedImport] = []
        for entry in all_entries:
            cfg = configs.get(entry.language)
            assert cfg is not None
            resolved.append(_resolve_one(entry, idx, cfg))
            stats.by_class[resolved[-1].dep_class] = stats.by_class.get(resolved[-1].dep_class, 0) + 1
            if resolved[-1].dep_class == "unresolved":
                stats.unresolved_paths.append(f"{entry.source_rel_path}: {entry.import_path}")

        await _insert_imports(conn, repo_id, resolved)
        refs_updated, calls_updated = await _link_cross_file(conn, repo_id, idx)
        stats.cross_file_refs_resolved = refs_updated
        stats.cross_file_calls_resolved = calls_updated

    return stats


def resolve_repo_imports_sync(repo_id: int, dsn: str | None = None) -> ResolutionStats:
    from db.connection import pool_ctx

    async def _run() -> ResolutionStats:
        async with pool_ctx(dsn) as pool:
            return await resolve_repo_imports(pool, repo_id)

    return asyncio.run(_run())
