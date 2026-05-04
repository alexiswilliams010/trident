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
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

import asyncpg

from .config_loader import LanguageConfig, load_language_config
from .grammar_meta import LANGUAGES
from .node_resolution import (
    TsconfigPaths,
    is_relative_specifier,
    load_tsconfig_paths,
    package_name_for_specifier,
    resolve_relative as _resolve_relative_node,
    resolve_tsconfig_alias as _resolve_tsconfig_alias,
)


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
    package_index_go: dict[str, int]                    # module-relative pkg dir → representative file_id
    go_pkg_files: dict[str, list[int]]                  # module-relative pkg dir → every file_id in that pkg
    go_module_path: str | None                          # value from `module …` line in go.mod, if present
    files_by_id: dict[int, str]                         # file_id → rel_path
    file_languages: dict[int, str]                      # file_id → language
    repo_id: int
    # JS/TS only: parsed `compilerOptions.paths` from tsconfig.json (None if no
    # tsconfig present or no JS/TS files in the repo).
    node_tsconfig: TsconfigPaths | None = None
    # Rust only — crate-relative module path → file_id (e.g. "utils" → utils.rs's
    # id, "foo::bar" → src/foo/bar.rs or src/foo/bar/mod.rs's id, "" → the crate
    # root file lib.rs or main.rs). Populated only when the repo has a parseable
    # Cargo.toml at root.
    package_index_rust: dict[str, int] = field(default_factory=dict)
    # Rust only — file_id → its crate-relative module path. Used to resolve
    # `super::` and `self::` paths against the importer's location.
    rust_module_for_file: dict[int, str] = field(default_factory=dict)
    # Rust only — value from `[package].name` in Cargo.toml. Used to recognise
    # absolute paths that name the current crate explicitly (`use myapp::utils`).
    rust_crate_name: str | None = None


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


def _extract_imports_go(file_id: int, source_rel_path: str, ts_root, db_id_for) -> list[ImportEntry]:
    """Pull import_spec nodes (one per imported path) from a Go CST.

    Both `import "fmt"` and grouped `import ( "fmt"; alias "x/y" )` produce
    import_spec nodes; the latter wraps them in an import_spec_list. The DFS
    walks both shapes uniformly. Each spec carries one path; we attach the row
    to the spec's node_id (not the enclosing import_declaration) so each
    imported path has its own row.
    """
    out: list[ImportEntry] = []
    for ts in _dfs(ts_root):
        if ts.type != "import_spec":
            continue
        path_node = ts.child_by_field_name("path")
        if path_node is None:
            continue
        raw = _strip_quotes(_text(path_node))
        # Optional `alias "x/y"` form. Skip blank-import (`_`) and dot-import (`.`)
        # — they don't bind a name we can resolve references against.
        alias_node = ts.child_by_field_name("name")
        if alias_node is not None and alias_node.type == "package_identifier":
            imported = _text(alias_node)
        else:
            imported = raw.rsplit("/", 1)[-1]
        out.append(
            ImportEntry(
                file_id=file_id,
                node_id=db_id_for[ts.id],
                language="go",
                source_rel_path=source_rel_path,
                import_path=raw,
                imported_names=[imported],
                is_relative=False,
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


def _rust_path_text(path_node) -> str:
    """Reconstruct a use-path string from a tree-sitter-rust path-shaped node.
    Handles identifier, type_identifier, crate / self / super, scoped_identifier,
    metavariable. Falls back to the node's raw text for anything else."""
    t = path_node.type
    if t in ("identifier", "type_identifier", "metavariable"):
        return _text(path_node)
    if t in ("crate", "self", "super"):
        return _text(path_node)
    if t == "scoped_identifier":
        p = path_node.child_by_field_name("path")
        n = path_node.child_by_field_name("name")
        prefix = _rust_path_text(p) if p is not None else ""
        suffix = _text(n) if n is not None else ""
        if prefix and suffix:
            return f"{prefix}::{suffix}"
        return prefix or suffix
    return _text(path_node)


def _flatten_rust_use(arg_node, prefix: str):
    """Yield (full_path, last_name_or_None) leaves from a use_declaration's
    argument. None signifies a wildcard (`use a::*` / `a::{*}`).

    Examples (top-level call uses prefix=''):
        use a::b::c;            → ('a::b::c', 'c')
        use a::{b, c};          → ('a::b', 'b'), ('a::c', 'c')
        use a::{b::c, d};       → ('a::b::c', 'c'), ('a::d', 'd')
        use a::*;               → ('a', None)
        use a::B as Bb;         → ('a::B', 'B')   # original name, not alias
    """
    t = arg_node.type
    if t == "use_as_clause":
        p = arg_node.child_by_field_name("path")
        if p is not None:
            yield from _flatten_rust_use(p, prefix)
        return
    if t == "use_wildcard":
        # The path child (if any) is the only named child apart from the
        # implicit `*` token. `use foo::*;` parses path='foo'; bare `use *;` is
        # not legal Rust, so we always expect one.
        p = next((c for c in arg_node.children if c.is_named), None)
        if p is None:
            yield (prefix, None)
            return
        path_str = _rust_path_text(p)
        full = f"{prefix}::{path_str}" if prefix and path_str else (path_str or prefix)
        yield (full, None)
        return
    if t == "scoped_use_list":
        p = arg_node.child_by_field_name("path")
        list_node = arg_node.child_by_field_name("list")
        path_str = _rust_path_text(p) if p is not None else ""
        new_prefix = (
            f"{prefix}::{path_str}" if prefix and path_str
            else (path_str or prefix)
        )
        if list_node is None:
            return
        for c in list_node.children:
            if c.is_named:
                yield from _flatten_rust_use(c, new_prefix)
        return
    # Leaf: identifier / type_identifier / scoped_identifier / crate / self / super.
    path_str = _rust_path_text(arg_node)
    full = f"{prefix}::{path_str}" if prefix else path_str
    last = full.rsplit("::", 1)[-1] if "::" in full else full
    yield (full, last)


def _extract_imports_rust(file_id: int, source_rel_path: str, ts_root, db_id_for) -> list[ImportEntry]:
    """Walk `use_declaration` nodes; flatten grouped/nested forms into one
    ImportEntry per leaf path. Each entry's import_path includes the trailing
    item name; _resolve_rust strips it to find the containing module file.

    Limitations the future Deno resolver phase will fix:
      - `#[path = "..."]` module attributes are ignored; we assume the
        canonical filesystem layout (foo.rs / foo/mod.rs).
      - Workspace imports (`use other_crate::...`) are classified as external
        even when the other crate is intra-repo; cross-crate workspace
        resolution needs per-crate package indexes.
    """
    out: list[ImportEntry] = []
    for ts in _dfs(ts_root):
        if ts.type != "use_declaration":
            continue
        argument = ts.child_by_field_name("argument")
        if argument is None:
            continue
        for full_path, last_name in _flatten_rust_use(argument, ""):
            if not full_path:
                continue
            is_relative = (
                full_path.startswith("self::")
                or full_path.startswith("super::")
                or full_path in ("self", "super")
            )
            names = [last_name] if last_name else []
            out.append(
                ImportEntry(
                    file_id=file_id,
                    node_id=db_id_for[ts.id],
                    language="rust",
                    source_rel_path=source_rel_path,
                    import_path=full_path,
                    imported_names=names,
                    is_relative=is_relative,
                )
            )
    return out


def _collect_node_import_clause_names(import_stmt) -> list[str]:
    """Names bound by an `import_statement`. We capture the ORIGINAL exported
    name (matching against the target file's defs in Tier-A) rather than the
    local alias — same pattern as the Python extractor.

    - `import { foo, bar as baz } from "x"` → ["foo", "bar"]
    - `import defaultExport from "x"`      → ["defaultExport"]
    - `import * as ns from "x"`            → ["ns"]   (best-effort; member
       access via `ns.foo()` is linked by Tier-B fuzzy matching)
    - `import "side-effect"`               → []
    """
    names: list[str] = []
    clause = None
    for c in import_stmt.children:
        if c.type == "import_clause":
            clause = c
            break
    if clause is None:
        return names
    for c in clause.children:
        if c.type == "identifier":
            # Default import — the bound name in the importing scope.
            names.append(_text(c))
        elif c.type == "namespace_import":
            # `* as ns` — record the alias (won't link Tier-A but flags the import).
            for cc in c.children:
                if cc.type == "identifier":
                    names.append(_text(cc))
        elif c.type == "named_imports":
            for spec in c.children:
                if spec.type != "import_specifier":
                    continue
                name_node = spec.child_by_field_name("name")
                if name_node is not None:
                    names.append(_text(name_node))
    return names


def _collect_node_export_names(export_stmt) -> list[str]:
    """Names re-exported by `export { x, y } from "m"`. `export * from "m"`
    yields no specific names."""
    names: list[str] = []
    for c in export_stmt.children:
        if c.type != "export_clause":
            continue
        for spec in c.children:
            if spec.type != "export_specifier":
                continue
            name_node = spec.child_by_field_name("name")
            if name_node is not None:
                names.append(_text(name_node))
    return names


def _make_node_extractor(language: str):
    """Return an extractor closure tagged with the right language string.
    JS and TS share parsing logic; the per-call language tag is what
    `_resolve_one` keys off."""

    def extract(file_id: int, source_rel_path: str, ts_root, db_id_for) -> list[ImportEntry]:
        out: list[ImportEntry] = []
        for ts in _dfs(ts_root):
            if ts.type == "import_statement":
                src_node = ts.child_by_field_name("source")
                if src_node is None:
                    continue
                raw = _strip_quotes(_text(src_node))
                names = _collect_node_import_clause_names(ts)
                out.append(
                    ImportEntry(
                        file_id=file_id,
                        node_id=db_id_for[ts.id],
                        language=language,
                        source_rel_path=source_rel_path,
                        import_path=raw,
                        imported_names=names,
                        is_relative=is_relative_specifier(raw),
                    )
                )
            elif ts.type == "export_statement":
                # Only re-exports (those with a `source` field) are imports.
                src_node = ts.child_by_field_name("source")
                if src_node is None:
                    continue
                raw = _strip_quotes(_text(src_node))
                names = _collect_node_export_names(ts)
                out.append(
                    ImportEntry(
                        file_id=file_id,
                        node_id=db_id_for[ts.id],
                        language=language,
                        source_rel_path=source_rel_path,
                        import_path=raw,
                        imported_names=names,
                        is_relative=is_relative_specifier(raw),
                    )
                )
            elif ts.type == "call_expression":
                # `require("…")` — CommonJS import. Other call_expressions are
                # ignored. `imported_names` is left empty since CommonJS
                # exports are anonymous (Tier-B fuzzy matching links member
                # accesses against the resolved file).
                fexpr = ts.child_by_field_name("function")
                if fexpr is None or fexpr.type != "identifier" or _text(fexpr) != "require":
                    continue
                args = ts.child_by_field_name("arguments")
                if args is None:
                    continue
                str_node = next((c for c in args.children if c.type == "string"), None)
                if str_node is None:
                    continue
                raw = _strip_quotes(_text(str_node))
                out.append(
                    ImportEntry(
                        file_id=file_id,
                        node_id=db_id_for[ts.id],
                        language=language,
                        source_rel_path=source_rel_path,
                        import_path=raw,
                        imported_names=[],
                        is_relative=is_relative_specifier(raw),
                    )
                )
        return out

    return extract


_EXTRACTORS = {
    "python": _extract_imports_python,
    "solidity": _extract_imports_solidity,
    "go": _extract_imports_go,
    "javascript": _make_node_extractor("javascript"),
    "typescript": _make_node_extractor("typescript"),
    "rust": _extract_imports_rust,
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


def _rust_module_path_for(rel_path: str, crate_root_path: str) -> str | None:
    """Map a .rs file's repo-relative path to its crate-relative module path.

    Examples (crate_root_path='src/lib.rs'):
        'src/lib.rs'           → ''             (the crate root itself)
        'src/utils.rs'         → 'utils'
        'src/foo/bar.rs'       → 'foo::bar'
        'src/foo/mod.rs'       → 'foo'
        'src/foo/bar/mod.rs'   → 'foo::bar'

    Returns None for files that don't sit under the crate root's directory
    (e.g. test files in `tests/`, examples/, build scripts), which we leave
    unindexed in the heuristic tier.
    """
    if not rel_path.endswith(".rs"):
        return None
    if rel_path == crate_root_path:
        return ""
    crate_dir = crate_root_path.rsplit("/", 1)[0] if "/" in crate_root_path else ""
    if crate_dir:
        if not rel_path.startswith(crate_dir + "/"):
            return None
        rel = rel_path[len(crate_dir) + 1:]
    else:
        rel = rel_path
    no_ext = rel[:-3]
    parts = no_ext.split("/")
    if parts and parts[-1] == "mod":
        parts = parts[:-1]
    return "::".join(parts) if parts else ""


def _rust_crate_metadata(repo_root: Path) -> tuple[str | None, str]:
    """Return (crate_name, crate_root_path) discovered from Cargo.toml at the
    repo root. crate_root_path is repo-relative (e.g. 'src/lib.rs').

    Defaults: if Cargo.toml is missing or unparsable, falls back to the
    on-disk file `src/lib.rs` if present, else `src/main.rs`. crate_name
    falls back to the repo root directory name. We pick lib over bin so
    library re-exports are preferred when both are present (the common
    cargo new-with-binary layout).
    """
    crate_name: str | None = None
    crate_root_path: str | None = None
    cargo = repo_root / "Cargo.toml"
    if cargo.is_file():
        try:
            data = tomllib.loads(cargo.read_text())
        except Exception:
            data = {}
        pkg = data.get("package", {})
        if isinstance(pkg, dict):
            n = pkg.get("name")
            if isinstance(n, str):
                crate_name = n.replace("-", "_")  # cargo normalizes dashes
        lib = data.get("lib", {})
        if isinstance(lib, dict):
            p = lib.get("path")
            if isinstance(p, str):
                crate_root_path = p
        if crate_root_path is None:
            bins = data.get("bin", [])
            if isinstance(bins, list) and bins:
                first = bins[0]
                if isinstance(first, dict) and isinstance(first.get("path"), str):
                    crate_root_path = first["path"]
    if crate_root_path is None:
        if (repo_root / "src" / "lib.rs").is_file():
            crate_root_path = "src/lib.rs"
        elif (repo_root / "src" / "main.rs").is_file():
            crate_root_path = "src/main.rs"
        else:
            crate_root_path = "src/lib.rs"  # best-effort default
    if crate_name is None:
        crate_name = repo_root.name.replace("-", "_") or None
    return crate_name, crate_root_path


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
    repo_row = await conn.fetchrow(
        "SELECT root_path FROM repos WHERE id=$1", repo_id,
    )
    repo_root = repo_row["root_path"] if repo_row else None

    file_index: dict[str, int] = {}
    files_by_id: dict[int, str] = {}
    file_languages: dict[int, str] = {}
    pkg_index: dict[str, int] = {}
    pkg_index_go: dict[str, int] = {}
    pkg_files_go: dict[str, list[int]] = {}
    rust_files: list[tuple[int, str]] = []  # (file_id, rel_path); built first, indexed below once we know the crate root
    has_go = False
    has_node = False
    has_rust = False
    for f in files:
        fid = f["id"]
        file_index[f["path"]] = fid
        files_by_id[fid] = f["path"]
        file_languages[fid] = f["language"]
        if f["language"] == "python":
            dotted = _python_dotted_for(f["path"])
            if dotted:
                pkg_index[dotted] = fid
        elif f["language"] == "go":
            has_go = True
            # Package dir = parent directory. `pkg_index_go` maps to one
            # representative for the imports.resolved_file_id FK (which is
            # singular); `pkg_files_go` keeps the full list so Phase 3's
            # cross-file linker can fuzzy-match across the whole package.
            pkg_dir = "/".join(f["path"].split("/")[:-1])
            pkg_index_go.setdefault(pkg_dir, fid)
            pkg_files_go.setdefault(pkg_dir, []).append(fid)
        elif f["language"] in ("javascript", "typescript"):
            has_node = True
        elif f["language"] == "rust":
            has_rust = True
            rust_files.append((fid, f["path"]))

    # Read go.mod once if any Go file is present and a root_path is known.
    go_module_path: str | None = None
    if has_go and repo_root:
        gomod = Path(repo_root) / "go.mod"
        if gomod.is_file():
            for line in gomod.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("module "):
                    go_module_path = stripped.split(None, 1)[1].strip().strip('"')
                    break

    # Read tsconfig.json once if any JS/TS file is present. The same parsed
    # paths config applies to both javascript.yaml and typescript.yaml.
    node_tsconfig: TsconfigPaths | None = None
    if has_node and repo_root:
        node_tsconfig = load_tsconfig_paths(Path(repo_root))

    # Rust: parse Cargo.toml once per repo to discover the crate name and
    # crate root path; then derive each .rs file's crate-relative module path.
    # Workspace support (multiple Cargo.toml files) is deliberately deferred —
    # the heuristic tier handles the single-crate case and external-crate
    # classification, with workspace member resolution flagged in the
    # Architecture doc as a Phase 8 follow-up.
    rust_pkg_index: dict[str, int] = {}
    rust_module_for_file: dict[int, str] = {}
    rust_crate_name: str | None = None
    if has_rust and repo_root:
        rust_crate_name, crate_root_path = _rust_crate_metadata(Path(repo_root))
        for fid, rel in rust_files:
            mod_path = _rust_module_path_for(rel, crate_root_path)
            if mod_path is None:
                continue
            rust_pkg_index.setdefault(mod_path, fid)
            rust_module_for_file[fid] = mod_path

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
        package_index_go=pkg_index_go,
        go_pkg_files=pkg_files_go,
        go_module_path=go_module_path,
        files_by_id=files_by_id,
        file_languages=file_languages,
        repo_id=repo_id,
        node_tsconfig=node_tsconfig,
        package_index_rust=rust_pkg_index,
        rust_module_for_file=rust_module_for_file,
        rust_crate_name=rust_crate_name,
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


def _resolve_go(entry: ImportEntry, idx: RepoIndex) -> ResolvedImport:
    """Classify a Go import path.

    Three buckets, in priority order:
      - intra_repo: matches the module path declared in go.mod, suffix maps
        to a known package directory.
      - external (stdlib): first segment has no dot ("fmt", "net/http", …).
      - external (third-party): everything else, e.g. github.com/x/y. The
        package_name keeps the org/repo prefix so the same dependency rolls
        up across multiple subpackage imports.
    """
    raw = entry.import_path
    mod = idx.go_module_path
    if mod and (raw == mod or raw.startswith(mod + "/")):
        suffix = "" if raw == mod else raw[len(mod) + 1 :]
        target = idx.package_index_go.get(suffix)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_id=target)
        return ResolvedImport(entry, "unresolved")

    first = raw.split("/", 1)[0]
    if "." not in first:
        return ResolvedImport(entry, "external", package_name=raw)

    parts = raw.split("/")
    if first in {"github.com", "gitlab.com", "bitbucket.org"} and len(parts) >= 3:
        pkg = "/".join(parts[:3])
    else:
        pkg = parts[0]
    return ResolvedImport(entry, "external", package_name=pkg)


def _try_rust_module_path(entry: ImportEntry, idx: RepoIndex, parts: list[str]) -> ResolvedImport:
    """Try `parts` as a crate-relative module path; on miss, drop trailing
    segments and retry. Same shape as the Python resolver's parent-prefix
    fallback — handles the ambiguity between `use crate::utils::helper`
    (helper is an item in utils.rs) and `use crate::utils::helpers` (helpers
    might be a submodule file). Empty parts resolves to the crate root."""
    while parts:
        cand = "::".join(parts)
        target = idx.package_index_rust.get(cand)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_id=target)
        parts = parts[:-1]
    crate_root = idx.package_index_rust.get("")
    if crate_root is not None:
        return ResolvedImport(entry, "intra_repo", resolved_file_id=crate_root)
    return ResolvedImport(entry, "unresolved")


def _resolve_rust(entry: ImportEntry, idx: RepoIndex) -> ResolvedImport:
    """Classify a Rust use-path.

    Buckets, in priority order:
      • relative (`self::…` / `super::…`) — anchored at the importer's
        module path; ascend per leading `super`, then descend per the
        remaining tail.
      • `crate::…` — strip prefix, look up against package_index_rust.
      • Bare path whose head equals the current crate's name (`use myapp::…`)
        — same treatment as `crate::…`.
      • Anything else — external. Package = first path segment.

    Workspace member crates currently fall into the external bucket; once
    the Deno resolver lands they'll be reclassified by walking sibling
    Cargo.toml files. We don't try to guess that here — wrong intra-repo
    edges are worse than honest external classification.
    """
    raw = entry.import_path
    if not raw:
        return ResolvedImport(entry, "unresolved")
    parts = raw.split("::")

    if entry.is_relative:
        src_module = idx.rust_module_for_file.get(entry.file_id)
        if src_module is None:
            return ResolvedImport(entry, "unresolved")
        src_parts = src_module.split("::") if src_module else []
        ascend = 0
        i = 0
        while i < len(parts):
            if parts[i] == "self":
                i += 1
                continue
            if parts[i] == "super":
                ascend += 1
                i += 1
                continue
            break
        if ascend > len(src_parts):
            return ResolvedImport(entry, "unresolved")
        base = src_parts[: len(src_parts) - ascend]
        tail = parts[i:]
        return _try_rust_module_path(entry, idx, base + tail)

    head = parts[0]
    if head == "crate":
        return _try_rust_module_path(entry, idx, parts[1:])
    if idx.rust_crate_name and head == idx.rust_crate_name:
        return _try_rust_module_path(entry, idx, parts[1:])
    return ResolvedImport(entry, "external", package_name=head)


def _resolve_node_import(entry: ImportEntry, idx: RepoIndex) -> ResolvedImport:
    """Classify a JS/TS import.

    Order:
      1. tsconfig `paths` alias — `@app/*` style; intra_repo when the rewritten
         path lands on a known file.
      2. Relative path — Node-style extension probe (.ts → .tsx → .js → .jsx
         → .mjs → .cjs, then `index.<ext>`); intra_repo on hit.
      3. Bare specifier — external. The package name is rolled up so
         `react/jsx-runtime`, `react/server`, and `react` all share one
         external_dependencies row.

    Note: `node_modules/` contents are not indexed in v1, so vendored
    packages still classify as external (matches existing Solidity behavior).
    """
    raw = entry.import_path

    if idx.node_tsconfig is not None:
        target = _resolve_tsconfig_alias(raw, idx.node_tsconfig, idx.file_index)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_id=target)

    if entry.is_relative:
        target = _resolve_relative_node(entry.source_rel_path, raw, idx.file_index)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_id=target)
        return ResolvedImport(entry, "unresolved")

    pkg = package_name_for_specifier(raw)
    return ResolvedImport(entry, "external", package_name=pkg)


def _resolve_one(entry: ImportEntry, idx: RepoIndex, cfg: LanguageConfig) -> ResolvedImport:
    if entry.language == "python":
        return _resolve_python(entry, idx)
    if entry.language == "solidity":
        return _resolve_solidity(entry, idx, cfg)
    if entry.language == "go":
        return _resolve_go(entry, idx)
    if entry.language in ("javascript", "typescript"):
        return _resolve_node_import(entry, idx)
    if entry.language == "rust":
        return _resolve_rust(entry, idx)
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
) -> tuple[int, int, dict[int, set[int]]]:
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

    # Go: files in the same package share scope without an explicit import. Seed
    # every Go file with its package peers so cross-file ref/call/inheritance
    # linking can find symbols declared by a sibling file.
    for pkg_dir, peers in idx.go_pkg_files.items():
        peer_set = set(peers)
        for fid in peers:
            imported_files_by.setdefault(fid, set()).update(peer_set - {fid})

    # importer_file_id → name → target_def_id (Tier A direct hits)
    direct_by: dict[int, dict[str, int]] = {}
    for row in intra_imports:
        importer = row["file_id"]
        target_file = row["resolved_file_id"]
        # Go: an `import "x/y/z"` references a package, not a single file.
        # Expand to every sibling .go file so Tier-B fuzzy matching can find
        # symbols defined in any peer file of the imported package.
        target_files: set[int] = {target_file}
        if idx.file_languages.get(target_file) == "go":
            target_path = idx.files_by_id.get(target_file, "")
            pkg_dir = "/".join(target_path.split("/")[:-1])
            peers = idx.go_pkg_files.get(pkg_dir, [])
            target_files.update(peers)
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

    return refs_updated, calls_updated, imported_files_by


async def _link_cross_file_inheritance(
    conn: asyncpg.Connection,
    repo_id: int,
    idx: RepoIndex,
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
        SELECT ie.id, ie.child_def_id, ie.base_name, d.file_id AS child_file_id
        FROM inherits_edges ie
        JOIN definitions d ON d.id = ie.child_def_id
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1 AND ie.base_def_id IS NULL
        """,
        repo_id,
    )
    if not pending:
        return 0

    target_rows = await conn.fetch(
        """
        SELECT d.id, d.file_id, d.name
        FROM definitions d JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1 AND d.kind IN ('contract', 'interface', 'class', 'type')
        """,
        repo_id,
    )
    targets_by_name: dict[str, list[tuple[int, int]]] = {}
    for r in target_rows:
        targets_by_name.setdefault(r["name"], []).append((r["file_id"], r["id"]))

    updates: list[tuple[int, int, str]] = []  # (edge_id, base_def_id, confidence)
    for r in pending:
        name = r["base_name"]
        child_file = r["child_file_id"]
        imported = imported_files_by.get(child_file, set())
        cands = targets_by_name.get(name, [])
        in_imported = [(fid, did) for fid, did in cands if fid in imported]
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
    repo_id: int,
) -> int:
    """For each resolved (child_class, base_class) inheritance pair, find
    method/function/modifier defs inside the child whose `name` matches one in
    the base (or any transitive ancestor — closest match wins). Insert into
    overrides_edges. Pre-clears repo's overrides so re-runs are idempotent.
    """
    # Drop any pre-existing rows scoped to this repo.
    await conn.execute(
        """
        DELETE FROM overrides_edges
        WHERE child_def_id IN (
            SELECT d.id FROM definitions d
            JOIN files f ON f.id = d.file_id
            WHERE f.repo_id = $1
        )
        """,
        repo_id,
    )

    inh_resolved = await conn.fetch(
        """
        SELECT ie.child_def_id AS child_class, ie.base_def_id AS base_class
        FROM inherits_edges ie
        JOIN definitions d ON d.id = ie.child_def_id
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1 AND ie.base_def_id IS NOT NULL
        ORDER BY ie.child_def_id, ie.ord
        """,
        repo_id,
    )
    if not inh_resolved:
        return 0

    # child_class → ordered list of direct bases (preserve declaration order).
    direct_bases: dict[int, list[int]] = {}
    for r in inh_resolved:
        direct_bases.setdefault(r["child_class"], []).append(r["base_class"])

    # Per-class methods: scope_id → [(name, kind, def_id), ...].
    method_rows = await conn.fetch(
        """
        SELECT d.id, d.name, d.kind, d.scope_id
        FROM definitions d
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1
          AND d.kind IN ('function', 'method', 'modifier', 'constructor')
          AND d.scope_id IS NOT NULL
        """,
        repo_id,
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
        "INSERT INTO overrides_edges (child_def_id, base_def_id) VALUES ($1, $2) "
        "ON CONFLICT (child_def_id, base_def_id) DO NOTHING",
        pairs,
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
            parser = LANGUAGES[lang].parser(PurePosixPath(f["path"]).suffix.lower())
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
        refs_updated, calls_updated, imported_files_by = await _link_cross_file(conn, repo_id, idx)
        stats.cross_file_refs_resolved = refs_updated
        stats.cross_file_calls_resolved = calls_updated

        # Inheritance: cross-file linking + override generation. Both are
        # repo-scoped (semantic_resolver wrote the rows with intra-file
        # base_def_id where possible; we fill in the rest).
        stats.cross_file_inherits_resolved = await _link_cross_file_inheritance(
            conn, repo_id, idx, imported_files_by,
        )
        stats.overrides_inserted = await _generate_overrides(conn, repo_id)

    return stats


def resolve_repo_imports_sync(repo_id: int, dsn: str | None = None) -> ResolutionStats:
    from db.connection import pool_ctx

    async def _run() -> ResolutionStats:
        async with pool_ctx(dsn) as pool:
            return await resolve_repo_imports(pool, repo_id)

    return asyncio.run(_run())
