"""Import extraction for JavaScript and TypeScript.

JS and TS share the same import / export / require syntax; only the
language tag stamped onto each `ImportEntry` differs. Both handlers
compose `extract_node_imports` with their language name.
"""

from __future__ import annotations

from ..base import ImportEntry
from .node_resolution import is_relative_specifier
from .ts_helpers import dfs, strip_quotes, text


def _collect_import_clause_names(import_stmt) -> list[str]:
    """Names bound by an `import_statement`. Captures the ORIGINAL exported
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
            names.append(text(c))
        elif c.type == "namespace_import":
            for cc in c.children:
                if cc.type == "identifier":
                    names.append(text(cc))
        elif c.type == "named_imports":
            for spec in c.children:
                if spec.type != "import_specifier":
                    continue
                name_node = spec.child_by_field_name("name")
                if name_node is not None:
                    names.append(text(name_node))
    return names


def _collect_export_names(export_stmt) -> list[str]:
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
                names.append(text(name_node))
    return names


def extract_node_imports(
    language: str,
    file_version_id: int,
    source_rel_path: str,
    ts_root,
    db_id_for: dict[int, int],
) -> list[ImportEntry]:
    """Walk a JS/TS CST and emit one ImportEntry per import/export/require."""
    out: list[ImportEntry] = []
    for ts in dfs(ts_root):
        if ts.type == "import_statement":
            src_node = ts.child_by_field_name("source")
            if src_node is None:
                continue
            raw = strip_quotes(text(src_node))
            names = _collect_import_clause_names(ts)
            out.append(
                ImportEntry(
                    file_version_id=file_version_id,
                    node_id=db_id_for[ts.id],
                    language=language,
                    source_rel_path=source_rel_path,
                    import_path=raw,
                    imported_names=names,
                    is_relative=is_relative_specifier(raw),
                )
            )
        elif ts.type == "export_statement":
            src_node = ts.child_by_field_name("source")
            if src_node is None:
                continue
            raw = strip_quotes(text(src_node))
            names = _collect_export_names(ts)
            out.append(
                ImportEntry(
                    file_version_id=file_version_id,
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
            # ignored. `imported_names` stays empty since CommonJS exports
            # are anonymous (Tier-B fuzzy matching links member accesses).
            fexpr = ts.child_by_field_name("function")
            if fexpr is None or fexpr.type != "identifier" or text(fexpr) != "require":
                continue
            args = ts.child_by_field_name("arguments")
            if args is None:
                continue
            str_node = next((c for c in args.children if c.type == "string"), None)
            if str_node is None:
                continue
            raw = strip_quotes(text(str_node))
            out.append(
                ImportEntry(
                    file_version_id=file_version_id,
                    node_id=db_id_for[ts.id],
                    language=language,
                    source_rel_path=source_rel_path,
                    import_path=raw,
                    imported_names=[],
                    is_relative=is_relative_specifier(raw),
                )
            )
    return out
