"""Import extraction for Rust.

Walks `use_declaration` nodes and flattens grouped/nested forms into one
ImportEntry per leaf path. Each entry's import_path includes the trailing
item name; the resolver strips it to find the containing module file.

Test-gated imports (inside `#[cfg(test)] mod tests { … }` / `#[test]`)
are skipped so the imports table mirrors the semantic-layer skip and
test-only crate dependencies don't surface in retrieval. Files in
`tests/` / `benches/` / `examples/` directories are skipped at the
resolver-loop level (see resolve_branch_imports).

Other limitations the future Deno resolver phase will fix:
  - `#[path = "..."]` module attributes are ignored; we assume the
    canonical filesystem layout (foo.rs / foo/mod.rs).
"""

from __future__ import annotations

from .._shared.ts_helpers import dfs
from ..base import ImportEntry
from .ast_helpers import flatten_rust_use, rust_inside_test_subtree


def extract_rust_imports(
    file_version_id: int,
    source_rel_path: str,
    ts_root,
    db_id_for: dict[int, int],
) -> list[ImportEntry]:
    # Whole-file skip mirroring semantic_resolver's _is_rust_test_path:
    # `tests/`, `benches/`, `examples/` are cargo's separate-compilation
    # dirs; `test_utils/` is the cross-crate fixture convention. Imports
    # from any of these would never contribute production-relevant edges.
    parts = source_rel_path.split("/")
    if any(p in ("tests", "benches", "examples", "test_utils") for p in parts):
        return []
    out: list[ImportEntry] = []
    for ts in dfs(ts_root):
        if ts.type != "use_declaration":
            continue
        if rust_inside_test_subtree(ts):
            continue
        argument = ts.child_by_field_name("argument")
        if argument is None:
            continue
        for full_path, last_name in flatten_rust_use(argument, ""):
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
                    file_version_id=file_version_id,
                    node_id=db_id_for[ts.id],
                    language="rust",
                    source_rel_path=source_rel_path,
                    import_path=full_path,
                    imported_names=names,
                    is_relative=is_relative,
                )
            )
    return out
