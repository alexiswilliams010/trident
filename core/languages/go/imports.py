"""Import extraction for Go.

Both `import "fmt"` and grouped `import ( "fmt"; alias "x/y" )` produce
import_spec nodes; the latter wraps them in an import_spec_list. The DFS
walks both shapes uniformly. Each spec carries one path; we attach the row
to the spec's node_id (not the enclosing import_declaration) so each
imported path has its own row.
"""

from __future__ import annotations

from .._shared.ts_helpers import dfs, strip_quotes, text
from ..base import ImportEntry


def extract_go_imports(
    file_version_id: int,
    source_rel_path: str,
    ts_root,
    db_id_for: dict[int, int],
) -> list[ImportEntry]:
    out: list[ImportEntry] = []
    for ts in dfs(ts_root):
        if ts.type != "import_spec":
            continue
        path_node = ts.child_by_field_name("path")
        if path_node is None:
            continue
        raw = strip_quotes(text(path_node))
        # Optional `alias "x/y"` form. Skip blank-import (`_`) and dot-import (`.`)
        # — they don't bind a name we can resolve references against.
        alias_node = ts.child_by_field_name("name")
        if alias_node is not None and alias_node.type == "package_identifier":
            imported = text(alias_node)
        else:
            imported = raw.rsplit("/", 1)[-1]
        out.append(
            ImportEntry(
                file_version_id=file_version_id,
                node_id=db_id_for[ts.id],
                language="go",
                source_rel_path=source_rel_path,
                import_path=raw,
                imported_names=[imported],
                is_relative=False,
            )
        )
    return out
