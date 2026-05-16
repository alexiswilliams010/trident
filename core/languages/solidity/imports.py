"""Import extraction for Solidity."""

from __future__ import annotations

from .._shared.ts_helpers import dfs, strip_quotes, text
from ..base import ImportEntry


def extract_solidity_imports(
    file_version_id: int,
    source_rel_path: str,
    ts_root,
    db_id_for: dict[int, int],
) -> list[ImportEntry]:
    out: list[ImportEntry] = []
    for ts in dfs(ts_root):
        if ts.type != "import_directive":
            continue
        src_node = ts.child_by_field_name("source")
        if src_node is None:
            continue
        raw_path = strip_quotes(text(src_node))
        names: list[str] = []
        # `import {Foo, Bar} from "..."` — `import_name` field on each named item.
        for i in range(ts.child_count):
            if ts.field_name_for_child(i) == "import_name":
                names.append(text(ts.children[i]))
        is_relative = raw_path.startswith("./") or raw_path.startswith("../")
        out.append(
            ImportEntry(
                file_version_id=file_version_id,
                node_id=db_id_for[ts.id],
                language="solidity",
                source_rel_path=source_rel_path,
                import_path=raw_path,
                imported_names=names,
                is_relative=is_relative,
            )
        )
    return out
