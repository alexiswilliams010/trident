"""Import extraction for Python."""

from __future__ import annotations

from .._shared.ts_helpers import dfs, text
from ..base import ImportEntry


def extract_python_imports(
    file_version_id: int,
    source_rel_path: str,
    ts_root,
    db_id_for: dict[int, int],
) -> list[ImportEntry]:
    out: list[ImportEntry] = []
    for ts in dfs(ts_root):
        if ts.type == "import_statement":
            # `import a, b.c` — each `name` field is a dotted_name we treat as one import.
            for i in range(ts.child_count):
                if ts.field_name_for_child(i) != "name":
                    continue
                dotted = text(ts.children[i])
                out.append(
                    ImportEntry(
                        file_version_id=file_version_id,
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
                        dot_count += text(c).count(".")
                    elif c.type == "dotted_name":
                        tail = text(c)
                import_path = "." * dot_count + tail
            else:
                import_path = text(mod_node)

            names: list[str] = []
            for i in range(ts.child_count):
                if ts.field_name_for_child(i) == "name":
                    names.append(text(ts.children[i]))
            out.append(
                ImportEntry(
                    file_version_id=file_version_id,
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
