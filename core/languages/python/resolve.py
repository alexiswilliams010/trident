"""Import resolution for Python."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..base import ImportEntry, ResolvedImport

if TYPE_CHECKING:
    from ...heuristic_resolver import BranchIndex


def resolve_python(entry: ImportEntry, idx: "BranchIndex") -> ResolvedImport:
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
                return ResolvedImport(entry, "intra_repo", resolved_file_version_id=idx.file_index[cand])
            # Try as package __init__.py
            cand = "/".join(base_parts + tail_parts) + "/__init__.py"
            if cand in idx.file_index:
                return ResolvedImport(entry, "intra_repo", resolved_file_version_id=idx.file_index[cand])
            # Try: each imported name is itself a sibling module (`from . import siblings`)
            # falls through if tail was given but didn't resolve — leave unresolved.
        else:
            # `from . import name` — each imported name is a sibling module.
            # Resolve the FIRST name to populate resolved_file_version_id (full multi-name handled below).
            for name in entry.imported_names:
                cand = "/".join(base_parts + [name]) + ".py"
                if cand in idx.file_index:
                    return ResolvedImport(entry, "intra_repo", resolved_file_version_id=idx.file_index[cand])
                cand = "/".join(base_parts + [name]) + "/__init__.py"
                if cand in idx.file_index:
                    return ResolvedImport(entry, "intra_repo", resolved_file_version_id=idx.file_index[cand])
        return ResolvedImport(entry, "unresolved")

    # Absolute import: try the full dotted path + parent dotted prefixes.
    py_state = idx.lang_state.get("python")
    pkg_index = py_state.pkg_index if py_state is not None else {}
    parts = entry.import_path.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in pkg_index:
            return ResolvedImport(entry, "intra_repo", resolved_file_version_id=pkg_index[cand])
        parts.pop()

    # Top-level segment isn't local → external (e.g., `import requests`).
    top = entry.import_path.split(".")[0]
    return ResolvedImport(entry, "external", package_name=top)
