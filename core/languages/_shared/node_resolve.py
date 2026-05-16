"""Import resolution for JavaScript and TypeScript.

Both handlers share the same Node-style resolution semantics — they delegate
into `resolve_node` and the only difference is the language tag stamped on
the ImportEntry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import ImportEntry, ResolvedImport
from .node_resolution import (
    package_name_for_specifier,
    resolve_relative,
    resolve_tsconfig_alias,
)

if TYPE_CHECKING:
    from ...heuristic_resolver import BranchIndex


def resolve_node(entry: ImportEntry, idx: "BranchIndex") -> ResolvedImport:
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

    node_state = idx.lang_state.get(entry.language)
    tsconfig = node_state.tsconfig if node_state is not None else None
    if tsconfig is not None:
        target = resolve_tsconfig_alias(raw, tsconfig, idx.file_index)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_version_id=target)

    if entry.is_relative:
        target = resolve_relative(entry.source_rel_path, raw, idx.file_index)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_version_id=target)
        return ResolvedImport(entry, "unresolved")

    pkg = package_name_for_specifier(raw)
    return ResolvedImport(entry, "external", package_name=pkg)
