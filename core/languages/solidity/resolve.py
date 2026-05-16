"""Import resolution for Solidity."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from .._shared.node_resolution import normalize_relative_posix
from ..base import ImportEntry, ResolvedImport

if TYPE_CHECKING:
    from ...config_loader import LanguageConfig
    from ...heuristic_resolver import BranchIndex


def resolve_solidity(
    entry: ImportEntry, idx: "BranchIndex", cfg: "LanguageConfig",
) -> ResolvedImport:
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
        target = normalize_relative_posix((src_dir / raw).as_posix())
        if target in idx.file_index:
            return ResolvedImport(entry, "intra_repo", resolved_file_version_id=idx.file_index[target])
        return ResolvedImport(entry, "unresolved")

    # Bare repo-relative path that names a real file (rare but valid).
    if raw in idx.file_index:
        return ResolvedImport(entry, "intra_repo", resolved_file_version_id=idx.file_index[raw])

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
