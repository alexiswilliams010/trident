"""Import resolution for Rust."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import ImportEntry, ResolvedImport

if TYPE_CHECKING:
    from ...heuristic_resolver import BranchIndex


def _try_rust_module_path(
    entry: ImportEntry,
    crate_index: dict[str, int],
    parts: list[str],
) -> ResolvedImport:
    """Try `parts` as a module path inside `crate_index` (one crate's module
    map); drop trailing segments on miss. Mirrors the Python resolver's
    parent-prefix fallback — handles the ambiguity between
    `use crate::utils::helper` (helper is an item in utils.rs) and
    `use crate::utils::helpers` (helpers might be a submodule file). Empty
    parts maps to the crate root entry, if present."""
    while parts:
        cand = "::".join(parts)
        target = crate_index.get(cand)
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_version_id=target)
        parts = parts[:-1]
    crate_root = crate_index.get("")
    if crate_root is not None:
        return ResolvedImport(entry, "intra_repo", resolved_file_version_id=crate_root)
    return ResolvedImport(entry, "unresolved")


def resolve_rust(entry: ImportEntry, idx: "BranchIndex") -> ResolvedImport:
    """Classify a Rust use-path.

    Buckets, in priority order:
      • relative (`self::…` / `super::…`) — anchored at the importer's
        module path within its own crate; ascend per leading `super`, then
        descend per the remaining tail.
      • `crate::…` — strip prefix, look up in the source file's owning
        crate's index.
      • Absolute path whose head matches a known workspace member crate
        (cargo-normalised, dashes → underscores) — strip the head, look up
        in that crate's index. Covers both same-crate references that name
        the crate explicitly (`use my_crate::utils`) and cross-crate
        references inside a workspace (`use other_member::foo`).
      • Anything else — external. Package = first path segment.

    Edge case: if the source file has no owning crate (e.g. it lives outside
    every discovered crate's src tree), `crate::` and relative paths can't
    be resolved and we return unresolved rather than guessing.
    """
    raw = entry.import_path
    if not raw:
        return ResolvedImport(entry, "unresolved")
    parts = raw.split("::")

    rust_state = idx.lang_state.get("rust")
    if rust_state is None:
        return ResolvedImport(entry, "external", package_name=parts[0])
    src_owner = rust_state.module_for_file.get(entry.file_version_id)

    if entry.is_relative:
        if src_owner is None:
            return ResolvedImport(entry, "unresolved")
        src_crate, src_module = src_owner
        crate_index = rust_state.package_index.get(src_crate, {})
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
        return _try_rust_module_path(entry, crate_index, base + tail)

    head = parts[0]
    if head == "crate":
        if src_owner is None:
            return ResolvedImport(entry, "unresolved")
        crate_index = rust_state.package_index.get(src_owner[0], {})
        return _try_rust_module_path(entry, crate_index, parts[1:])

    # Cargo normalises `-` to `_` in crate names *as referenced from code*,
    # so `use anchor_lang::…` matches a Cargo.toml declaring `anchor-lang`.
    head_norm = head.replace("-", "_")
    if head_norm in rust_state.crate_names:
        crate_index = rust_state.package_index.get(head_norm, {})
        return _try_rust_module_path(entry, crate_index, parts[1:])

    # Fallback: bare path whose head names a top-level module of the source
    # file's own crate — `use common::foo;` from a lib.rs that declared
    # `mod common;`. Strict edition-2018 style would write `use crate::…`
    # but plenty of real code (anchor programs, libs vendored from Rust
    # 2015) uses the bare form. We require an exact module match (not just
    # any path prefix) to keep false positives low.
    if src_owner is not None:
        src_crate_index = rust_state.package_index.get(src_owner[0], {})
        if head in src_crate_index or any(k.startswith(head + "::") for k in src_crate_index):
            return _try_rust_module_path(entry, src_crate_index, parts)

    return ResolvedImport(entry, "external", package_name=head)
