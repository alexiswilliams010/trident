"""Rust crate discovery + per-file module-path mapping."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


# Directories never traversed when searching for Cargo.toml files. Build
# artefacts (`target/`) carry per-dependency Cargo.tomls that would otherwise
# pollute the crate set. The rest are common dependency / VCS dirs.
_RUST_CARGO_SCAN_PRUNE = frozenset({"target", "node_modules", ".git", "vendor"})


@dataclass(frozen=True)
class RustCrate:
    """One Cargo package discovered under the repo root.

    `name` follows Cargo's normalisation (dashes → underscores) so it can be
    matched against `use <name>::…` import heads literally. `root_path` and
    `package_dir` are repo-relative POSIX paths; `package_dir` is "" when the
    crate's Cargo.toml sits at the repo root.
    """

    name: str
    root_path: str          # repo-relative path of lib.rs / main.rs
    package_dir: str        # repo-relative dir holding Cargo.toml ("" if at repo root)


@dataclass
class RustIndexState:
    """Per-branch Rust index. Workspaces produce one entry per member.

    `package_index` outer key is the crate name (Cargo-normalised);
    inner key is the crate-relative module path. `module_for_file` records
    per-file ownership so `super::` / `self::` / `crate::` can navigate
    within the source's own crate. `crate_names` is the set of all member
    crates discovered in the workspace.
    """

    package_index: dict[str, dict[str, int]] = field(default_factory=dict)
    module_for_file: dict[int, tuple[str, str]] = field(default_factory=dict)
    crate_names: frozenset[str] = field(default_factory=frozenset)
    # Internal: files we've seen but haven't yet resolved to crates. Populated
    # by index_file; consumed by finalize_index once crate discovery is done.
    pending_files: list[tuple[int, str]] = field(default_factory=list)


def rust_module_path_for(rel_path: str, crate_root_path: str) -> str | None:
    """Map a .rs file's repo-relative path to its crate-relative module path.

    Examples (crate_root_path='src/lib.rs'):
        'src/lib.rs'           → ''             (the crate root itself)
        'src/utils.rs'         → 'utils'
        'src/foo/bar.rs'       → 'foo::bar'
        'src/foo/mod.rs'       → 'foo'
        'src/foo/bar/mod.rs'   → 'foo::bar'

    Returns None for files that don't sit under the crate root's directory
    (e.g. test files in `tests/`, examples/, build scripts).
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


def discover_rust_crates(repo_root: Path) -> list[RustCrate]:
    """Walk the repo for every Cargo.toml with a `[package]` section.

    Members of a workspace virtual root are picked up automatically because
    they each carry their own Cargo.toml. Workspace virtual roots (no
    `[package]`) are skipped — they only declare members.

    Per-crate root file is determined in priority order:
      1. `[lib].path` — explicit override.
      2. `[[bin]]` first entry's `path` — for bin-only crates.
      3. `<package_dir>/src/lib.rs` if it exists on disk.
      4. `<package_dir>/src/main.rs` if it exists.
    Every Cargo path is interpreted relative to the package's own directory.
    """
    crates: list[RustCrate] = []
    for cargo in repo_root.rglob("Cargo.toml"):
        rel = cargo.relative_to(repo_root)
        if any(part in _RUST_CARGO_SCAN_PRUNE for part in rel.parts):
            continue
        try:
            data = tomllib.loads(cargo.read_text())
        except Exception:
            continue
        pkg = data.get("package")
        if not isinstance(pkg, dict):
            continue
        name = pkg.get("name")
        if not isinstance(name, str) or not name:
            continue
        crate_name = name.replace("-", "_")

        package_dir_path = cargo.parent
        package_dir = package_dir_path.relative_to(repo_root).as_posix()
        if package_dir == ".":
            package_dir = ""

        crate_root_abs: Path | None = None
        lib = data.get("lib")
        if isinstance(lib, dict) and isinstance(lib.get("path"), str):
            crate_root_abs = (package_dir_path / lib["path"]).resolve()
        if crate_root_abs is None:
            bins = data.get("bin", [])
            if isinstance(bins, list) and bins:
                first = bins[0]
                if isinstance(first, dict) and isinstance(first.get("path"), str):
                    crate_root_abs = (package_dir_path / first["path"]).resolve()
        if crate_root_abs is None:
            if (package_dir_path / "src" / "lib.rs").is_file():
                crate_root_abs = package_dir_path / "src" / "lib.rs"
            elif (package_dir_path / "src" / "main.rs").is_file():
                crate_root_abs = package_dir_path / "src" / "main.rs"
            else:
                # Best-effort default — no source root found on disk. Module
                # path mapping will return None for every file in this crate.
                crate_root_abs = package_dir_path / "src" / "lib.rs"
        try:
            crate_root_rel = crate_root_abs.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            continue
        crates.append(RustCrate(
            name=crate_name,
            root_path=crate_root_rel,
            package_dir=package_dir,
        ))
    return crates


def crate_for_file(rel_path: str, crates: list[RustCrate]) -> RustCrate | None:
    """Pick the crate whose `package_dir` is the longest path prefix of
    `rel_path`. Falls back to a repo-root crate (package_dir="") if present
    and nothing else matched."""
    best: RustCrate | None = None
    best_len = -1
    for c in crates:
        if c.package_dir == "":
            if best_len < 0:
                best = c
                best_len = 0
            continue
        prefix = c.package_dir + "/"
        if rel_path.startswith(prefix) and len(c.package_dir) > best_len:
            best = c
            best_len = len(c.package_dir)
    return best


def finalize_rust_state(repo_root: Path | None, state: RustIndexState) -> None:
    """Discover crates and resolve every pending file's module path."""
    if repo_root is None:
        return
    crates = discover_rust_crates(repo_root)
    state.crate_names = frozenset(c.name for c in crates)
    # Pre-create empty per-crate maps so the resolver's lookups don't have to
    # special-case missing keys for known crates.
    for c in crates:
        state.package_index.setdefault(c.name, {})
    for fvid, rel in state.pending_files:
        owner = crate_for_file(rel, crates)
        if owner is None:
            continue
        mod_path = rust_module_path_for(rel, owner.root_path)
        if mod_path is None:
            continue
        state.package_index[owner.name].setdefault(mod_path, fvid)
        state.module_for_file[fvid] = (owner.name, mod_path)
    state.pending_files.clear()
