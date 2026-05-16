"""Per-branch index state for Go."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoIndexState:
    """`pkg_index` maps a repo-relative package dir to one representative
    file_version_id (singular FK target for the imports table).
    `pkg_files` keeps the full list so the cross-file linker can fuzzy-match
    against every file in the package.
    `module_path` is the `module …` line from go.mod, if present.
    """

    pkg_index: dict[str, int] = field(default_factory=dict)
    pkg_files: dict[str, list[int]] = field(default_factory=dict)
    module_path: str | None = None


def index_go_file(fvid: int, rel_path: str, state: GoIndexState) -> None:
    pkg_dir = "/".join(rel_path.split("/")[:-1])
    state.pkg_index.setdefault(pkg_dir, fvid)
    state.pkg_files.setdefault(pkg_dir, []).append(fvid)


def finalize_go_state(repo_root: Path | None, state: GoIndexState) -> None:
    if repo_root is None:
        return
    gomod = repo_root / "go.mod"
    if not gomod.is_file():
        return
    for line in gomod.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("module "):
            state.module_path = stripped.split(None, 1)[1].strip().strip('"')
            return
