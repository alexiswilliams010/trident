"""Per-branch index state shared by JavaScript and TypeScript handlers.

Both handlers carry their own `NodeIndexState` instance under their own key
in `BranchIndex.lang_state`, but the state shape is identical — repos
typically have at most one tsconfig.json that applies to both JS and TS
files in the same tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .node_resolution import TsconfigPaths, load_tsconfig_paths


@dataclass
class NodeIndexState:
    tsconfig: TsconfigPaths | None = None


def finalize_node_state(repo_root: Path | None, state: NodeIndexState) -> None:
    if repo_root is None:
        return
    state.tsconfig = load_tsconfig_paths(repo_root)
