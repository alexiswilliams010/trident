"""Per-branch index state for Python: dotted module name → file_version_id."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PythonIndexState:
    """`mypackage.utils` → file_version_id."""

    pkg_index: dict[str, int] = field(default_factory=dict)


def python_dotted_for(rel_path: str) -> str | None:
    """`mypackage/utils.py` → `mypackage.utils`. `mypackage/__init__.py` → `mypackage`."""
    if not rel_path.endswith(".py"):
        return None
    no_ext = rel_path[:-3]
    parts = no_ext.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)
