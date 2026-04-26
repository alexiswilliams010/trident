"""Tree-sitter language registry + node-types.json metadata.

Each supported language is registered here with:
- An extension list used by the file walker.
- A loader that returns a `tree_sitter.Language` object.

The node-types.json metadata (shipped by every grammar) is parsed lazily on
first request. Tier 2 uses it to validate YAML configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from typing import Callable

from tree_sitter import Language, Parser


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    module_name: str  # e.g. "tree_sitter_python"
    language_attr: str = "language"  # function in the module returning the language ptr

    def language(self) -> Language:
        return _load_language(self)

    def parser(self) -> Parser:
        return Parser(self.language())


# Module-level registry. Adding a language = one entry here + a YAML config.
LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        extensions=(".py",),
        module_name="tree_sitter_python",
    ),
    "solidity": LanguageSpec(
        name="solidity",
        extensions=(".sol",),
        module_name="tree_sitter_solidity",
    ),
    "go": LanguageSpec(
        name="go",
        extensions=(".go",),
        module_name="tree_sitter_go",
    ),
}


_EXT_TO_LANG: dict[str, str] = {
    ext: spec.name for spec in LANGUAGES.values() for ext in spec.extensions
}


def language_for_path(path: str | Path) -> str | None:
    """Return the language name for a file path based on extension, or None."""
    suffix = Path(path).suffix.lower()
    return _EXT_TO_LANG.get(suffix)


@cache
def _load_language(spec: LanguageSpec) -> Language:
    mod = import_module(spec.module_name)
    fn: Callable[[], object] = getattr(mod, spec.language_attr)
    return Language(fn())


@dataclass
class NodeTypeInfo:
    """Slice of node-types.json relevant to Tier 2 validation."""

    type: str
    named: bool
    fields: dict[str, dict] = field(default_factory=dict)
    children: dict | None = None


@cache
def node_types(language_name: str) -> dict[str, NodeTypeInfo]:
    """Parse node-types.json from the language's package, keyed by node type."""
    spec = LANGUAGES[language_name]
    pkg = files(spec.module_name)
    candidates = [
        pkg / "node-types.json",
        pkg / "src" / "node-types.json",
    ]
    raw = None
    for p in candidates:
        try:
            if p.is_file():
                raw = p.read_text()
                break
        except (FileNotFoundError, NotADirectoryError):
            continue
    if raw is None:
        return {}

    data = json.loads(raw)
    result: dict[str, NodeTypeInfo] = {}
    for entry in data:
        result[entry["type"]] = NodeTypeInfo(
            type=entry["type"],
            named=entry.get("named", False),
            fields=entry.get("fields", {}),
            children=entry.get("children"),
        )
    return result


def is_named(language_name: str, node_type: str) -> bool:
    info = node_types(language_name).get(node_type)
    return bool(info and info.named)
