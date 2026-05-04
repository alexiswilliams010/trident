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
    # Optional per-extension override. Used for TypeScript, where the same
    # `tree_sitter_typescript` package ships two grammars: language_typescript()
    # for `.ts` and language_tsx() for `.tsx`. Keys must include a leading dot.
    extension_to_language_attr: tuple[tuple[str, str], ...] = ()

    def language(self, extension: str | None = None) -> Language:
        attr = self._attr_for_extension(extension)
        return _load_language_attr(self.module_name, attr)

    def parser(self, extension: str | None = None) -> Parser:
        return Parser(self.language(extension))

    def _attr_for_extension(self, extension: str | None) -> str:
        if extension is None or not self.extension_to_language_attr:
            return self.language_attr
        ext = extension.lower()
        for k, v in self.extension_to_language_attr:
            if k == ext:
                return v
        return self.language_attr


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
    "javascript": LanguageSpec(
        name="javascript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        module_name="tree_sitter_javascript",
    ),
    "typescript": LanguageSpec(
        name="typescript",
        extensions=(".ts", ".tsx"),
        module_name="tree_sitter_typescript",
        language_attr="language_typescript",
        extension_to_language_attr=(
            (".ts", "language_typescript"),
            (".tsx", "language_tsx"),
        ),
    ),
    "rust": LanguageSpec(
        name="rust",
        extensions=(".rs",),
        module_name="tree_sitter_rust",
    ),
}


_EXT_TO_LANG: dict[str, str] = {
    ext: spec.name for spec in LANGUAGES.values() for ext in spec.extensions
}


def language_for_path(path: str | Path) -> str | None:
    """Return the language name for a file path based on extension, or None."""
    suffix = Path(path).suffix.lower()
    return _EXT_TO_LANG.get(suffix)


def parser_for_path(path: str | Path) -> Parser | None:
    """Return a tree-sitter Parser configured with the right grammar for `path`,
    honoring per-extension grammar selection (e.g. `.tsx` vs `.ts`).
    Returns None if the path's language is unsupported."""
    suffix = Path(path).suffix.lower()
    lang = _EXT_TO_LANG.get(suffix)
    if lang is None:
        return None
    return LANGUAGES[lang].parser(suffix)


@cache
def _load_language_attr(module_name: str, attr: str) -> Language:
    mod = import_module(module_name)
    fn: Callable[[], object] = getattr(mod, attr)
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
