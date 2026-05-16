"""Language handler registry.

Adding a language = one entry in `core.grammar_meta.LANGUAGES`, one YAML config
in `configs/`, and one `LanguageHandler` subclass registered here. The startup
assertion below catches drift between the grammar registry and the handler
registry.
"""

from __future__ import annotations

from ..grammar_meta import LANGUAGES
from .base import (
    ImportEntry,
    InheritanceEdge,
    LanguageHandler,
    ResolvedImport,
    SemanticContext,
)
from .go import GoHandler
from .javascript import JavaScriptHandler
from .python import PythonHandler
from .rust import RustHandler
from .solidity import SolidityHandler
from .typescript import TypeScriptHandler


HANDLERS: dict[str, LanguageHandler] = {
    h.name: h
    for h in (
        PythonHandler(),
        SolidityHandler(),
        GoHandler(),
        JavaScriptHandler(),
        TypeScriptHandler(),
        RustHandler(),
    )
}


def get_handler(name: str) -> LanguageHandler:
    return HANDLERS[name]


_handler_keys = set(HANDLERS)
_grammar_keys = set(LANGUAGES)
if _handler_keys != _grammar_keys:
    raise RuntimeError(
        "language handler / grammar_meta drift: "
        f"missing handlers={sorted(_grammar_keys - _handler_keys)}, "
        f"orphan handlers={sorted(_handler_keys - _grammar_keys)}"
    )


__all__ = [
    "HANDLERS",
    "ImportEntry",
    "InheritanceEdge",
    "LanguageHandler",
    "ResolvedImport",
    "SemanticContext",
    "get_handler",
]
