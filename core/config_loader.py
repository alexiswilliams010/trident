"""Load + validate per-language YAML configs against configs/_schema.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import jsonschema
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
SCHEMA_PATH = CONFIGS_DIR / "_schema.json"


@dataclass(frozen=True)
class DefinitionRule:
    node_type: str
    kind: str
    name_field: str | None = None
    scope_boundary: bool = False
    visibility_field: str | None = None
    require_enclosing_scope_kind: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceRule:
    node_type: str
    name_field: str | None = None
    exclude_parent_field: tuple[str, ...] = ()
    confidence: str = "certain"


@dataclass(frozen=True)
class CallRule:
    node_type: str
    function_field: str
    name_field: str | None = None


@dataclass(frozen=True)
class DataAccessConfig:
    target_kinds: tuple[str, ...] = ()
    write_when_parent_field: tuple[str, ...] = ()


@dataclass(frozen=True)
class LanguageConfig:
    language: str
    module_node_type: str
    dependency_paths: tuple[str, ...]
    definitions: tuple[DefinitionRule, ...]
    references: tuple[ReferenceRule, ...]
    calls: tuple[CallRule, ...]
    data_access: DataAccessConfig | None = None
    raw: dict = field(default_factory=dict)  # full parsed YAML (Phase 3 reads `imports`)

    def definition_rule_for(self, node_type: str) -> DefinitionRule | None:
        for r in self.definitions:
            if r.node_type == node_type:
                return r
        return None

    def scope_boundary_node_types(self) -> set[str]:
        return {r.node_type for r in self.definitions if r.scope_boundary}


@cache
def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def load_language_config(language: str, configs_dir: Path | None = None) -> LanguageConfig:
    """Load `<language>.yaml`, validate against `_schema.json`, return typed config."""
    base = configs_dir or CONFIGS_DIR
    path = base / f"{language}.yaml"
    raw = yaml.safe_load(path.read_text())
    jsonschema.validate(raw, _schema())

    defs = tuple(
        DefinitionRule(
            node_type=d["node_type"],
            kind=d["kind"],
            name_field=d.get("name_field"),
            scope_boundary=d.get("scope_boundary", False),
            visibility_field=d.get("visibility_field"),
            require_enclosing_scope_kind=tuple(d.get("require_enclosing_scope_kind", [])),
        )
        for d in raw.get("definitions", [])
    )
    refs = tuple(
        ReferenceRule(
            node_type=r["node_type"],
            name_field=r.get("name_field"),
            exclude_parent_field=tuple(r.get("exclude_parent_field", [])),
            confidence=r.get("confidence", "certain"),
        )
        for r in raw.get("references", [])
    )
    calls = tuple(
        CallRule(
            node_type=c["node_type"],
            function_field=c["function_field"],
            name_field=c.get("name_field"),
        )
        for c in raw.get("calls", [])
    )
    da_raw = raw.get("data_access")
    da: DataAccessConfig | None = None
    if da_raw:
        da = DataAccessConfig(
            target_kinds=tuple(da_raw.get("target_kinds", [])),
            write_when_parent_field=tuple(da_raw.get("write_when_parent_field", [])),
        )

    return LanguageConfig(
        language=raw["language"],
        module_node_type=raw["module_node_type"],
        dependency_paths=tuple(raw.get("dependency_paths", [])),
        definitions=defs,
        references=refs,
        calls=calls,
        data_access=da,
        raw=raw,
    )
