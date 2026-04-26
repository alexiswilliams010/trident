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
    # Multi-name shapes: e.g. Go's `var a, b int` — one var_spec yields two defs.
    name_field_multiple: bool = False
    # Prepend a segment derived from a field on the def node (e.g. method receiver).
    qualified_name_prefix_from_field: str | None = None


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
class ImportsConfig:
    node_types: tuple[str, ...] = ()
    external_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InheritanceConfig:
    parent_node_types: tuple[str, ...]
    # Solidity-style: walk children of parent_node_types matching `child_node_type`,
    # then descend into the named field to find the base identifier.
    child_node_type: str | None = None
    child_name_field: str | None = None
    # Python-style: the parent has a `bases_field` pointing at a list-like node
    # whose identifier children are the bases.
    bases_field: str | None = None
    # Go-style (interface embedding): drill through one named field on the parent
    # before iterating children. `type_spec.type` → `interface_type` whose
    # `type_elem` children carry the bases.
    child_via_field: str | None = None
    # Go-style (struct embedding): after `child_via_field` lands on a wrapper
    # node, descend into the first child of this type before iterating. Used to
    # bridge `struct_type` → `field_declaration_list` for embedded struct fields.
    child_via_node_type: str | None = None
    # Filter: only iterate children where this named field is absent. Captures
    # Go embedded struct fields, which are `field_declaration` nodes lacking a
    # `name` field — distinguishing them from regular named fields.
    child_only_when_field_absent: str | None = None


@dataclass(frozen=True)
class LanguageConfig:
    language: str
    module_node_type: str
    dependency_paths: tuple[str, ...]
    definitions: tuple[DefinitionRule, ...]
    references: tuple[ReferenceRule, ...]
    calls: tuple[CallRule, ...]
    data_access: DataAccessConfig | None = None
    imports: ImportsConfig | None = None
    # Empty tuple = no inheritance modeled. Each rule can target a distinct
    # AST shape (e.g. Go: one rule for interface embedding, one for struct).
    inheritance: tuple[InheritanceConfig, ...] = ()
    raw: dict = field(default_factory=dict)  # full parsed YAML

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
            name_field_multiple=d.get("name_field_multiple", False),
            qualified_name_prefix_from_field=d.get("qualified_name_prefix_from_field"),
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

    imp_raw = raw.get("imports")
    imp: ImportsConfig | None = None
    if imp_raw:
        imp = ImportsConfig(
            node_types=tuple(imp_raw.get("node_types", [])),
            external_prefixes=tuple(imp_raw.get("external_prefixes", [])),
        )

    inh_raw = raw.get("inheritance")
    inh_rules: list[InheritanceConfig] = []
    # Accept both single-rule (object) and multi-rule (array) forms — Python and
    # Solidity stay as single rules; Go uses an array to model interface vs.
    # struct embedding under the same `type_spec` parent.
    inh_iter: list[dict]
    if inh_raw is None:
        inh_iter = []
    elif isinstance(inh_raw, list):
        inh_iter = inh_raw
    else:
        inh_iter = [inh_raw]
    for r in inh_iter:
        inh_rules.append(
            InheritanceConfig(
                parent_node_types=tuple(r["parent_node_types"]),
                child_node_type=r.get("child_node_type"),
                child_name_field=r.get("child_name_field"),
                bases_field=r.get("bases_field"),
                child_via_field=r.get("child_via_field"),
                child_via_node_type=r.get("child_via_node_type"),
                child_only_when_field_absent=r.get("child_only_when_field_absent"),
            )
        )

    return LanguageConfig(
        language=raw["language"],
        module_node_type=raw["module_node_type"],
        dependency_paths=tuple(raw.get("dependency_paths", [])),
        definitions=defs,
        references=refs,
        calls=calls,
        data_access=da,
        imports=imp,
        inheritance=tuple(inh_rules),
        raw=raw,
    )
