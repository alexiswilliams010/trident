"""Typed JSON schemas and formatting helpers for CLI output.

TypedDicts document the exact JSON shape for each command so agents can
reference the types to parse structured output.
"""

from __future__ import annotations

import json
import sys
from typing import TypedDict

from core.graph import DefInfo, ImportInfo, InheritanceNode


# ────────────────────────────────────────────────────────────────────
# JSON schemas (TypedDict)
# ────────────────────────────────────────────────────────────────────


class DefInfoJSON(TypedDict):
    def_id: int
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    visibility: str | None
    source: str | None


class DefWithDepthJSON(TypedDict):
    def_id: int
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    visibility: str | None
    source: str | None
    depth: int


class GraphResultJSON(TypedDict):
    command: str
    target: str
    repo: str
    results: list[DefInfoJSON]


class DepthResultJSON(TypedDict):
    command: str
    target: str
    repo: str
    results: list[DefWithDepthJSON]


class PathsResultJSON(TypedDict):
    command: str
    source: str
    target: str
    repo: str
    paths: list[list[DefInfoJSON]]


class ImportInfoJSON(TypedDict):
    import_path: str
    imported_names: list[str]
    dep_class: str
    resolved_file: str | None
    file_path: str


class ImportsResultJSON(TypedDict):
    command: str
    repo: str
    file: str | None
    results: list[ImportInfoJSON]


class InheritanceNodeJSON(TypedDict):
    def_id: int
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    visibility: str | None
    bases: list[str]
    children: list[str]


class InheritanceResultJSON(TypedDict):
    command: str
    target: str
    repo: str
    results: list[InheritanceNodeJSON]


# ────────────────────────────────────────────────────────────────────
# Serialization helpers
# ────────────────────────────────────────────────────────────────────


def def_to_json(d: DefInfo) -> dict:
    out: dict = {
        "def_id": d.def_id,
        "name": d.name,
        "qualified_name": d.qualified_name,
        "kind": d.kind,
        "file_path": d.file_path,
        "start_line": d.start_line,
        "end_line": d.end_line,
        "visibility": d.visibility,
    }
    if d.source is not None:
        out["source"] = d.source
    if d.depth is not None:
        out["depth"] = d.depth
    return out


def import_to_json(imp: ImportInfo) -> dict:
    return {
        "import_path": imp.import_path,
        "imported_names": imp.imported_names,
        "dep_class": imp.dep_class,
        "resolved_file": imp.resolved_file,
        "file_path": imp.file_path,
    }


def inheritance_node_to_json(node: InheritanceNode) -> dict:
    out = def_to_json(node.def_info)
    out["bases"] = node.bases
    out["children"] = node.children
    return out


# ────────────────────────────────────────────────────────────────────
# Output formatting
# ────────────────────────────────────────────────────────────────────


def format_def(d: DefInfo, show_source: bool = False) -> str:
    loc = f"{d.file_path}:{d.start_line}-{d.end_line}"
    vis = f"  {d.visibility}" if d.visibility else ""
    depth = f"  depth={d.depth}" if d.depth is not None else ""
    line = f"  {d.qualified_name:50s} {d.kind:12s} {loc}{vis}{depth}"
    if show_source and d.source:
        line += f"\n{'─' * 60}\n{d.source}\n{'─' * 60}"
    return line


def format_import(imp: ImportInfo) -> str:
    names = ", ".join(imp.imported_names) if imp.imported_names else "*"
    resolved = f" → {imp.resolved_file}" if imp.resolved_file else ""
    return f"  {imp.import_path:40s} [{imp.dep_class}] ({names}){resolved}"


def format_inheritance_node(node: InheritanceNode) -> str:
    d = node.def_info
    loc = f"{d.file_path}:{d.start_line}-{d.end_line}"
    bases = f"  bases=[{', '.join(node.bases)}]" if node.bases else ""
    children = f"  children=[{', '.join(node.children)}]" if node.children else ""
    return f"  {d.qualified_name:50s} {d.kind:12s} {loc}{bases}{children}"


def emit(data: dict, as_json: bool) -> None:
    if as_json:
        json.dump(data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
