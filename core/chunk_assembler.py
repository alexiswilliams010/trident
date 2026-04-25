"""Tier 3a: graph-informed chunk assembly (Architecture §6.1).

Three granularities per definition:

    function     — anchor function source + state vars it touches
                   + override base signature (always, if exists)
                   + inheritance chain (always, if any bases)
                   + signatures of inherited members from ancestors
                   + signatures of `certain` callees (1 hop)
                   + bodies of certain callees if budget allows
                   + signatures of `inferred` callees if budget allows
                   + synthetic comments for external calls
                   SOFT BUDGET = 512, HARD CAP = 768

    module       — every definition in a file + imports preamble
                   + external dep summary
                   TOKEN_BUDGET = 1024

    cross-module — anchor function source
                   + full source of callees up to 2 hops (any confidence)
                   + shared state across the chain
                   TOKEN_BUDGET = 2048

Each chunk is preceded by a JSON metadata preamble (anchor, kind, language,
file, dependencies, callers, external_deps, overrides, inheritance_chain,
granularity). The preamble is inside the same TEXT we embed so the vector
encodes structural info.

Function chunks use a soft budget (512) plus a hard cap (768) so inheritance
context — the override base signature, ancestor chain, and one or two
inherited member signatures — survives even when the function body is large.
The body, override block, and chain are mandatory; inherited members and
callees are greedy under the hard cap.

Token counting uses tiktoken's `cl100k_base` — overestimates ~10 % on code
versus newer encoders (o200k_base), which is safe for budget fitting.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

import asyncpg
import tiktoken


GRANULARITY_FUNCTION = "function"
GRANULARITY_MODULE = "module"
GRANULARITY_CROSS_MODULE = "cross-module"

TOKEN_BUDGETS = {
    GRANULARITY_FUNCTION: 512,
    GRANULARITY_MODULE: 1024,
    GRANULARITY_CROSS_MODULE: 2048,
}

# Function chunks may exceed the soft budget by up to 50% so inheritance
# context survives. Anything past this is treated as pathological — the
# enrichment greedy fit stops at the hard cap.
HARD_CAP_MULTIPLIER = 1.5

FUNCTION_KINDS = {"function", "method", "constructor", "modifier"}
MEMBER_KINDS = FUNCTION_KINDS | {"modifier"}  # eligible for inherited-member listings
CONTAINER_KINDS = {"contract", "interface", "library", "class"}


# ────────────────────────────────────────────────────────────────────
# Token counting
# ────────────────────────────────────────────────────────────────────


_ENCODER: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


@dataclass
class _DefRow:
    id: int
    file_id: int
    kind: str
    name: str
    qualified_name: str
    scope_id: int | None
    start_byte: int
    end_byte: int
    raw_content: str
    file_path: str
    language: str

    def source(self) -> str:
        return self.raw_content[self.start_byte : self.end_byte]


@dataclass
class _ChunkRow:
    file_id: int
    anchor_def_id: int
    granularity: str
    metadata: dict
    content: str
    token_count: int
    content_hash: str


def _md_preamble(metadata: dict) -> str:
    return "/* tsgrep-meta: " + json.dumps(metadata, separators=(",", ":")) + " */\n\n"


def _format_signature(d: _DefRow) -> str:
    """First line of the def's source — stand-in for a signature."""
    src = d.source()
    first_line = src.splitlines()[0] if src else ""
    return f"# signature: {d.qualified_name} [{d.kind}]\n{first_line}"


def _format_body(d: _DefRow, header: str | None = None) -> str:
    label = header or f"# {d.kind}: {d.qualified_name}"
    return f"{label}\n{d.source()}"


def _hash_content(metadata: dict, content: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    h.update(b"\n")
    h.update(content.encode("utf-8", errors="replace"))
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────
# Loading rows
# ────────────────────────────────────────────────────────────────────


async def _load_defs(conn: asyncpg.Connection, repo_id: int) -> list[_DefRow]:
    rows = await conn.fetch(
        """
        SELECT d.id, d.file_id, d.kind, d.name, d.qualified_name, d.scope_id,
               n.start_byte, n.end_byte,
               f.raw_content, f.path, f.language
        FROM definitions d
        JOIN nodes n ON n.id = d.node_id
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1
        ORDER BY d.id
        """,
        repo_id,
    )
    return [
        _DefRow(
            id=r["id"], file_id=r["file_id"], kind=r["kind"], name=r["name"],
            qualified_name=r["qualified_name"], scope_id=r["scope_id"],
            start_byte=r["start_byte"], end_byte=r["end_byte"],
            raw_content=r["raw_content"] or "", file_path=r["path"], language=r["language"],
        )
        for r in rows
    ]


async def _load_call_edges(conn: asyncpg.Connection, repo_id: int) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT ce.caller_def_id, ce.callee_def_id, ce.callee_name, ce.confidence
        FROM call_edges ce
        JOIN definitions caller ON caller.id = ce.caller_def_id
        JOIN files f ON f.id = caller.file_id
        WHERE f.repo_id = $1
        """,
        repo_id,
    )


async def _load_data_access(conn: asyncpg.Connection, repo_id: int) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT da.accessor_def_id, da.target_def_id, da.access_type
        FROM data_access da
        JOIN definitions accessor ON accessor.id = da.accessor_def_id
        JOIN files f ON f.id = accessor.file_id
        WHERE f.repo_id = $1
        """,
        repo_id,
    )


async def _load_imports(conn: asyncpg.Connection, repo_id: int) -> dict[int, list[asyncpg.Record]]:
    rows = await conn.fetch(
        """
        SELECT i.file_id, i.import_path, i.dep_class, e.package_name
        FROM imports i
        JOIN files f ON f.id = i.file_id
        LEFT JOIN external_dependencies e ON e.id = i.external_dep_id
        WHERE f.repo_id = $1
        """,
        repo_id,
    )
    out: dict[int, list[asyncpg.Record]] = {}
    for row in rows:
        out.setdefault(row["file_id"], []).append(row)
    return out


async def _load_inherits_edges(conn: asyncpg.Connection, repo_id: int) -> list[asyncpg.Record]:
    """All resolved inherits_edges for the repo, ordered by (child, ord) so
    direct-base lists preserve declaration order."""
    return await conn.fetch(
        """
        SELECT ie.child_def_id, ie.base_def_id, ie.ord
        FROM inherits_edges ie
        JOIN definitions d ON d.id = ie.child_def_id
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1 AND ie.base_def_id IS NOT NULL
        ORDER BY ie.child_def_id, ie.ord
        """,
        repo_id,
    )


async def _load_overrides_edges(conn: asyncpg.Connection, repo_id: int) -> list[asyncpg.Record]:
    """All overrides_edges for the repo. Each child has at most one row
    (the resolver picks the nearest ancestor's matching method)."""
    return await conn.fetch(
        """
        SELECT oe.child_def_id, oe.base_def_id
        FROM overrides_edges oe
        JOIN definitions d ON d.id = oe.child_def_id
        JOIN files f ON f.id = d.file_id
        WHERE f.repo_id = $1
        """,
        repo_id,
    )


# ────────────────────────────────────────────────────────────────────
# Index for in-memory traversal
# ────────────────────────────────────────────────────────────────────


@dataclass
class _GraphIndex:
    defs_by_id: dict[int, _DefRow]
    callees_by_caller: dict[int, list[tuple[int | None, str | None, str]]]  # (callee_id, callee_name, conf)
    callers_by_callee: dict[int, list[int]]
    data_access_by_accessor: dict[int, list[tuple[int, str]]]   # (target_def_id, op)
    defs_by_file: dict[int, list[int]]                          # file_id -> [def_id, ...]
    imports_by_file: dict[int, list[asyncpg.Record]]
    # Inheritance: child class def_id → ordered list of direct base class def_ids.
    bases_by_child: dict[int, list[int]]
    # Method override: child method def_id → base method def_id (one or none).
    override_base_by_child: dict[int, int]
    # All members of a container, indexed by scope (contract/class def_id).
    members_by_container: dict[int, list[int]]


def _build_index(
    defs: list[_DefRow],
    edges: list[asyncpg.Record],
    da: list[asyncpg.Record],
    imports: dict[int, list[asyncpg.Record]],
    inherits: list[asyncpg.Record],
    overrides: list[asyncpg.Record],
) -> _GraphIndex:
    defs_by_id = {d.id: d for d in defs}
    callees: dict[int, list[tuple[int | None, str | None, str]]] = {}
    callers: dict[int, list[int]] = {}
    for e in edges:
        callees.setdefault(e["caller_def_id"], []).append(
            (e["callee_def_id"], e["callee_name"], e["confidence"])
        )
        if e["callee_def_id"] is not None:
            callers.setdefault(e["callee_def_id"], []).append(e["caller_def_id"])
    da_by: dict[int, list[tuple[int, str]]] = {}
    for r in da:
        da_by.setdefault(r["accessor_def_id"], []).append((r["target_def_id"], r["access_type"]))
    by_file: dict[int, list[int]] = {}
    for d in defs:
        by_file.setdefault(d.file_id, []).append(d.id)
    bases_by_child: dict[int, list[int]] = {}
    for r in inherits:  # already ordered by (child, ord)
        bases_by_child.setdefault(r["child_def_id"], []).append(r["base_def_id"])
    override_base_by_child: dict[int, int] = {
        r["child_def_id"]: r["base_def_id"] for r in overrides
    }
    members_by_container: dict[int, list[int]] = {}
    for d in defs:
        if d.scope_id is not None and d.kind in MEMBER_KINDS:
            members_by_container.setdefault(d.scope_id, []).append(d.id)
    return _GraphIndex(
        defs_by_id=defs_by_id,
        callees_by_caller=callees,
        callers_by_callee=callers,
        data_access_by_accessor=da_by,
        defs_by_file=by_file,
        imports_by_file=imports,
        bases_by_child=bases_by_child,
        override_base_by_child=override_base_by_child,
        members_by_container=members_by_container,
    )


def _ancestor_chain(idx: _GraphIndex, container_id: int | None) -> list[int]:
    """BFS over `bases_by_child`, closest first. Returns ancestors of
    `container_id`, excluding the container itself."""
    if container_id is None:
        return []
    seen: set[int] = set()
    out: list[int] = []
    frontier = list(idx.bases_by_child.get(container_id, []))
    while frontier:
        next_frontier: list[int] = []
        for a in frontier:
            if a in seen:
                continue
            seen.add(a)
            out.append(a)
            next_frontier.extend(idx.bases_by_child.get(a, []))
        frontier = next_frontier
    return out


def _inherited_members_for(
    idx: _GraphIndex,
    container_id: int | None,
    exclude_def_ids: set[int],
) -> list[_DefRow]:
    """Methods/modifiers defined on any ancestor of `container_id`, deduped by
    name (closer ancestor wins, since BFS yields them in proximity order).
    Excludes any def_id in `exclude_def_ids` (typically the override base
    we're already showing as a dedicated block)."""
    if container_id is None:
        return []
    out: list[_DefRow] = []
    seen_names: set[str] = set()
    for ancestor_id in _ancestor_chain(idx, container_id):
        for member_id in idx.members_by_container.get(ancestor_id, []):
            if member_id in exclude_def_ids:
                continue
            d = idx.defs_by_id.get(member_id)
            if d is None or d.name in seen_names:
                continue
            seen_names.add(d.name)
            out.append(d)
    return out


# ────────────────────────────────────────────────────────────────────
# Chunk builders
# ────────────────────────────────────────────────────────────────────


def _external_callees_for(idx: _GraphIndex, caller_id: int) -> list[str]:
    """Names of unresolved (external / cross-package) callees at the call site."""
    out: list[str] = []
    for callee_id, name, _ in idx.callees_by_caller.get(caller_id, []):
        if callee_id is None and name:
            out.append(name)
    return sorted(set(out))


def _data_access_summary(idx: _GraphIndex, accessor_id: int) -> list[str]:
    rows = idx.data_access_by_accessor.get(accessor_id, [])
    out: list[str] = []
    seen: set[tuple[int, str]] = set()
    for tid, op in rows:
        if (tid, op) in seen:
            continue
        seen.add((tid, op))
        td = idx.defs_by_id.get(tid)
        if td is not None:
            out.append(f"{op}: {td.qualified_name}")
    return out


def _build_function_chunk(d: _DefRow, idx: _GraphIndex) -> _ChunkRow | None:
    if d.kind not in FUNCTION_KINDS:
        return None

    # Required: function source.
    body = _format_body(d)
    parts: list[str] = []
    parts.append(body)

    # Data-access dependencies (state vars touched).
    da_lines = _data_access_summary(idx, d.id)
    if da_lines:
        parts.append("# state vars touched\n" + "\n".join("- " + s for s in da_lines))

    # Callees, in confidence order.
    certain: list[_DefRow] = []
    inferred: list[_DefRow] = []
    for callee_id, _, conf in idx.callees_by_caller.get(d.id, []):
        if callee_id is None:
            continue
        cd = idx.defs_by_id.get(callee_id)
        if cd is None or cd.id == d.id:
            continue
        if conf == "certain":
            certain.append(cd)
        elif conf == "inferred":
            inferred.append(cd)

    # External calls (callees we couldn't resolve).
    ext = _external_callees_for(idx, d.id)
    if ext:
        parts.append("# external calls (unresolved)\n" + "\n".join(f"// external: {x}()" for x in ext))

    # ── Inheritance enrichment (always-include tiers) ──
    # Override base: the specific method this one shadows.
    override_base_id = idx.override_base_by_child.get(d.id)
    override_base = idx.defs_by_id.get(override_base_id) if override_base_id else None
    if override_base is not None:
        parts.append(
            f"# overrides: {override_base.qualified_name}\n{_format_signature(override_base)}"
        )

    # Inheritance chain of the enclosing container (e.g. contract → base → interface).
    container_id = d.scope_id if d.scope_id and idx.defs_by_id.get(d.scope_id) and idx.defs_by_id[d.scope_id].kind in CONTAINER_KINDS else None
    ancestor_ids = _ancestor_chain(idx, container_id)
    if container_id is not None and ancestor_ids:
        chain_names = [idx.defs_by_id[container_id].name] + [
            idx.defs_by_id[a].name for a in ancestor_ids if a in idx.defs_by_id
        ]
        parts.append("# inheritance chain: " + " → ".join(chain_names))

    soft_budget = TOKEN_BUDGETS[GRANULARITY_FUNCTION]
    hard_cap = int(soft_budget * HARD_CAP_MULTIPLIER)

    metadata = {
        "anchor": d.qualified_name,
        "kind": d.kind,
        "language": d.language,
        "file": d.file_path,
        "dependencies": [s.split(": ", 1)[1] for s in da_lines],
        "callers": sorted({idx.defs_by_id[c].qualified_name for c in idx.callers_by_callee.get(d.id, []) if c in idx.defs_by_id}),
        "external_deps": ext,
        "overrides": override_base.qualified_name if override_base is not None else None,
        "inheritance_chain": [
            idx.defs_by_id[a].qualified_name for a in ancestor_ids if a in idx.defs_by_id
        ],
        "granularity": GRANULARITY_FUNCTION,
    }

    def _join(extras: list[str] = []) -> str:
        return _md_preamble(metadata) + "\n\n".join(parts + extras)

    extras: list[str] = []

    # Phase: inherited member signatures (greedy under hard_cap). Excluded:
    # the override base (already shown as its own block above).
    inherited = _inherited_members_for(
        idx, container_id,
        exclude_def_ids={override_base_id} if override_base_id else set(),
    )
    if inherited:
        sigs: list[str] = []
        for m in inherited:
            trial = sigs + [_format_signature(m)]
            block = "# inherited members\n" + "\n\n".join(trial)
            if count_tokens(_join(extras + [block])) > hard_cap:
                break
            sigs = trial
        if sigs:
            extras.append("# inherited members\n" + "\n\n".join(sigs))

    # Phase 1: signatures of certain callees.
    if certain:
        sig_block = "# callees [certain]\n" + "\n\n".join(_format_signature(c) for c in certain)
        if count_tokens(_join(extras + [sig_block])) <= hard_cap:
            extras.append(sig_block)
        else:
            sigs: list[str] = []
            for c in certain:
                trial = sigs + [_format_signature(c)]
                block = "# callees [certain]\n" + "\n\n".join(trial)
                if count_tokens(_join(extras + [block])) > hard_cap:
                    break
                sigs = trial
            if sigs:
                extras.append("# callees [certain]\n" + "\n\n".join(sigs))

    # Phase 2: bodies of certain callees if budget allows.
    if certain:
        body_chunks: list[str] = []
        for c in certain:
            trial = body_chunks + [_format_body(c, header=f"# callee body: {c.qualified_name}")]
            block = "# callee bodies [certain]\n" + "\n\n".join(trial)
            if count_tokens(_join(extras + [block])) > hard_cap:
                break
            body_chunks = trial
        if body_chunks:
            extras.append("# callee bodies [certain]\n" + "\n\n".join(body_chunks))

    # Phase 3: signatures of inferred callees.
    if inferred:
        sigs: list[str] = []
        for c in inferred:
            trial = sigs + [_format_signature(c)]
            block = "# callees [inferred]\n" + "\n\n".join(trial)
            if count_tokens(_join(extras + [block])) > hard_cap:
                break
            sigs = trial
        if sigs:
            extras.append("# callees [inferred]\n" + "\n\n".join(sigs))

    content = _join(extras)
    return _ChunkRow(
        file_id=d.file_id,
        anchor_def_id=d.id,
        granularity=GRANULARITY_FUNCTION,
        metadata=metadata,
        content=content,
        token_count=count_tokens(content),
        content_hash=_hash_content(metadata, content),
    )


def _build_module_chunk(module_def: _DefRow, idx: _GraphIndex) -> _ChunkRow:
    file_id = module_def.file_id
    # All defs in this file (excluding the synthetic module def itself).
    def_ids = idx.defs_by_file.get(file_id, [])
    children = [idx.defs_by_id[i] for i in def_ids if i != module_def.id]

    imports = idx.imports_by_file.get(file_id, [])
    intra = [r["import_path"] for r in imports if r["dep_class"] == "intra_repo"]
    external = sorted({r["package_name"] or r["import_path"] for r in imports if r["dep_class"] == "external"})
    unresolved = [r["import_path"] for r in imports if r["dep_class"] == "unresolved"]

    metadata = {
        "anchor": module_def.qualified_name,
        "kind": "module",
        "language": module_def.language,
        "file": module_def.file_path,
        "dependencies": [c.qualified_name for c in children],
        "external_deps": external,
        "granularity": GRANULARITY_MODULE,
    }

    parts: list[str] = []
    if intra or external or unresolved:
        lines = []
        if intra:
            lines.append("intra-repo: " + ", ".join(intra))
        if external:
            lines.append("external:   " + ", ".join(external))
        if unresolved:
            lines.append("unresolved: " + ", ".join(unresolved))
        parts.append("# imports\n" + "\n".join(lines))

    parts.append(f"# module: {module_def.qualified_name} ({module_def.file_path})")

    # Greedy include children: full body for short ones, otherwise signature.
    budget = TOKEN_BUDGETS[GRANULARITY_MODULE]
    accumulated: list[str] = []
    for c in children:
        # Try full body, fall back to signature.
        for piece in [_format_body(c), _format_signature(c)]:
            trial = accumulated + [piece]
            content_try = _md_preamble(metadata) + "\n\n".join(parts + trial)
            if count_tokens(content_try) <= budget:
                accumulated = trial
                break

    parts.extend(accumulated)
    content = _md_preamble(metadata) + "\n\n".join(parts)
    return _ChunkRow(
        file_id=file_id,
        anchor_def_id=module_def.id,
        granularity=GRANULARITY_MODULE,
        metadata=metadata,
        content=content,
        token_count=count_tokens(content),
        content_hash=_hash_content(metadata, content),
    )


def _build_cross_module_chunk(d: _DefRow, idx: _GraphIndex) -> _ChunkRow | None:
    if d.kind not in FUNCTION_KINDS:
        return None
    # Walk callees up to 2 hops (BFS). Include all confidence levels.
    visited: set[int] = {d.id}
    layer1: list[_DefRow] = []
    for cid, _, _ in idx.callees_by_caller.get(d.id, []):
        if cid is None or cid in visited:
            continue
        visited.add(cid)
        cd = idx.defs_by_id.get(cid)
        if cd is not None:
            layer1.append(cd)
    if not layer1:
        return None  # no cross-module value if no resolved callees

    layer2: list[_DefRow] = []
    for c in layer1:
        for cid, _, _ in idx.callees_by_caller.get(c.id, []):
            if cid is None or cid in visited:
                continue
            visited.add(cid)
            cd = idx.defs_by_id.get(cid)
            if cd is not None:
                layer2.append(cd)

    # Shared state across the chain: union of data_access targets.
    shared_state: list[str] = []
    seen: set[int] = set()
    for accessor in [d, *layer1, *layer2]:
        for tid, op in idx.data_access_by_accessor.get(accessor.id, []):
            if tid in seen:
                continue
            seen.add(tid)
            td = idx.defs_by_id.get(tid)
            if td:
                shared_state.append(f"- {op}: {td.qualified_name}")

    metadata = {
        "anchor": d.qualified_name,
        "kind": d.kind,
        "language": d.language,
        "file": d.file_path,
        "dependencies": [c.qualified_name for c in layer1] + [c.qualified_name for c in layer2],
        "external_deps": _external_callees_for(idx, d.id),
        "granularity": GRANULARITY_CROSS_MODULE,
        "hops": 2,
    }

    parts: list[str] = [_format_body(d, header=f"# anchor: {d.qualified_name}")]
    if shared_state:
        parts.append("# shared state\n" + "\n".join(shared_state))

    budget = TOKEN_BUDGETS[GRANULARITY_CROSS_MODULE]

    def _join(extras: list[str]) -> str:
        return _md_preamble(metadata) + "\n\n".join(parts + extras)

    extras: list[str] = []
    # Layer 1 bodies first.
    layer1_bodies: list[str] = []
    for c in layer1:
        trial = layer1_bodies + [_format_body(c, header=f"# callee (1 hop): {c.qualified_name}")]
        block = "# callees (1 hop)\n" + "\n\n".join(trial)
        if count_tokens(_join([block])) > budget:
            break
        layer1_bodies = trial
    if layer1_bodies:
        extras.append("# callees (1 hop)\n" + "\n\n".join(layer1_bodies))

    # Layer 2 signatures only (saves budget).
    if layer2:
        sigs: list[str] = []
        for c in layer2:
            trial = sigs + [_format_signature(c)]
            block = "# callees (2 hop)\n" + "\n\n".join(trial)
            if count_tokens(_join(extras + [block])) > budget:
                break
            sigs = trial
        if sigs:
            extras.append("# callees (2 hop)\n" + "\n\n".join(sigs))

    content = _join(extras)
    return _ChunkRow(
        file_id=d.file_id,
        anchor_def_id=d.id,
        granularity=GRANULARITY_CROSS_MODULE,
        metadata=metadata,
        content=content,
        token_count=count_tokens(content),
        content_hash=_hash_content(metadata, content),
    )


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────


@dataclass
class ChunkStats:
    n_function: int = 0
    n_module: int = 0
    n_cross_module: int = 0
    n_inserted: int = 0
    n_updated: int = 0
    n_unchanged: int = 0

    @property
    def total(self) -> int:
        return self.n_function + self.n_module + self.n_cross_module


async def _upsert_chunk(conn: asyncpg.Connection, chunk: _ChunkRow) -> str:
    """Returns 'inserted' | 'updated' | 'unchanged'."""
    existing = await conn.fetchrow(
        "SELECT id, content_hash FROM chunks WHERE anchor_def_id=$1 AND granularity=$2",
        chunk.anchor_def_id, chunk.granularity,
    )
    if existing is not None and existing["content_hash"] == chunk.content_hash:
        return "unchanged"
    if existing is not None:
        await conn.execute(
            """
            UPDATE chunks
            SET file_id=$1, content=$2, token_count=$3, metadata=$4, content_hash=$5
            WHERE id=$6
            """,
            chunk.file_id, chunk.content, chunk.token_count,
            json.dumps(chunk.metadata), chunk.content_hash, existing["id"],
        )
        # Embedding for this chunk is invalid — drop it.
        await conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id=$1", existing["id"])
        return "updated"
    await conn.execute(
        """
        INSERT INTO chunks (file_id, anchor_def_id, granularity, content, token_count, metadata, content_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        chunk.file_id, chunk.anchor_def_id, chunk.granularity,
        chunk.content, chunk.token_count, json.dumps(chunk.metadata), chunk.content_hash,
    )
    return "inserted"


async def assemble_chunks(pool: asyncpg.Pool, repo_id: int) -> ChunkStats:
    """Build all chunks at all three granularities for a repo."""
    stats = ChunkStats()
    async with pool.acquire() as conn:
        defs = await _load_defs(conn, repo_id)
        edges = await _load_call_edges(conn, repo_id)
        da = await _load_data_access(conn, repo_id)
        imports = await _load_imports(conn, repo_id)
        inherits = await _load_inherits_edges(conn, repo_id)
        overrides = await _load_overrides_edges(conn, repo_id)
        idx = _build_index(defs, edges, da, imports, inherits, overrides)

        chunks: list[_ChunkRow] = []
        for d in defs:
            if d.kind in FUNCTION_KINDS:
                fc = _build_function_chunk(d, idx)
                if fc:
                    chunks.append(fc)
                    stats.n_function += 1
                cm = _build_cross_module_chunk(d, idx)
                if cm:
                    chunks.append(cm)
                    stats.n_cross_module += 1
            elif d.kind == "module":
                mc = _build_module_chunk(d, idx)
                chunks.append(mc)
                stats.n_module += 1

        async with conn.transaction():
            for c in chunks:
                outcome = await _upsert_chunk(conn, c)
                if outcome == "inserted":
                    stats.n_inserted += 1
                elif outcome == "updated":
                    stats.n_updated += 1
                else:
                    stats.n_unchanged += 1

    return stats


def assemble_chunks_sync(repo_id: int, dsn: str | None = None) -> ChunkStats:
    from db.connection import pool_ctx

    async def _run():
        async with pool_ctx(dsn) as pool:
            return await assemble_chunks(pool, repo_id)

    return asyncio.run(_run())
