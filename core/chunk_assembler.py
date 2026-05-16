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

Branch model:
  • Definitions are content-shared (per file_version), so the same anchor_def
    can have one chunk per branch (UNIQUE on `branch_id, anchor_def_id,
    granularity`). Per-branch is necessary because the skeleton inlined into
    the chunk content references branch-resolved cross-file targets.
  • `chunk_embeddings` is keyed by chunk content_hash, so when two branches
    happen to assemble byte-identical chunks (typical for an unchanged file
    with stable cross-file edges), the embedding row is reused.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
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


# Approximate token cost of joining two pieces with "\n\n". cl100k_base
# encodes "\n\n" as a single token; the BPE may merge across boundaries with
# slightly different totals, but the error is sub-1% and the budgets are soft
# targets, so a small constant is good enough to avoid re-tokenizing.
_JOIN_TOK = 1


def _greedy_pack_pieces(
    base_tokens: int,
    candidates: Iterable[list[str]],
    budget: int,
) -> list[str]:
    """Append-with-fallback pattern (used by `_build_module_chunk` for
    children: try the body, fall back to the signature, skip if neither fits).

    Each `candidates` element is an ordered list of fallbacks for one slot;
    the first that fits is appended. Returns the accepted pieces in order.

    Replaces an O(N²) re-tokenization with O(N): each piece is tokenized
    exactly once and a running total is compared against the budget.
    """
    accepted: list[str] = []
    used = 0
    for fallbacks in candidates:
        for piece in fallbacks:
            cost = count_tokens(piece) + _JOIN_TOK
            if base_tokens + used + cost <= budget:
                accepted.append(piece)
                used += cost
                break
    return accepted


def _greedy_pack_section(
    base_tokens: int,
    header: str,
    candidates: Iterable[str],
    budget: int,
) -> str | None:
    """Headed-section pattern (used everywhere else: callees [certain]
    sigs/bodies, inferred sigs, inherited members, layer1/2 hops).

    Builds a block of the form `header + "\n" + "\n\n".join(accepted)` while
    `base_tokens + block_tokens` stays under `budget`. Returns the block
    string, or None if no candidates fit.
    """
    header_with_nl_tok = count_tokens(header + "\n")
    accepted: list[str] = []
    inner_tokens = 0  # tokens of the joined candidates so far
    for piece in candidates:
        piece_tok = count_tokens(piece)
        new_inner = inner_tokens + (_JOIN_TOK + piece_tok if accepted else piece_tok)
        # The whole block costs sep + header + "\n" + inner, glued onto the
        # base content by another sep.
        proposed = base_tokens + _JOIN_TOK + header_with_nl_tok + new_inner
        if proposed > budget:
            break
        accepted.append(piece)
        inner_tokens = new_inner
    if not accepted:
        return None
    return header + "\n" + "\n\n".join(accepted)


# Beyond a few dozen entries the `dependencies` listing in metadata stops
# being useful retrieval signal — and on pathological inputs (auto-generated
# bindings, minified bundles) the JSON serialization of thousands of names
# alone exceeded the chunk's token budget. Cap and tag.
MAX_DEPS_IN_METADATA = 50

# Defense-in-depth ceiling per granularity. Anything past this is replaced
# with a minimal placeholder chunk before persistence — see
# `_degrade_if_oversize`. The embedder's `DEFAULT_MAX_INPUT_TOKENS` is the
# outermost guard; these caps catch things earlier so the `chunks` table
# doesn't accumulate unembeddable rows.
HARD_OUTPUT_CAP = {
    GRANULARITY_FUNCTION:     4000,
    GRANULARITY_MODULE:       6000,
    GRANULARITY_CROSS_MODULE: 8000,
}

_METADATA_TRIM_ORDER = (
    "dependencies_truncated",
    "dependencies",
    "external_deps",
    "inheritance_chain",
    "overrides",
    "callers",
)


def _truncate_deps(deps: list[str]) -> tuple[list[str], int]:
    """Cap the dependency list. Returns (capped, n_truncated)."""
    if len(deps) <= MAX_DEPS_IN_METADATA:
        return deps, 0
    return deps[:MAX_DEPS_IN_METADATA], len(deps) - MAX_DEPS_IN_METADATA


def _degrade_if_oversize(
    granularity: str,
    metadata: dict,
    content: str,
    anchor_label: str,
    file_path: str,
    body_only: str = "",
) -> tuple[dict, str, int]:
    """If `content` exceeds the per-granularity hard cap, first try shedding
    the enrichment context (callee bodies, signatures, inheritance summaries)
    and embed just the anchor body. Only when even body-alone overflows —
    typically generated/minified code — fall back to a minimal stub.
    """
    tc = count_tokens(content)
    cap = HARD_OUTPUT_CAP.get(granularity)
    if cap is None or tc <= cap:
        return metadata, content, tc

    if body_only:
        shed_metadata = {**metadata, "degraded": "enrichment_shed", "original_token_count": tc}
        shed_content = _md_preamble(shed_metadata) + body_only
        shed_tc = count_tokens(shed_content)
        if shed_tc <= cap:
            return shed_metadata, shed_content, shed_tc

        trimmed = {**metadata, "degraded": "metadata_trimmed", "original_token_count": tc}
        for key in _METADATA_TRIM_ORDER:
            if not trimmed.get(key):
                continue
            del trimmed[key]
            attempt_content = _md_preamble(trimmed) + body_only
            attempt_tc = count_tokens(attempt_content)
            if attempt_tc <= cap:
                return trimmed, attempt_content, attempt_tc

    degraded = {**metadata, "degraded": "oversize", "original_token_count": tc}
    stub = f"# {granularity} (degraded): {anchor_label} ({file_path})"
    new_content = _md_preamble(degraded) + "\n\n" + stub
    return degraded, new_content, count_tokens(new_content)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


@dataclass
class _DefRow:
    id: int
    file_version_id: int
    kind: str
    name: str
    qualified_name: str
    scope_id: int | None
    start_byte: int
    end_byte: int
    # UTF-8 bytes, not str — tree-sitter offsets are byte offsets and slicing
    # a Python str by them silently drifts on every multi-byte character.
    raw_content: bytes
    file_path: str
    language: str

    def source(self) -> str:
        return self.raw_content[self.start_byte : self.end_byte].decode("utf-8", errors="replace")


@dataclass
class _ChunkRow:
    branch_id: int
    file_version_id: int
    anchor_def_id: int
    granularity: str
    metadata: dict
    content: str
    token_count: int
    content_hash: str


def _md_preamble(metadata: dict) -> str:
    return "/* trident-meta: " + json.dumps(metadata, separators=(",", ":")) + " */\n\n"


def _semantic_header(d: _DefRow, idx: _GraphIndex) -> str:
    """Natural-language line placed at the top of a chunk so the embedding
    model gets a clean relevance signal in its earliest tokens."""
    name = d.name.lstrip("_")
    words = _split_camel(name).replace("_", " ").lower()
    container = None
    if d.scope_id:
        cd = idx.defs_by_id.get(d.scope_id)
        if cd and cd.kind in CONTAINER_KINDS:
            container = cd.name
    parts = [words]
    if container:
        parts.append(container)
    parts.append(f"{d.language} {d.kind}")
    return " | ".join(parts)


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
# Loading rows (branch-scoped)
# ────────────────────────────────────────────────────────────────────


async def _load_defs(conn: asyncpg.Connection, branch_id: int) -> list[_DefRow]:
    """Load all defs whose file_version is mapped by this branch, hydrating
    raw_content + path from file_versions + branch_files."""
    # DISTINCT ON (fv.id): a file_version can be reached by multiple branch_files
    # paths (identical content). Pick one deterministic path per fv so the join
    # below doesn't multiply each definition by the number of sharing paths.
    file_rows = await conn.fetch(
        """
        SELECT DISTINCT ON (fv.id)
            fv.id, fv.raw_content, bf.path, fv.language
        FROM branch_files bf
        JOIN file_versions fv ON fv.id = bf.file_version_id
        WHERE bf.branch_id = $1
        ORDER BY fv.id, bf.path
        """,
        branch_id,
    )
    file_meta: dict[int, tuple[bytes, str, str]] = {
        r["id"]: ((r["raw_content"] or "").encode("utf-8"), r["path"], r["language"])
        for r in file_rows
    }

    def_rows = await conn.fetch(
        """
        SELECT d.id, d.file_version_id, d.kind, d.name, d.qualified_name, d.scope_id,
               n.start_byte, n.end_byte
        FROM definitions d
        JOIN nodes n ON n.id = d.node_id
        WHERE d.file_version_id IN (
            SELECT file_version_id FROM branch_files WHERE branch_id = $1
        )
        ORDER BY d.id
        """,
        branch_id,
    )
    out: list[_DefRow] = []
    for r in def_rows:
        meta = file_meta.get(r["file_version_id"])
        if meta is None:
            continue
        raw_content, file_path, language = meta
        out.append(
            _DefRow(
                id=r["id"], file_version_id=r["file_version_id"], kind=r["kind"], name=r["name"],
                qualified_name=r["qualified_name"], scope_id=r["scope_id"],
                start_byte=r["start_byte"], end_byte=r["end_byte"],
                raw_content=raw_content, file_path=file_path, language=language,
            )
        )
    return out


async def _load_call_edges(conn: asyncpg.Connection, branch_id: int) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT ce.caller_def_id, ce.callee_def_id, ce.callee_name, ce.confidence
        FROM call_edges ce
        WHERE ce.branch_id = $1
        """,
        branch_id,
    )


async def _load_data_access(conn: asyncpg.Connection, branch_id: int) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT da.accessor_def_id, da.target_def_id, da.access_type
        FROM data_access da
        WHERE da.branch_id = $1
        """,
        branch_id,
    )


async def _load_imports(conn: asyncpg.Connection, branch_id: int) -> dict[int, list[asyncpg.Record]]:
    rows = await conn.fetch(
        """
        SELECT i.file_version_id, i.import_path, i.dep_class, e.package_name
        FROM imports i
        LEFT JOIN external_dependencies e ON e.id = i.external_dep_id
        WHERE i.branch_id = $1
        """,
        branch_id,
    )
    out: dict[int, list[asyncpg.Record]] = {}
    for row in rows:
        out.setdefault(row["file_version_id"], []).append(row)
    return out


async def _load_inherits_edges(conn: asyncpg.Connection, branch_id: int) -> list[asyncpg.Record]:
    """Resolved inherits_edges for the branch, ordered by (child, ord) so
    direct-base lists preserve declaration order."""
    return await conn.fetch(
        """
        SELECT ie.child_def_id, ie.base_def_id, ie.ord
        FROM inherits_edges ie
        WHERE ie.branch_id = $1 AND ie.base_def_id IS NOT NULL
        ORDER BY ie.child_def_id, ie.ord
        """,
        branch_id,
    )


async def _load_overrides_edges(conn: asyncpg.Connection, branch_id: int) -> list[asyncpg.Record]:
    """All overrides_edges for the branch."""
    return await conn.fetch(
        """
        SELECT oe.child_def_id, oe.base_def_id
        FROM overrides_edges oe
        WHERE oe.branch_id = $1
        """,
        branch_id,
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
    defs_by_file_version: dict[int, list[int]]                  # file_version_id -> [def_id, ...]
    imports_by_file_version: dict[int, list[asyncpg.Record]]
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
        by_file.setdefault(d.file_version_id, []).append(d.id)
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
        defs_by_file_version=by_file,
        imports_by_file_version=imports,
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
    name (closer ancestor wins, since BFS yields them in proximity order)."""
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


def _build_function_chunk(d: _DefRow, idx: _GraphIndex, branch_id: int) -> _ChunkRow | None:
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
    override_base_id = idx.override_base_by_child.get(d.id)
    override_base = idx.defs_by_id.get(override_base_id) if override_base_id else None
    if override_base is not None:
        parts.append(
            f"# overrides: {override_base.qualified_name}\n{_format_signature(override_base)}"
        )

    container_id = d.scope_id if d.scope_id and idx.defs_by_id.get(d.scope_id) and idx.defs_by_id[d.scope_id].kind in CONTAINER_KINDS else None
    ancestor_ids = _ancestor_chain(idx, container_id)
    if container_id is not None and ancestor_ids:
        chain_names = [idx.defs_by_id[container_id].name] + [
            idx.defs_by_id[a].name for a in ancestor_ids if a in idx.defs_by_id
        ]
        parts.append("# inheritance chain: " + " → ".join(chain_names))

    soft_budget = TOKEN_BUDGETS[GRANULARITY_FUNCTION]
    hard_cap = int(soft_budget * HARD_CAP_MULTIPLIER)

    deps_full = [s.split(": ", 1)[1] for s in da_lines]
    deps_capped, deps_truncated = _truncate_deps(deps_full)
    metadata = {
        "anchor": d.qualified_name,
        "kind": d.kind,
        "language": d.language,
        "file": d.file_path,
        "dependencies": deps_capped,
        "callers": sorted({idx.defs_by_id[c].qualified_name for c in idx.callers_by_callee.get(d.id, []) if c in idx.defs_by_id}),
        "external_deps": ext,
        "overrides": override_base.qualified_name if override_base is not None else None,
        "inheritance_chain": [
            idx.defs_by_id[a].qualified_name for a in ancestor_ids if a in idx.defs_by_id
        ],
        "granularity": GRANULARITY_FUNCTION,
    }
    if deps_truncated:
        metadata["dependencies_truncated"] = deps_truncated

    sem_header = _semantic_header(d, idx)

    def _join(extras: list[str] = []) -> str:
        return sem_header + "\n\n" + _md_preamble(metadata) + "\n\n".join(parts + extras)

    extras: list[str] = []
    base_tokens = count_tokens(_join())

    def _commit(block: str) -> None:
        nonlocal base_tokens
        extras.append(block)
        base_tokens += _JOIN_TOK + count_tokens(block)

    inherited = _inherited_members_for(
        idx, container_id,
        exclude_def_ids={override_base_id} if override_base_id else set(),
    )
    if inherited:
        block = _greedy_pack_section(
            base_tokens, "# inherited members",
            (_format_signature(m) for m in inherited), hard_cap,
        )
        if block is not None:
            _commit(block)

    if certain:
        sig_block = "# callees [certain]\n" + "\n\n".join(_format_signature(c) for c in certain)
        if base_tokens + _JOIN_TOK + count_tokens(sig_block) <= hard_cap:
            _commit(sig_block)
        else:
            block = _greedy_pack_section(
                base_tokens, "# callees [certain]",
                (_format_signature(c) for c in certain), hard_cap,
            )
            if block is not None:
                _commit(block)

    if certain:
        block = _greedy_pack_section(
            base_tokens, "# callee bodies [certain]",
            (_format_body(c, header=f"# callee body: {c.qualified_name}") for c in certain),
            hard_cap,
        )
        if block is not None:
            _commit(block)

    if inferred:
        block = _greedy_pack_section(
            base_tokens, "# callees [inferred]",
            (_format_signature(c) for c in inferred), hard_cap,
        )
        if block is not None:
            _commit(block)

    content = _join(extras)
    metadata, content, token_count = _degrade_if_oversize(
        GRANULARITY_FUNCTION, metadata, content,
        anchor_label=d.qualified_name, file_path=d.file_path,
        body_only=body,
    )
    return _ChunkRow(
        branch_id=branch_id,
        file_version_id=d.file_version_id,
        anchor_def_id=d.id,
        granularity=GRANULARITY_FUNCTION,
        metadata=metadata,
        content=content,
        token_count=token_count,
        content_hash=_hash_content(metadata, content),
    )


def _build_module_chunk(module_def: _DefRow, idx: _GraphIndex, branch_id: int) -> _ChunkRow:
    file_version_id = module_def.file_version_id
    def_ids = idx.defs_by_file_version.get(file_version_id, [])
    children = [idx.defs_by_id[i] for i in def_ids if i != module_def.id]

    imports = idx.imports_by_file_version.get(file_version_id, [])
    intra = [r["import_path"] for r in imports if r["dep_class"] == "intra_repo"]
    external = sorted({r["package_name"] or r["import_path"] for r in imports if r["dep_class"] == "external"})
    unresolved = [r["import_path"] for r in imports if r["dep_class"] == "unresolved"]

    deps_full = [c.qualified_name for c in children]
    deps_capped, deps_truncated = _truncate_deps(deps_full)
    metadata = {
        "anchor": module_def.qualified_name,
        "kind": "module",
        "language": module_def.language,
        "file": module_def.file_path,
        "dependencies": deps_capped,
        "external_deps": external,
        "granularity": GRANULARITY_MODULE,
    }
    if deps_truncated:
        metadata["dependencies_truncated"] = deps_truncated

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

    budget = TOKEN_BUDGETS[GRANULARITY_MODULE]
    base_tokens = count_tokens(_md_preamble(metadata) + "\n\n".join(parts))
    accumulated = _greedy_pack_pieces(
        base_tokens,
        ([_format_body(c), _format_signature(c)] for c in children),
        budget,
    )

    parts.extend(accumulated)
    content = _md_preamble(metadata) + "\n\n".join(parts)
    metadata, content, token_count = _degrade_if_oversize(
        GRANULARITY_MODULE, metadata, content,
        anchor_label=module_def.qualified_name, file_path=module_def.file_path,
    )
    return _ChunkRow(
        branch_id=branch_id,
        file_version_id=file_version_id,
        anchor_def_id=module_def.id,
        granularity=GRANULARITY_MODULE,
        metadata=metadata,
        content=content,
        token_count=token_count,
        content_hash=_hash_content(metadata, content),
    )


def _build_cross_module_chunk(d: _DefRow, idx: _GraphIndex, branch_id: int) -> _ChunkRow | None:
    if d.kind not in FUNCTION_KINDS:
        return None
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
        return None

    layer2: list[_DefRow] = []
    for c in layer1:
        for cid, _, _ in idx.callees_by_caller.get(c.id, []):
            if cid is None or cid in visited:
                continue
            visited.add(cid)
            cd = idx.defs_by_id.get(cid)
            if cd is not None:
                layer2.append(cd)

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

    deps_full = [c.qualified_name for c in layer1] + [c.qualified_name for c in layer2]
    deps_capped, deps_truncated = _truncate_deps(deps_full)
    metadata = {
        "anchor": d.qualified_name,
        "kind": d.kind,
        "language": d.language,
        "file": d.file_path,
        "dependencies": deps_capped,
        "external_deps": _external_callees_for(idx, d.id),
        "granularity": GRANULARITY_CROSS_MODULE,
        "hops": 2,
    }
    if deps_truncated:
        metadata["dependencies_truncated"] = deps_truncated

    src = d.source()
    anchor_first_line = src.splitlines()[0] if src else ""
    parts: list[str] = [f"# anchor: {d.qualified_name} [{d.kind}]\n{anchor_first_line}"]
    if shared_state:
        parts.append("# shared state\n" + "\n".join(shared_state))

    budget = TOKEN_BUDGETS[GRANULARITY_CROSS_MODULE]
    sem_header = _semantic_header(d, idx)

    def _join(extras: list[str]) -> str:
        return sem_header + "\n\n" + _md_preamble(metadata) + "\n\n".join(parts + extras)

    extras: list[str] = []
    base_tokens = count_tokens(_join([]))

    def _commit(block: str) -> None:
        nonlocal base_tokens
        extras.append(block)
        base_tokens += _JOIN_TOK + count_tokens(block)

    block = _greedy_pack_section(
        base_tokens, "# callees (1 hop)",
        (_format_body(c, header=f"# callee (1 hop): {c.qualified_name}") for c in layer1),
        budget,
    )
    if block is not None:
        _commit(block)

    if layer2:
        block = _greedy_pack_section(
            base_tokens, "# callees (2 hop)",
            (_format_signature(c) for c in layer2),
            budget,
        )
        if block is not None:
            _commit(block)

    content = _join(extras)
    metadata, content, token_count = _degrade_if_oversize(
        GRANULARITY_CROSS_MODULE, metadata, content,
        anchor_label=d.qualified_name, file_path=d.file_path,
        body_only=parts[0],
    )
    return _ChunkRow(
        branch_id=branch_id,
        file_version_id=d.file_version_id,
        anchor_def_id=d.id,
        granularity=GRANULARITY_CROSS_MODULE,
        metadata=metadata,
        content=content,
        token_count=token_count,
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


_CAMEL_BOUNDARY_RE = re.compile(r"([a-z\d])([A-Z])|([A-Z]+)([A-Z][a-z])")


def _split_camel(s: str) -> str:
    """`fooBar` → `foo Bar`; `XMLParser` → `XML Parser`."""
    return _CAMEL_BOUNDARY_RE.sub(
        lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}",
        s,
    )


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _expand_idents(text: str) -> str:
    """Append camelCase- and snake_case-split forms of every identifier-shaped
    token so the english parser indexes both `getuserbyid` and the subtokens.
    """
    extras: list[str] = []
    for m in _IDENT_RE.finditer(text):
        tok = m.group(0)
        if "_" in tok:
            extras.extend(p for p in tok.split("_") if p)
        camel = _split_camel(tok)
        if camel != tok:
            extras.append(camel)
    return text if not extras else text + " " + " ".join(extras)


def _fts_text(qualified_name: str | None, content: str) -> str:
    qn = qualified_name or ""
    return f"{qn} {_split_camel(qn)} {_expand_idents(content)}"


async def _bulk_persist_chunks(
    conn: asyncpg.Connection, branch_id: int, chunks: list[_ChunkRow], stats: ChunkStats,
) -> None:
    """Partition `chunks` into insert / update / unchanged against the existing
    rows for this branch, then issue at most one INSERT and one UPDATE for
    the whole set.

    chunk_embeddings are content-keyed and shared across chunks (and
    branches): we never delete embedding rows here. If a chunk's content
    changed, the new content_hash either already has an embedding (reused)
    or will be embedded by the next embedder pass. Stale embeddings are
    reclaimed by `make gc`.
    """
    if not chunks:
        return

    existing = await conn.fetch(
        """
        SELECT id, anchor_def_id, granularity, content_hash
        FROM chunks WHERE branch_id=$1
        """,
        branch_id,
    )
    existing_by_key: dict[tuple[int, str], tuple[int, str]] = {
        (r["anchor_def_id"], r["granularity"]): (r["id"], r["content_hash"]) for r in existing
    }

    to_insert: list[_ChunkRow] = []
    to_update: list[tuple[int, _ChunkRow]] = []  # (existing chunk id, new row)
    for c in chunks:
        ex = existing_by_key.get((c.anchor_def_id, c.granularity))
        if ex is None:
            to_insert.append(c)
        elif ex[1] != c.content_hash:
            to_update.append((ex[0], c))
        else:
            stats.n_unchanged += 1

    if to_insert:
        await conn.execute(
            """
            INSERT INTO chunks
                (branch_id, file_version_id, anchor_def_id, granularity, content,
                 token_count, metadata, content_hash, fts_doc)
            SELECT $1, fv_id, anchor, gran, content, tc, meta, ch,
                   to_tsvector('english', fts_text)
            FROM UNNEST(
                $2::bigint[], $3::bigint[], $4::text[], $5::text[],
                $6::int[],    $7::jsonb[],  $8::text[], $9::text[]
            ) AS t(fv_id, anchor, gran, content, tc, meta, ch, fts_text)
            """,
            branch_id,
            [c.file_version_id for c in to_insert],
            [c.anchor_def_id for c in to_insert],
            [c.granularity for c in to_insert],
            [c.content for c in to_insert],
            [c.token_count for c in to_insert],
            [json.dumps(c.metadata) for c in to_insert],
            [c.content_hash for c in to_insert],
            [_fts_text(c.metadata.get("anchor"), c.content) for c in to_insert],
        )
        stats.n_inserted += len(to_insert)

    if to_update:
        await conn.execute(
            """
            UPDATE chunks SET
                file_version_id = u.fv_id,
                content         = u.content,
                token_count     = u.tc,
                metadata        = u.meta,
                content_hash    = u.ch,
                fts_doc         = to_tsvector('english', u.fts_text)
            FROM UNNEST(
                $1::bigint[], $2::bigint[], $3::text[], $4::int[],
                $5::jsonb[],  $6::text[],   $7::text[]
            ) AS u(id, fv_id, content, tc, meta, ch, fts_text)
            WHERE chunks.id = u.id
            """,
            [eid for eid, _ in to_update],
            [c.file_version_id for _, c in to_update],
            [c.content for _, c in to_update],
            [c.token_count for _, c in to_update],
            [json.dumps(c.metadata) for _, c in to_update],
            [c.content_hash for _, c in to_update],
            [_fts_text(c.metadata.get("anchor"), c.content) for _, c in to_update],
        )
        stats.n_updated += len(to_update)


async def assemble_chunks(pool: asyncpg.Pool, repo_id: int, branch_id: int) -> ChunkStats:
    """Build all chunks at all three granularities for a (repo, branch).

    `repo_id` is currently unused in the body — branch_id implies repo via
    the branches table — but kept in the signature so callers can pass both
    explicitly without a redundant lookup.
    """
    del repo_id  # implied by branch_id; kept in signature for caller ergonomics
    stats = ChunkStats()
    async with pool.acquire() as conn:
        defs = await _load_defs(conn, branch_id)
        edges = await _load_call_edges(conn, branch_id)
        da = await _load_data_access(conn, branch_id)
        imports = await _load_imports(conn, branch_id)
        inherits = await _load_inherits_edges(conn, branch_id)
        overrides = await _load_overrides_edges(conn, branch_id)
        idx = _build_index(defs, edges, da, imports, inherits, overrides)

        chunks: list[_ChunkRow] = []
        for d in defs:
            if d.kind in FUNCTION_KINDS:
                fc = _build_function_chunk(d, idx, branch_id)
                if fc:
                    chunks.append(fc)
                    stats.n_function += 1
                cm = _build_cross_module_chunk(d, idx, branch_id)
                if cm:
                    chunks.append(cm)
                    stats.n_cross_module += 1
            elif d.kind == "module":
                mc = _build_module_chunk(d, idx, branch_id)
                chunks.append(mc)
                stats.n_module += 1

        async with conn.transaction():
            await _bulk_persist_chunks(conn, branch_id, chunks, stats)

    return stats


def assemble_chunks_sync(repo_id: int, branch_id: int, dsn: str | None = None) -> ChunkStats:
    from db.connection import pool_ctx

    async def _run():
        async with pool_ctx(dsn) as pool:
            return await assemble_chunks(pool, repo_id, branch_id)

    return asyncio.run(_run())
