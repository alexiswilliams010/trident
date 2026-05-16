"""Tier 2: YAML-driven semantic resolver (within-file).

Produces `definitions`, `"references"`, `call_edges`, `data_access` rows from
the CST stored in `nodes`. Cross-file import resolution and the cross-file
references it enables are Phase 3 work.

Pipeline (Architecture §5.2 — P4 is deferred to Phase 3):
    P1 Definitions         — match definition rules, attach scope_id + qualified_name
    P2 Scope tree          — implicit; built as a (scope_def_id, name) → def_id index
    P3 References          — match reference rules, walk scope chain to resolve
    P5 Calls               — resolve callee, tag confidence (certain/inferred/uncertain)
    P6 Data access         — function references targeting variable-kind defs

Branch-aware model:
  • file_versions / nodes / definitions are content-keyed and shared across
    branches. Tier-1 emission (definitions, intra-file scope tree) only runs
    once per file_version, even when multiple branches index that content.
    The first branch to resolve the file_version writes definitions; later
    branches enter "hydrate mode" — same walk, but instead of allocating new
    def_ids and INSERTing, look up existing def_ids from the DB and just
    rebuild the in-memory scope tables.

  • References, calls, data_access, intra-file inheritance edges are tagged
    with branch_id and regenerated per branch (cross-file resolution depends
    on which files are visible in the branch).

The resolver re-parses each file from `file_versions.raw_content` (cheap;
tree-sitter parses millions of LOC/sec). DB ids are paired to ts_nodes by
walking in the same DFS preorder used by the Tier 1 extractor.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator

import asyncpg

from db.connection import reserve_definition_ids

from .config_loader import LanguageConfig, load_language_config
from .grammar_meta import LANGUAGES
from .languages import SemanticContext, get_handler


# ────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────


@dataclass
class _DefRecord:
    ts_node: object
    db_node_id: int
    kind: str
    name: str
    scope_id: int | None  # def_id of enclosing scope (set after parent scope is inserted)
    qualified_name: str
    visibility: str | None
    is_scope_boundary: bool


@dataclass
class _RefRecord:
    db_node_id: int
    file_version_id: int
    name: str
    confidence: str  # 'certain' | 'inferred' | 'uncertain'
    target_def_id: int | None


@dataclass
class _CallRecord:
    callsite_db_node_id: int
    caller_def_id: int | None
    callee_def_id: int | None
    callee_name: str | None
    confidence: str


@dataclass
class _DataAccessRecord:
    accessor_def_id: int
    target_def_id: int
    access_type: str  # 'read' | 'write'
    db_node_id: int


@dataclass
class ResolveFileResult:
    file_version_id: int
    language: str
    n_definitions: int
    n_references: int
    n_call_edges: int
    n_data_access: int
    n_inherits_edges: int = 0


# ────────────────────────────────────────────────────────────────────
# Tree walking helpers
# ────────────────────────────────────────────────────────────────────


def _dfs(ts_root) -> Iterator[object]:
    """DFS preorder iterator. Same order as Tier 1's _walk_tree."""
    stack = [ts_root]
    while stack:
        n = stack.pop()
        yield n
        for i in range(n.child_count - 1, -1, -1):
            stack.append(n.children[i])


def _peel_expression(ts_node):
    """Solidity wraps many constructs in `expression`. Peel single-child wrappers."""
    while ts_node is not None and ts_node.type == "expression" and ts_node.child_count == 1:
        ts_node = ts_node.children[0]
    return ts_node


def _text(ts_node) -> str:
    return ts_node.text.decode("utf-8", errors="replace")


def _parent_field_of(ts_node) -> tuple[str, str] | None:
    """Return (parent_type, field_name_for_this_child) or None."""
    parent = ts_node.parent
    if parent is None:
        return None
    for i in range(parent.child_count):
        if parent.children[i].id == ts_node.id:
            fname = parent.field_name_for_child(i)
            return (parent.type, fname) if fname else None
    return None


def _ancestor_field_chain(ts_node):
    """Yield (ancestor_type, field_name) for each step walking up to the root."""
    cur = ts_node
    while cur.parent is not None:
        parent = cur.parent
        for i in range(parent.child_count):
            if parent.children[i].id == cur.id:
                fname = parent.field_name_for_child(i)
                if fname:
                    yield (parent.type, fname)
                break
        cur = parent


def _module_name(rel_path: str) -> str:
    """Stem of the file path used as the synthetic module definition's name."""
    return PurePosixPath(rel_path).stem or rel_path


def _extract_name_from_field(ts_node, field_name: str | None) -> str | None:
    """Pull a usable name string from `ts_node.field_name` (or the node itself).

    Handles common shapes:
      - direct identifier
      - `expression -> identifier` (Solidity wrapping)
      - attribute / member_expression with a `.attribute` / `.property` leaf
      - `pattern_list` (Python tuple unpack) → return None (multi-name target)
    """
    if field_name is None:
        target = ts_node
    else:
        target = ts_node.child_by_field_name(field_name)
    target = _peel_expression(target)
    if target is None:
        return None
    if target.type in (
        "identifier",
        "type_identifier",
        "field_identifier",
        "package_identifier",
        # JS/TS: `method_definition.name` is a `property_identifier`. Same for
        # most object-literal-style member names.
        "property_identifier",
    ):
        return _text(target)
    if target.type == "attribute":
        prop = target.child_by_field_name("attribute")
        return _text(prop) if prop is not None else None
    if target.type == "member_expression":
        prop = target.child_by_field_name("property")
        return _text(prop) if prop is not None else None
    if target.type in ("pattern_list", "tuple_pattern"):
        return None  # multiple names — skip in MVP
    return None


def _terminal_identifier(ts_node) -> str | None:
    """Walk into wrapper nodes (user_defined_type, expression, attribute, …) to
    pull out the trailing identifier text. Used by inheritance extraction where
    the base name is wrapped in a type node."""
    node = _peel_expression(ts_node)
    if node is None:
        return None
    if node.type in ("identifier", "type_identifier"):
        return _text(node)
    if node.type == "user_defined_type":
        # Solidity: user_defined_type wraps the base identifier (or a dotted path).
        last_ident: str | None = None
        for c in node.children:
            if c.type in ("identifier", "type_identifier"):
                last_ident = _text(c)
        return last_ident
    if node.type == "qualified_type":
        # Go: `pkg.Foo` — the type-side identifier is what we resolve against.
        name_node = node.child_by_field_name("name")
        return _text(name_node) if name_node is not None else None
    if node.type == "type_elem":
        # Go interface embedding: a type_elem wraps either a bare
        # type_identifier (same-package embed) or a qualified_type
        # (cross-package embed). Both are unwrapped here.
        for c in node.children:
            if c.type == "type_identifier":
                return _text(c)
            if c.type == "qualified_type":
                name_node = c.child_by_field_name("name")
                return _text(name_node) if name_node is not None else None
        return None
    if node.type == "pointer_type":
        # Go embedded fields can be `*Header` — peel the pointer and recurse so
        # the inner type_identifier / qualified_type lookup applies uniformly.
        for c in node.children:
            if c.is_named:
                return _terminal_identifier(c)
        return None
    if node.type == "class_heritage":
        # JS: `class Dog extends Animal` — class_heritage holds an `extends`
        # keyword and an `identifier` directly. TS: the same shape but the
        # identifier is wrapped in `extends_clause` (with field `value`); a
        # sibling `implements_clause` may also be present and is handled by a
        # separate inheritance rule. Here we surface only the extends side.
        for c in node.children:
            if c.type == "identifier":
                return _text(c)
            if c.type == "extends_clause":
                v = c.child_by_field_name("value")
                return _text(v) if v is not None else None
        return None
    if node.type == "attribute":
        prop = node.child_by_field_name("attribute")
        return _text(prop) if prop is not None else None
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        return _text(prop) if prop is not None else None
    if node.type == "scoped_type_identifier":
        # Rust: `std::fmt::Display` — the trailing `name` field is the trait.
        name_node = node.child_by_field_name("name")
        return _text(name_node) if name_node is not None else None
    if node.type == "scoped_identifier":
        # Rust: `serde::Serialize` used in trait position via generic_type.
        name_node = node.child_by_field_name("name")
        return _text(name_node) if name_node is not None else None
    if node.type == "generic_type":
        # Rust: `From<u32>` — drop the type arguments, recurse on the base.
        base = node.child_by_field_name("type")
        return _terminal_identifier(base) if base is not None else None
    return None



def _extract_bases(ts_node, cfg) -> list[str]:
    """Return ordered list of base-class names declared on this node, per the
    inheritance config. Two shapes:

      Solidity — `parent.children[type=child_node_type]`, each with a field
      pointing at a (possibly wrapped) identifier:

        contract Foo is Bar, Baz {...}
          ↳ inheritance_specifier.ancestor → user_defined_type → identifier "Bar"
          ↳ inheritance_specifier.ancestor → user_defined_type → identifier "Baz"

      Python — `parent.bases_field` resolves to a list-like node whose
      identifier children are the base names:

        class Foo(Bar, Baz): ...
          ↳ class_definition.superclasses → argument_list → identifier "Bar", "Baz"
    """
    out: list[str] = []
    iter_node = ts_node
    if cfg.child_via_field:
        # Go interface: drill type_spec.type → interface_type before iterating
        # type_elem children. Non-interface type_specs drop out at the next
        # filter because the inner node has no type_elem children.
        intermediate = ts_node.child_by_field_name(cfg.child_via_field)
        if intermediate is None:
            return out
        iter_node = intermediate
    if cfg.child_via_node_type:
        # Go struct: after drilling .type onto a struct_type, descend into the
        # first child of type field_declaration_list. Two-step indirection
        # because field_declaration_list is the only un-named-field child of
        # struct_type and holds the field_declarations we want to iterate.
        descended = None
        for c in iter_node.children:
            if c.type == cfg.child_via_node_type:
                descended = c
                break
        if descended is None:
            return out
        iter_node = descended
    if cfg.child_node_type:
        for c in iter_node.children:
            if c.type != cfg.child_node_type:
                continue
            if cfg.child_only_when_field_absent:
                # Embedded struct fields are field_declarations whose `name`
                # field is absent; regular fields have `name` populated and
                # must be skipped.
                if c.child_by_field_name(cfg.child_only_when_field_absent) is not None:
                    continue
            if cfg.child_iterate_identifiers:
                # Multi-identifier wrapper: iterate this node's named children
                # and emit one base per child. TS: `implements_clause` carries
                # multiple type_identifiers; `extends_type_clause` ditto under
                # interface declarations.
                for inner in c.children:
                    if not inner.is_named:
                        continue
                    name = _terminal_identifier(inner)
                    if name:
                        out.append(name)
                continue
            target = c.child_by_field_name(cfg.child_name_field) if cfg.child_name_field else c
            name = _terminal_identifier(target)
            if name:
                out.append(name)
    elif cfg.bases_field:
        list_node = iter_node.child_by_field_name(cfg.bases_field)
        if list_node is not None:
            for c in list_node.children:
                name = _terminal_identifier(c)
                if name:
                    out.append(name)
    return out


# ────────────────────────────────────────────────────────────────────
# Resolver
# ────────────────────────────────────────────────────────────────────


async def _bulk_clear_branch_semantic_for_file_versions(
    conn: asyncpg.Connection,
    branch_id: int,
    file_version_ids: list[int],
) -> None:
    """Idempotency: drop any prior branch-scoped Tier 2 rows tied to these
    (branch, file_version) pairs. Six DELETEs total — one per edge table —
    regardless of how many file_versions are passed. Never touches
    definitions, nodes, or chunks (content-shared across branches).
    """
    if not file_version_ids:
        return
    await conn.execute(
        "DELETE FROM call_edges WHERE branch_id=$1 "
        "AND callsite_node_id IN (SELECT id FROM nodes WHERE file_version_id = ANY($2::bigint[]))",
        branch_id, file_version_ids,
    )
    await conn.execute(
        "DELETE FROM data_access WHERE branch_id=$1 "
        "AND accessor_def_id IN (SELECT id FROM definitions WHERE file_version_id = ANY($2::bigint[]))",
        branch_id, file_version_ids,
    )
    await conn.execute(
        'DELETE FROM "references" WHERE branch_id=$1 AND file_version_id = ANY($2::bigint[])',
        branch_id, file_version_ids,
    )
    await conn.execute(
        "DELETE FROM overrides_edges WHERE branch_id=$1 "
        "AND child_def_id IN (SELECT id FROM definitions WHERE file_version_id = ANY($2::bigint[]))",
        branch_id, file_version_ids,
    )
    await conn.execute(
        "DELETE FROM inherits_edges WHERE branch_id=$1 "
        "AND child_def_id IN (SELECT id FROM definitions WHERE file_version_id = ANY($2::bigint[]))",
        branch_id, file_version_ids,
    )
    await conn.execute(
        "DELETE FROM imports WHERE branch_id=$1 AND file_version_id = ANY($2::bigint[])",
        branch_id, file_version_ids,
    )


async def resolve_file(
    conn: asyncpg.Connection,
    branch_id: int,
    file_version_id: int,
    rel_path: str,
    config: LanguageConfig,
    *,
    node_ids: list[int] | None = None,
    pre_cleared: bool = False,
    hydrate_mode: bool | None = None,
    def_id_start: int | None = None,
) -> ResolveFileResult:
    """Run Tier-2 resolution for one (branch, file_version) pair.

    Tier-1 outputs (definitions, intra-file scope tables) are content-derived,
    so when this file_version was already resolved by another branch the
    function enters "hydrate mode": it walks the AST exactly as before but
    looks up each existing def_id from the DB instead of allocating a new
    one. Tier-2 outputs (refs, calls, data_access, intra-file inheritance)
    are always emitted, tagged with branch_id, after the per-(branch,
    file_version) clear has removed any stale rows.

    `pre_cleared`, `hydrate_mode`, and `def_id_start` let `resolve_repo` batch
    the per-file clears, hydrate checks, and sequence reservations across all
    files; when None/False the function falls back to its own per-file calls.
    """
    row = await conn.fetchrow(
        "SELECT language, raw_content FROM file_versions WHERE id=$1",
        file_version_id,
    )
    if row is None:
        raise ValueError(f"file_version_id {file_version_id} not found")
    if row["language"] != config.language:
        raise ValueError(
            f"file language={row['language']} but config language={config.language}"
        )

    source = (row["raw_content"] or "").encode("utf-8")
    parser = LANGUAGES[config.language].parser(PurePosixPath(rel_path).suffix.lower())
    tree = parser.parse(source)

    # Pair ts_nodes to DB ids (same DFS preorder as Tier 1). `node_ids` may be
    # supplied by resolve_repo's bulk prefetch to skip the per-file SELECT.
    ts_walk: list = list(_dfs(tree.root_node))
    if node_ids is None:
        db_id_rows = await conn.fetch(
            "SELECT id FROM nodes WHERE file_version_id=$1 ORDER BY id",
            file_version_id,
        )
        node_ids = [r["id"] for r in db_id_rows]
    if len(ts_walk) != len(node_ids):
        raise RuntimeError(
            f"CST size mismatch for file_version_id={file_version_id}: "
            f"reparse produced {len(ts_walk)} nodes, DB has {len(node_ids)}"
        )
    db_id_for: dict[int, int] = {ts.id: node_ids[i] for i, ts in enumerate(ts_walk)}

    if not pre_cleared:
        await _bulk_clear_branch_semantic_for_file_versions(
            conn, branch_id, [file_version_id],
        )

    # Per-language hooks: whole-file skip (Rust uses this for `tests/`,
    # `benches/`, `examples/` dirs), plus per-file precompute that stashes
    # scratch state used by later emission loops (Rust's test-gated node IDs).
    handler = get_handler(config.language)
    if handler.should_skip_file(rel_path):
        return ResolveFileResult(
            file_version_id=file_version_id, language=config.language,
            n_definitions=0, n_references=0,
            n_call_edges=0, n_data_access=0,
        )

    sem_ctx = SemanticContext(config=config, file_path=rel_path)
    handler.precompute_file_state(ts_walk, sem_ctx)
    test_skip_ts_ids: frozenset[int] = sem_ctx.scratch.get(
        "test_skip_ts_ids", frozenset(),
    )

    # Hydrate mode: when another branch already resolved this file_version,
    # definitions and nodes are already in the DB. We rebuild the in-memory
    # scope tables from the existing rows instead of inserting new ones.
    if hydrate_mode is None:
        hydrate_mode = bool(await conn.fetchval(
            "SELECT 1 FROM definitions WHERE file_version_id=$1 LIMIT 1",
            file_version_id,
        ))

    # ── P1: definitions (synthetic module + matched rules) ──
    def_rules = {r.node_type: r for r in config.definitions}

    # In-memory scope tables — populated below either by walking + emitting
    # (cold path) or by querying the DB (hydrate path).
    pending_defs: list[tuple[int, int, int, str, str, str, int | None, str | None]] = []
    scope_def_id_by_ts: dict[int, int] = {}
    def_kind_by_id: dict[int, str] = {}
    def_meta_by_id: dict[int, tuple[str, int | None]] = {}
    defs_by_scope_and_name: dict[tuple[int, str], int] = {}
    file_def_ids: list[int] = []
    module_def_id: int

    if hydrate_mode:
        # Load existing defs and reconstruct in-memory state.
        existing_defs = await conn.fetch(
            """
            SELECT id, node_id, name, qualified_name, kind, scope_id
            FROM definitions
            WHERE file_version_id=$1
            """,
            file_version_id,
        )
        # Reverse map: db_node_id → ts.id, used to populate scope_def_id_by_ts.
        ts_id_for_db = {db_id: ts_id for ts_id, db_id in db_id_for.items()}
        # Scope-ownership is a property of the def's *rule* (`scope_boundary`)
        # plus the module def itself, NOT a property of whether the def
        # happens to have children referenced as scope_id. We derive it from
        # the language config — the same source of truth P1 uses on the cold
        # path. Without this, function-kind defs with no nested children
        # would be missing from scope_def_id_by_ts and P5's enclosing-scope
        # walk would attribute calls to the enclosing class/module instead of
        # the function.
        scope_boundary_kinds = {r.kind for r in config.definitions if r.scope_boundary}
        for d in existing_defs:
            if d["kind"] == "module":
                module_def_id = d["id"]
        for d in existing_defs:
            def_kind_by_id[d["id"]] = d["kind"]
            def_meta_by_id[d["id"]] = (d["name"], d["scope_id"])
            if d["scope_id"] is not None:
                defs_by_scope_and_name[(d["scope_id"], d["name"])] = d["id"]
            file_def_ids.append(d["id"])
            is_scope_owner = d["kind"] == "module" or d["kind"] in scope_boundary_kinds
            if is_scope_owner:
                ts_id = ts_id_for_db.get(d["node_id"])
                if ts_id is not None:
                    scope_def_id_by_ts[ts_id] = d["id"]
    else:
        # Cold path: walk + emit.
        #
        # Performance: we reserve a contiguous block of definition IDs from the
        # sequence and PREDICT each new def's id as we walk, instead of doing
        # `INSERT … RETURNING id` per row. All in-memory lookup tables
        # (scope_def_id_by_ts, def_meta_by_id, etc.) are populated with the
        # predicted ids, which are guaranteed to match the rows we batch-insert
        # at the end. A safe upper bound on the count is `len(ts_walk) + 1`
        # (one def per ts_node plus the synthetic module). Wasted ids inside the
        # reserved block become harmless sequence gaps.
        if def_id_start is not None:
            reserved_first = def_id_start
        else:
            reserved_first = await reserve_definition_ids(conn, len(ts_walk) + 1)
        next_def_id = reserved_first

        # Module / source_file root definition.
        module_db_node_id = db_id_for[tree.root_node.id]
        module_name = _module_name(rel_path)
        module_def_id = next_def_id
        next_def_id += 1

        pending_defs.append(
            (module_def_id, module_db_node_id, file_version_id, "module", module_name, module_name, None, None),
        )
        scope_def_id_by_ts[tree.root_node.id] = module_def_id
        def_kind_by_id[module_def_id] = "module"
        def_meta_by_id[module_def_id] = (module_name, None)
        file_def_ids.append(module_def_id)

        def _enclosing_scope_def_id_cold(ts_node) -> int | None:
            cur = ts_node.parent
            while cur is not None:
                db_did = scope_def_id_by_ts.get(cur.id)
                if db_did is not None:
                    return db_did
                cur = cur.parent
            return scope_def_id_by_ts.get(tree.root_node.id)

        # Walk in DFS preorder; an enclosing scope's def_id is always set before
        # children are processed, so scope_id resolution is straightforward.
        for ts in ts_walk:
            if ts.id in test_skip_ts_ids:
                continue
            rule = def_rules.get(ts.type)
            if rule is None:
                continue
            scope_id = _enclosing_scope_def_id_cold(ts)
            if rule.require_enclosing_scope_kind:
                allowed = set(rule.require_enclosing_scope_kind)
                scope_kind = def_kind_by_id.get(scope_id) if scope_id is not None else None
                if scope_kind not in allowed:
                    continue

            # One def node usually emits one definition row, but Go's `var a, b int`
            # / `const x, y = …` is a single var_spec/const_spec with multiple
            # identifier children in the `name` field. `definitions.node_id` is
            # UNIQUE, so each name gets its own identifier-child node_id rather
            # than reusing the spec's.
            named_targets: list[tuple[str, int]] = []  # (name, db_node_id)
            if rule.name_field_multiple and rule.name_field is not None:
                for i in range(ts.child_count):
                    if ts.field_name_for_child(i) != rule.name_field:
                        continue
                    child = ts.children[i]
                    if child.type in ("identifier", "type_identifier", "field_identifier"):
                        named_targets.append((_text(child), db_id_for[child.id]))
            else:
                if rule.name_field is not None or ts.type == "constructor_definition":
                    name = _extract_name_from_field(ts, rule.name_field)
                else:
                    name = None
                if name is None and rule.kind == "constructor":
                    # Constructors carry their contract's name implicitly.
                    owner = def_meta_by_id.get(scope_id) if scope_id is not None else None
                    name = owner[0] if owner else "constructor"
                if name is not None:
                    named_targets.append((name, db_id_for[ts.id]))
            if not named_targets:
                continue

            # qualified_name prefix from a field on the def node — Go method
            # receivers: `func (d *Dog) Bark()` → segment `Dog` inserted before
            # the method name so qualified_name becomes `<file>.Dog.Bark`.
            prefix_segment: str | None = None
            if rule.qualified_name_prefix_from_field:
                field_node = ts.child_by_field_name(rule.qualified_name_prefix_from_field)
                if field_node is not None:
                    stack = [field_node]
                    while stack:
                        n = stack.pop()
                        if n.type == "type_identifier":
                            prefix_segment = _text(n)
                            break
                        for i in range(n.child_count - 1, -1, -1):
                            stack.append(n.children[i])

            # Per-language hook: language-specific qualified-name prefix
            # segment that the YAML config can't express. Rust uses this to
            # prepend an impl block's target type to method names
            # (`Counter::new` → qualified_name includes 'Counter').
            if prefix_segment is None:
                prefix_segment = handler.qualified_name_prefix(ts, sem_ctx)

            visibility: str | None = None
            if rule.visibility_field:
                vnode = ts.child_by_field_name(rule.visibility_field)
                if vnode is None:
                    for child in ts.children:
                        if child.type == rule.visibility_field:
                            vnode = child
                            break
                if vnode is not None:
                    visibility = _text(vnode)

            for name, name_node_id in named_targets:
                # Build qualified name by walking up scope chain via def_meta_by_id.
                parts = [name]
                if prefix_segment is not None:
                    parts.append(prefix_segment)
                cur = scope_id
                while cur is not None:
                    parent_name, parent_scope = def_meta_by_id[cur]
                    parts.append(parent_name)
                    cur = parent_scope
                qualified_name = ".".join(reversed(parts))

                new_def_id = next_def_id
                next_def_id += 1
                pending_defs.append((
                    new_def_id, name_node_id, file_version_id, rule.kind, name, qualified_name,
                    scope_id, visibility,
                ))
                def_kind_by_id[new_def_id] = rule.kind
                def_meta_by_id[new_def_id] = (name, scope_id)
                if scope_id is not None:
                    defs_by_scope_and_name[(scope_id, name)] = new_def_id
                if rule.scope_boundary:
                    # Multi-name + scope_boundary doesn't make sense; `var_spec` and
                    # `const_spec` have scope_boundary=False so this maps cleanly to
                    # the single-name case where ts.id is the spec node.
                    scope_def_id_by_ts[ts.id] = new_def_id
                file_def_ids.append(new_def_id)

        # Flush all definitions for this file in a single round-trip. The UNNEST
        # arrays must align with the column list and tuple shape used above.
        if pending_defs:
            await conn.execute(
                """
                INSERT INTO definitions
                    (id, node_id, file_version_id, kind, name, qualified_name, scope_id, visibility)
                SELECT * FROM UNNEST(
                    $1::bigint[], $2::bigint[], $3::bigint[], $4::text[],
                    $5::text[],   $6::text[],   $7::bigint[], $8::text[]
                )
                """,
                [d[0] for d in pending_defs],
                [d[1] for d in pending_defs],
                [d[2] for d in pending_defs],
                [d[3] for d in pending_defs],
                [d[4] for d in pending_defs],
                [d[5] for d in pending_defs],
                [d[6] for d in pending_defs],
                [d[7] for d in pending_defs],
            )

    def _enclosing_scope_def_id(ts_node) -> int | None:
        cur = ts_node.parent
        while cur is not None:
            db_did = scope_def_id_by_ts.get(cur.id)
            if db_did is not None:
                return db_did
            cur = cur.parent
        return scope_def_id_by_ts.get(tree.root_node.id)

    # ── P2.5: inheritance edges (intra-file resolution; per-branch row) ──
    # `config.inheritance` is a tuple of rules. A single parent node may match
    # more than one rule (Go: `type_spec` is the parent for both interface
    # embedding and struct embedding). Bases from each rule are concatenated
    # in declaration order so the `ord` column reflects a stable ranking.
    # (child, base_name, ord, base_def_id, confidence)
    inh_records: list[tuple[int, str, int, int | None, str]] = []
    for ts in ts_walk:
        if ts.id in test_skip_ts_ids:
            continue
        child_def_id: int | None = None
        ordinal = 0
        for rule in config.inheritance:
            if ts.type not in rule.parent_node_types:
                continue
            if child_def_id is None:
                child_def_id = scope_def_id_by_ts.get(ts.id)
                if child_def_id is None:
                    break
            for base_name in _extract_bases(ts, rule):
                ordinal += 1
                # Try intra-file resolution: does any module-scope def in this file
                # match the base name? (Cross-file matches go through Phase 3.)
                base_def_id = defs_by_scope_and_name.get((module_def_id, base_name))
                inh_records.append((child_def_id, base_name, ordinal, base_def_id, "certain"))

    # ── P2.5b: handler-synthesized inheritance edges ──
    # Per-language hook for shapes the YAML inheritance machinery can't
    # express (Rust: `impl Trait for Type` and `#[derive(...)]` macro edges).
    # Ordinals are per-target so multiple synthesized edges sharing a target
    # def_id get stable, unique `ord` values.
    sem_ctx.module_def_id = module_def_id
    sem_ctx.defs_by_scope_and_name = defs_by_scope_and_name
    synthesized = handler.synthesize_inheritance(ts_walk, sem_ctx)
    if synthesized:
        ord_for: dict[int, int] = {}
        for edge in synthesized:
            n = ord_for.get(edge.child_def_id, 0) + 1
            ord_for[edge.child_def_id] = n
            inh_records.append(
                (edge.child_def_id, edge.base_name, n, edge.base_def_id, edge.confidence)
            )

    if inh_records:
        await conn.executemany(
            """
            INSERT INTO inherits_edges (branch_id, child_def_id, base_name, ord, base_def_id, confidence)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [(branch_id, *t) for t in inh_records],
        )

    # ── P3 + P5 + P6: references, calls, data_access ──
    ref_rules = list(config.references)
    call_rules = list(config.calls)
    data_access_cfg = config.data_access

    # Collect into batches.
    ref_records: list[_RefRecord] = []
    # Track ref_record index → ts_node, so P6 can re-examine it for write/read classification.
    ref_ts_for: list[object] = []

    def _resolve_in_scope(name: str, from_ts) -> int | None:
        cur = from_ts.parent
        while cur is not None:
            scope_did = scope_def_id_by_ts.get(cur.id)
            if scope_did is not None:
                hit = defs_by_scope_and_name.get((scope_did, name))
                if hit is not None:
                    return hit
            cur = cur.parent
        # Top scope = module.
        return defs_by_scope_and_name.get((module_def_id, name))

    # Quick lookup: any def-by-name in this file (used for inferred-confidence resolution).
    file_name_index: dict[str, list[int]] = {}
    for did in file_def_ids:
        meta = def_meta_by_id.get(did)
        if meta:
            file_name_index.setdefault(meta[0], []).append(did)

    # Pre-compute the set of "exclude_parent_field" predicates per ref rule.
    def _ref_excluded(ts_node, rule) -> bool:
        if not rule.exclude_parent_field:
            return False
        pf = _parent_field_of(ts_node)
        if pf is None:
            return False
        key = f"{pf[0]}.{pf[1]}"
        return key in rule.exclude_parent_field

    ref_rule_by_type = {r.node_type: r for r in ref_rules}
    call_rule_by_type = {r.node_type: r for r in call_rules}

    # Identifiers / attributes / member_expressions also serve as the function part of a
    # call. We let the reference rule fire on them — the call rule below additionally
    # creates a call_edge.
    for ts in ts_walk:
        if ts.id in test_skip_ts_ids:
            continue
        rule = ref_rule_by_type.get(ts.type)
        if rule is None:
            continue
        if _ref_excluded(ts, rule):
            continue
        name = _extract_name_from_field(ts, rule.name_field)
        if not name:
            continue
        # Handler gets first shot via the optional `resolve_reference` hook
        # so it can override the YAML rule's resolution + confidence when
        # AST context gives a better answer than name-only lookup. Fall
        # through to the rule's declared strategy if the hook abstains.
        confidence = rule.confidence
        target: int | None = None
        handler_hit = handler.resolve_reference(ts, rule, sem_ctx)
        if handler_hit is not None:
            target = handler_hit.target_def_id
            confidence = handler_hit.confidence
        elif rule.confidence == "inferred":
            cands = file_name_index.get(name, [])
            target = cands[0] if len(cands) == 1 else None
        else:
            target = _resolve_in_scope(name, ts)
        ref_records.append(
            _RefRecord(
                db_node_id=db_id_for[ts.id],
                file_version_id=file_version_id,
                name=name,
                confidence=confidence,
                target_def_id=target,
            )
        )
        ref_ts_for.append(ts)

    # Insert references; we need ids back for data_access rows.
    if ref_records:
        await conn.execute(
            """
            INSERT INTO "references" (branch_id, node_id, file_version_id, name, target_def_id, resolution_confidence)
            SELECT $1, * FROM UNNEST($2::bigint[], $3::bigint[], $4::text[], $5::bigint[], $6::float[])
            """,
            branch_id,
            [r.db_node_id for r in ref_records],
            [r.file_version_id for r in ref_records],
            [r.name for r in ref_records],
            [r.target_def_id for r in ref_records],
            [_confidence_score(r.confidence) for r in ref_records],
        )

    # ── P5: call edges ──
    call_records: list[_CallRecord] = []
    for ts in ts_walk:
        if ts.id in test_skip_ts_ids:
            continue
        rule = call_rule_by_type.get(ts.type)
        if rule is None:
            continue
        fexpr = ts.child_by_field_name(rule.function_field)
        fexpr = _peel_expression(fexpr)
        if fexpr is None:
            continue

        callee_name: str | None = None
        if fexpr.type in ("identifier", "type_identifier"):
            callee_name = _text(fexpr)
            target = _resolve_in_scope(callee_name, ts)
            confidence = "certain" if target is not None else "uncertain"
        elif fexpr.type in ("attribute", "member_expression", "selector_expression"):
            # Python: attribute.attribute. Solidity: member_expression.property.
            # Go: selector_expression.field. All three carry the right-hand
            # callee identifier in a node-type-specific field name.
            prop_field = {
                "attribute": "attribute",
                "member_expression": "property",
                "selector_expression": "field",
            }[fexpr.type]
            prop = fexpr.child_by_field_name(prop_field)
            callee_name = _text(prop) if prop is not None else None
            cands = file_name_index.get(callee_name, []) if callee_name else []
            target = cands[0] if len(cands) == 1 else None
            confidence = "inferred" if target is not None else "uncertain"
        else:
            target = None
            confidence = "uncertain"

        caller_def_id = _enclosing_scope_def_id(ts)
        call_records.append(
            _CallRecord(
                callsite_db_node_id=db_id_for[ts.id],
                caller_def_id=caller_def_id,
                callee_def_id=target,
                callee_name=callee_name,
                confidence=confidence,
            )
        )

    if call_records:
        await conn.execute(
            """
            INSERT INTO call_edges (branch_id, callsite_node_id, caller_def_id, callee_def_id, callee_name, confidence)
            SELECT $1, * FROM UNNEST(
                $2::bigint[], $3::bigint[], $4::bigint[], $5::text[], $6::text[]
            )
            """,
            branch_id,
            [r.callsite_db_node_id for r in call_records],
            [r.caller_def_id for r in call_records],
            [r.callee_def_id for r in call_records],
            [r.callee_name for r in call_records],
            [r.confidence for r in call_records],
        )

    # ── P6: data access ──
    da_records: list[_DataAccessRecord] = []
    if data_access_cfg and data_access_cfg.target_kinds:
        target_kinds = set(data_access_cfg.target_kinds)
        write_keys = set(data_access_cfg.write_when_parent_field)
        for ref, ref_ts in zip(ref_records, ref_ts_for):
            if ref.target_def_id is None:
                continue
            kind = def_kind_by_id.get(ref.target_def_id)
            if kind is None or kind not in target_kinds:
                continue
            accessor = _enclosing_scope_def_id(ref_ts)
            if accessor is None:
                continue
            access = "read"
            for atype, fname in _ancestor_field_chain(ref_ts):
                if f"{atype}.{fname}" in write_keys:
                    access = "write"
                    break
            da_records.append(
                _DataAccessRecord(
                    accessor_def_id=accessor,
                    target_def_id=ref.target_def_id,
                    access_type=access,
                    db_node_id=ref.db_node_id,
                )
            )

    if da_records:
        await conn.execute(
            """
            INSERT INTO data_access (branch_id, accessor_def_id, target_def_id, access_type, node_id)
            SELECT $1, * FROM UNNEST(
                $2::bigint[], $3::bigint[], $4::text[], $5::bigint[]
            )
            """,
            branch_id,
            [r.accessor_def_id for r in da_records],
            [r.target_def_id for r in da_records],
            [r.access_type for r in da_records],
            [r.db_node_id for r in da_records],
        )

    return ResolveFileResult(
        file_version_id=file_version_id,
        language=config.language,
        n_definitions=len(file_def_ids),
        n_references=len(ref_records),
        n_call_edges=len(call_records),
        n_data_access=len(da_records),
        n_inherits_edges=len(inh_records),
    )


def _confidence_score(level: str) -> float:
    return {"certain": 1.0, "inferred": 0.7, "uncertain": 0.4}.get(level, 1.0)


# ────────────────────────────────────────────────────────────────────
# Repo-level driver
# ────────────────────────────────────────────────────────────────────


async def resolve_repo(
    pool: asyncpg.Pool,
    repo_id: int,
    branch_id: int,
    *,
    only_file_version_ids: list[int] | None = None,
) -> list[ResolveFileResult]:
    """Run the resolver on every (path, file_version) mapped by branch_id
    whose language has a YAML config.

    `only_file_version_ids` restricts to a subset (used by the indexer to skip
    files whose CST didn't change).
    """
    async with pool.acquire() as conn:
        # DISTINCT ON (fv.id): two branch_files paths can share one file_version
        # (identical content). Resolution is per file_version, so collapse here
        # or concurrent _resolve_one tasks race and collide on definitions.node_id.
        if only_file_version_ids is None:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (fv.id)
                    bf.path, fv.id AS file_version_id, fv.language
                FROM branch_files bf
                JOIN file_versions fv ON fv.id = bf.file_version_id
                WHERE bf.branch_id = $1
                ORDER BY fv.id, bf.path
                """,
                branch_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (fv.id)
                    bf.path, fv.id AS file_version_id, fv.language
                FROM branch_files bf
                JOIN file_versions fv ON fv.id = bf.file_version_id
                WHERE bf.branch_id = $1 AND fv.id = ANY($2::bigint[])
                ORDER BY fv.id, bf.path
                """,
                branch_id,
                only_file_version_ids,
            )

    # Up-front batches: node id prefetch, repo-wide clear, hydrate-mode set,
    # and one definition-id reservation for all cold files. Each collapses N
    # per-file round-trips into 1 and (for the reservation) eliminates
    # advisory-lock contention between concurrent _resolve_one tasks.
    all_fv_ids = [row["file_version_id"] for row in rows]
    nodes_by_fv: dict[int, list[int]] = {}
    hydrate_set: set[int] = set()
    def_id_start_by_fv: dict[int, int] = {}
    async with pool.acquire() as conn:
        if all_fv_ids:
            node_rows = await conn.fetch(
                "SELECT file_version_id, id FROM nodes "
                "WHERE file_version_id = ANY($1::bigint[]) "
                "ORDER BY file_version_id, id",
                all_fv_ids,
            )
            for r in node_rows:
                nodes_by_fv.setdefault(r["file_version_id"], []).append(r["id"])

            await _bulk_clear_branch_semantic_for_file_versions(
                conn, branch_id, all_fv_ids,
            )

            hydrate_rows = await conn.fetch(
                "SELECT DISTINCT file_version_id FROM definitions "
                "WHERE file_version_id = ANY($1::bigint[])",
                all_fv_ids,
            )
            hydrate_set = {r["file_version_id"] for r in hydrate_rows}

            cold_fvs = [fv for fv in all_fv_ids if fv not in hydrate_set]
            if cold_fvs:
                bounds = [len(nodes_by_fv.get(fv, [])) + 1 for fv in cold_fvs]
                first = await reserve_definition_ids(conn, sum(bounds))
                cursor = first
                for fv, b in zip(cold_fvs, bounds):
                    def_id_start_by_fv[fv] = cursor
                    cursor += b

    # Load every needed language config up front so the per-file tasks can
    # read from `configs` without racing on lazy population.
    configs: dict[str, LanguageConfig] = {}
    for lang in {row["language"] for row in rows}:
        try:
            configs[lang] = load_language_config(lang)
        except FileNotFoundError:
            pass

    sem = asyncio.Semaphore(int(os.environ.get("RESOLVE_CONCURRENCY", "4")))

    async def _resolve_one(row) -> ResolveFileResult | None:
        cfg = configs.get(row["language"])
        if cfg is None:
            return None
        fv_id = row["file_version_id"]
        async with sem, pool.acquire() as conn, conn.transaction():
            return await resolve_file(
                conn, branch_id, fv_id, row["path"], cfg,
                node_ids=nodes_by_fv.get(fv_id, []),
                pre_cleared=True,
                hydrate_mode=fv_id in hydrate_set,
                def_id_start=def_id_start_by_fv.get(fv_id),
            )

    gathered = await asyncio.gather(*[_resolve_one(row) for row in rows])
    return [r for r in gathered if r is not None]


def resolve_repo_sync(
    repo_id: int, branch_id: int, dsn: str | None = None,
) -> list[ResolveFileResult]:
    from db.connection import pool_ctx

    async def _run():
        async with pool_ctx(dsn) as pool:
            return await resolve_repo(pool, repo_id, branch_id)

    return asyncio.run(_run())
