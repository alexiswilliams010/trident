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
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator

import asyncpg

from db.connection import reserve_definition_ids

from .config_loader import LanguageConfig, load_language_config
from .grammar_meta import LANGUAGES


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


_RUST_TEST_PATH_SEGMENTS = frozenset({"tests", "benches", "examples", "test_utils"})


def _is_rust_test_path(rel_path: str) -> bool:
    """True if a Rust source file lives in a directory we treat as test-only
    or test-scaffolding code:
      • cargo's separate-compilation-unit dirs: `tests/`, `benches/`,
        `examples/` — these never link into the production crate.
      • `test_utils/` — Solana/Anchor convention for cross-crate test
        fixtures. Not gated by `#[cfg(test)]` (so other crates' integration
        tests can pull them in) but functionally test scaffolding —
        retrieval-noise for production code questions.
    Detected by path segment, catching both top-level and per-crate
    workspace layouts (`programs/bridge/src/test_utils/mod.rs`)."""
    return any(p in _RUST_TEST_PATH_SEGMENTS for p in rel_path.split("/"))


def _rust_attribute_path_name(attr_ts) -> str | None:
    """Return the dotted-path head of an `attribute` node (the bit before
    the optional `(...)` arguments). For `#[cfg(test)]` returns `"cfg"`;
    for `#[serde(rename_all = "snake_case")]` returns `"serde"`; for
    `#[tokio::test]` returns `"test"` (the trailing identifier)."""
    for c in attr_ts.children:
        if c.type == "identifier":
            return _text(c)
        if c.type == "scoped_identifier":
            return _terminal_identifier(c)
    return None


def _token_tree_contains_test(args_ts) -> bool:
    """True if the `cfg(...)` token-tree mentions a literal `test` token
    anywhere in its named subtree. We don't try to interpret arbitrary cfg
    predicates — `cfg(any(test, feature = "x"))` and `cfg(all(test, …))`
    both correctly trip this. False positives are rare; the cost of
    occasionally over-skipping borderline test-utility code is far less
    than the cost of letting unit-test bodies dominate retrieval."""
    stack = [args_ts]
    while stack:
        n = stack.pop()
        if n.type == "identifier" and _text(n) == "test":
            return True
        for c in n.children:
            stack.append(c)
    return False


def _has_test_attribute(item_ts) -> bool:
    """Walk previous-sibling attribute_items looking for a test marker:
      • `#[test]` / `#[tokio::test]` / `#[<runner>::test]`
      • `#[cfg(test)]` / `#[cfg(any(test, …))]` / `#[cfg(all(test, …))]`
    Returns True on the first match."""
    cursor = item_ts.prev_named_sibling
    while cursor is not None and cursor.type == "attribute_item":
        attr = next((c for c in cursor.children if c.type == "attribute"), None)
        if attr is not None:
            path_name = _rust_attribute_path_name(attr)
            if path_name == "test":
                return True
            if path_name == "cfg":
                args = attr.child_by_field_name("arguments")
                if args is not None and _token_tree_contains_test(args):
                    return True
        cursor = cursor.prev_named_sibling
    return False


_RUST_SKIP_ITEM_TYPES = frozenset({
    "mod_item", "function_item", "impl_item",
    "struct_item", "enum_item", "union_item",
    "trait_item", "type_item", "const_item", "static_item",
    "macro_definition",
})

# Module names that, by convention, hold test scaffolding even when not
# `#[cfg(test)]`-gated. `tests` is the unit-test convention (almost always
# paired with cfg(test) but not strictly required). `test_utils` is the
# Solana/Anchor / cross-crate helper convention — fixtures live here so
# integration tests in other crates can pull them in, which means they
# can't be cfg(test)-gated. Both are noise for code retrieval.
_RUST_TEST_MOD_NAMES = frozenset({"tests", "test_utils"})


def _collect_rust_test_skip_ids(ts_walk) -> frozenset[int]:
    """Return the ts_node ids whose subtrees should be skipped during
    semantic emission because they are test code. A node is a skip-root
    when one of:
      • it carries a test attribute (#[test], #[cfg(test)], etc.)
      • it's a `mod_item` literally named `tests` (idiomatic test-mod
        convention; almost always paired with `#[cfg(test)]` but not
        strictly required by the language)

    The skip set includes all descendants of every skip-root, so any def
    rule matching anything inside `mod tests { … }` is filtered out before
    a definition row is reserved."""
    skip_roots: list = []
    for ts in ts_walk:
        if ts.type not in _RUST_SKIP_ITEM_TYPES:
            continue
        is_test = _has_test_attribute(ts)
        if not is_test and ts.type == "mod_item":
            name_node = ts.child_by_field_name("name")
            if name_node is not None and _text(name_node) in _RUST_TEST_MOD_NAMES:
                is_test = True
        if is_test:
            skip_roots.append(ts)
    if not skip_roots:
        return frozenset()
    out: set[int] = set()
    for root in skip_roots:
        stack = [root]
        while stack:
            n = stack.pop()
            out.add(n.id)
            for c in n.children:
                stack.append(c)
    return frozenset(out)


def _collect_rust_derives(item_ts) -> list[str]:
    """Walk an item's previous siblings to collect names from `#[derive(...)]`
    attributes. tree-sitter-rust represents attributes as siblings, not
    children, of the item they decorate.

    Stops at the first non-attribute_item sibling so a comment-or-blank-line
    gap between an attribute and the item still works (whitespace isn't a
    named child). `#[derive(Foo, Bar)]` and chained `#[derive(Foo)]
    #[derive(Bar)]` both yield `[Foo, Bar]`. Any non-derive attribute
    (`#[serde(rename = "...")]`, `#[cfg(test)]`, …) is skipped — only the
    derive list contributes graph edges.
    """
    out: list[str] = []
    cursor = item_ts.prev_named_sibling
    while cursor is not None and cursor.type == "attribute_item":
        attr = None
        for c in cursor.children:
            if c.type == "attribute":
                attr = c
                break
        if attr is None:
            cursor = cursor.prev_named_sibling
            continue
        # First named child of `attribute` is the path (`derive`, `cfg`, …).
        path_name: str | None = None
        for c in attr.children:
            if c.type == "identifier":
                path_name = _text(c)
                break
            if c.type == "scoped_identifier":
                path_name = _terminal_identifier(c)
                break
        if path_name == "derive":
            args = attr.child_by_field_name("arguments")
            if args is not None:
                # token_tree carries the parenthesized list; we collect every
                # identifier / scoped_identifier child as a derived trait. The
                # commas and parens are anonymous tokens and are skipped by
                # is_named. We prepend each attribute's derives so the result
                # reads in source order despite the prev-sibling walk.
                this_attr: list[str] = []
                for c in args.children:
                    if not c.is_named:
                        continue
                    if c.type in ("identifier", "type_identifier"):
                        this_attr.append(_text(c))
                    elif c.type == "scoped_identifier":
                        n = _terminal_identifier(c)
                        if n:
                            this_attr.append(n)
                out = this_attr + out
        cursor = cursor.prev_named_sibling
    return out


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


async def _clear_branch_semantic_for_file_version(
    conn: asyncpg.Connection,
    branch_id: int,
    file_version_id: int,
) -> None:
    """Idempotency: drop any prior branch-scoped Tier 2 rows tied to this
    (branch, file_version). Critically, this NEVER deletes definitions,
    nodes, chunks, or any content-derived table — those are shared across
    branches and only `make gc` reclaims them when truly unreferenced.

    Uses the simple-query protocol (no parameters) so all six statements
    run in order. branch_id and file_version_id are interpolated as ints.
    """
    bid = int(branch_id)
    fvid = int(file_version_id)
    await conn.execute(f"""
        DELETE FROM call_edges
          WHERE branch_id={bid}
            AND callsite_node_id IN (SELECT id FROM nodes WHERE file_version_id={fvid});
        DELETE FROM data_access
          WHERE branch_id={bid}
            AND accessor_def_id IN (SELECT id FROM definitions WHERE file_version_id={fvid});
        DELETE FROM "references" WHERE branch_id={bid} AND file_version_id={fvid};
        DELETE FROM overrides_edges
          WHERE branch_id={bid}
            AND child_def_id IN (SELECT id FROM definitions WHERE file_version_id={fvid});
        DELETE FROM inherits_edges
          WHERE branch_id={bid}
            AND child_def_id IN (SELECT id FROM definitions WHERE file_version_id={fvid});
        DELETE FROM imports WHERE branch_id={bid} AND file_version_id={fvid};
    """)


async def resolve_file(
    conn: asyncpg.Connection,
    branch_id: int,
    file_version_id: int,
    rel_path: str,
    config: LanguageConfig,
) -> ResolveFileResult:
    """Run Tier-2 resolution for one (branch, file_version) pair.

    Tier-1 outputs (definitions, intra-file scope tables) are content-derived,
    so when this file_version was already resolved by another branch the
    function enters "hydrate mode": it walks the AST exactly as before but
    looks up each existing def_id from the DB instead of allocating a new
    one. Tier-2 outputs (refs, calls, data_access, intra-file inheritance)
    are always emitted, tagged with branch_id, after the per-(branch,
    file_version) clear has removed any stale rows.
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

    # Pair ts_nodes to DB ids (same DFS preorder as Tier 1).
    ts_walk: list = list(_dfs(tree.root_node))
    db_ids = await conn.fetch(
        "SELECT id FROM nodes WHERE file_version_id=$1 ORDER BY id",
        file_version_id,
    )
    if len(ts_walk) != len(db_ids):
        raise RuntimeError(
            f"CST size mismatch for file_version_id={file_version_id}: "
            f"reparse produced {len(ts_walk)} nodes, DB has {len(db_ids)}"
        )
    db_id_for: dict[int, int] = {ts.id: db_ids[i]["id"] for i, ts in enumerate(ts_walk)}

    await _clear_branch_semantic_for_file_version(conn, branch_id, file_version_id)

    # Rust-only: drop integration-test files (`tests/`, `benches/`,
    # `examples/`) from semantic emission entirely — these are separate
    # compilation units and their bodies are noise for retrieval. Inline
    # `#[cfg(test)] mod tests { … }` blocks are filtered per-node below.
    if config.language == "rust" and _is_rust_test_path(rel_path):
        return ResolveFileResult(
            file_version_id=file_version_id, language=config.language,
            n_definitions=0, n_references=0,
            n_call_edges=0, n_data_access=0,
        )

    # Rust-only: pre-compute the set of ts_node ids inside test-gated items
    # (cfg(test)/test attribute or `mod tests { … }`). Every emission loop
    # below treats these as if they didn't exist.
    test_skip_ts_ids: frozenset[int] = (
        _collect_rust_test_skip_ids(ts_walk)
        if config.language == "rust"
        else frozenset()
    )

    # Hydrate mode: when another branch already resolved this file_version,
    # definitions and nodes are already in the DB. We rebuild the in-memory
    # scope tables from the existing rows instead of inserting new ones.
    hydrate_mode = await conn.fetchval(
        "SELECT 1 FROM definitions WHERE file_version_id=$1 LIMIT 1",
        file_version_id,
    )
    hydrate_mode = bool(hydrate_mode)

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

            # Rust-specific: a `function_item` whose AST grandparent is an
            # `impl_item` is a method; prefix its qualified_name with the impl's
            # target type (`Counter::new`). impl_item is intentionally not a
            # definition of its own — see configs/rust.yaml — so the prefix has
            # to come from the AST, not the scope chain.
            if (
                config.language == "rust"
                and prefix_segment is None
                and ts.type == "function_item"
            ):
                parent = ts.parent
                if parent is not None and parent.type == "declaration_list":
                    gp = parent.parent
                    if gp is not None and gp.type == "impl_item":
                        type_field = gp.child_by_field_name("type")
                        if type_field is not None:
                            prefix_segment = _terminal_identifier(type_field)

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

    # ── P2.5b: Rust-specific inheritance (impl-trait + derive macros) ──
    # Two synthetic shapes the YAML inheritance machinery can't express:
    #   • `impl Trait for Type { … }` → edge Type → Trait. The child here is
    #     the Type's existing struct/enum/union/type def, not the impl_item
    #     (which is intentionally not a definition).
    #   • `#[derive(Trait1, Trait2)]` on a struct/enum/union → one edge per
    #     derived trait. Tree-sitter never expands the macro, so the actual
    #     `impl Trait for Type { ... }` block rust-analyzer would see is
    #     invisible to us; we synthesize the edges best-effort. Confidence
    #     `inferred` reflects the heuristic nature of both shapes.
    if config.language == "rust":
        ord_for: dict[int, int] = {}

        def _next_ord(target: int) -> int:
            n = ord_for.get(target, 0) + 1
            ord_for[target] = n
            return n

        # impl Trait for Type → Type's def → Trait
        for ts in ts_walk:
            if ts.id in test_skip_ts_ids:
                continue
            if ts.type != "impl_item":
                continue
            type_field = ts.child_by_field_name("type")
            trait_field = ts.child_by_field_name("trait")
            if type_field is None or trait_field is None:
                continue
            target_name = _terminal_identifier(type_field)
            trait_name = _terminal_identifier(trait_field)
            if not target_name or not trait_name:
                continue
            target_def = defs_by_scope_and_name.get((module_def_id, target_name))
            if target_def is None:
                # Type defined in another file — Phase 3 will not currently
                # link this since inherits cross-file resolution keys on
                # child_def_id. Skip silently.
                continue
            base_def = defs_by_scope_and_name.get((module_def_id, trait_name))
            inh_records.append(
                (target_def, trait_name, _next_ord(target_def), base_def, "inferred")
            )

        # #[derive(Trait1, Trait2, …)] above struct/enum/union
        for ts in ts_walk:
            if ts.id in test_skip_ts_ids:
                continue
            if ts.type not in ("struct_item", "enum_item", "union_item"):
                continue
            name_node = ts.child_by_field_name("name")
            if name_node is None:
                continue
            target_name = _text(name_node)
            target_def = defs_by_scope_and_name.get((module_def_id, target_name))
            if target_def is None:
                continue
            for trait_name in _collect_rust_derives(ts):
                base_def = defs_by_scope_and_name.get((module_def_id, trait_name))
                inh_records.append(
                    (target_def, trait_name, _next_ord(target_def), base_def, "inferred")
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
        if rule.confidence == "inferred":
            # Try file-local name match for inferred confidence.
            cands = file_name_index.get(name, [])
            target = cands[0] if len(cands) == 1 else None
        else:
            target = _resolve_in_scope(name, ts)
        ref_records.append(
            _RefRecord(
                db_node_id=db_id_for[ts.id],
                file_version_id=file_version_id,
                name=name,
                confidence=rule.confidence,
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
        await conn.executemany(
            """
            INSERT INTO call_edges (branch_id, callsite_node_id, caller_def_id, callee_def_id, callee_name, confidence)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [
                (branch_id, r.callsite_db_node_id, r.caller_def_id, r.callee_def_id, r.callee_name, r.confidence)
                for r in call_records
            ],
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
        await conn.executemany(
            """
            INSERT INTO data_access (branch_id, accessor_def_id, target_def_id, access_type, node_id)
            VALUES ($1, $2, $3, $4, $5)
            """,
            [(branch_id, r.accessor_def_id, r.target_def_id, r.access_type, r.db_node_id) for r in da_records],
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
        if only_file_version_ids is None:
            rows = await conn.fetch(
                """
                SELECT bf.path, fv.id AS file_version_id, fv.language
                FROM branch_files bf
                JOIN file_versions fv ON fv.id = bf.file_version_id
                WHERE bf.branch_id = $1
                ORDER BY fv.id
                """,
                branch_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT bf.path, fv.id AS file_version_id, fv.language
                FROM branch_files bf
                JOIN file_versions fv ON fv.id = bf.file_version_id
                WHERE bf.branch_id = $1 AND fv.id = ANY($2::bigint[])
                ORDER BY fv.id
                """,
                branch_id,
                only_file_version_ids,
            )

    configs: dict[str, LanguageConfig] = {}
    results: list[ResolveFileResult] = []
    async with pool.acquire() as conn:
        for row in rows:
            lang = row["language"]
            if lang not in configs:
                try:
                    configs[lang] = load_language_config(lang)
                except FileNotFoundError:
                    continue
            cfg = configs[lang]
            async with conn.transaction():
                results.append(
                    await resolve_file(conn, branch_id, row["file_version_id"], row["path"], cfg)
                )
    return results


def resolve_repo_sync(
    repo_id: int, branch_id: int, dsn: str | None = None,
) -> list[ResolveFileResult]:
    from db.connection import pool_ctx

    async def _run():
        async with pool_ctx(dsn) as pool:
            return await resolve_repo(pool, repo_id, branch_id)

    return asyncio.run(_run())
