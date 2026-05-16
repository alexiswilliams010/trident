"""Rust-specific semantic-resolver hooks.

Covers the three Rust edge cases the YAML-driven resolver can't express:
  • whole-file skip for `tests/` / `benches/` / `examples/` / `test_utils/`
  • test-gated node filtering (`#[cfg(test)]`, `#[test]`, `mod tests { … }`)
    — precomputed once per file, consumed by every emission loop
  • inheritance synthesis from `impl Trait for Type` and `#[derive(...)]`
  • qualified-name prefixing for `impl` methods (`Counter::new` → `Counter`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._shared.ts_helpers import terminal_identifier, text
from ..base import InheritanceEdge, ResolvedReference

if TYPE_CHECKING:
    from ..base import SemanticContext


_RUST_TEST_PATH_SEGMENTS = frozenset({"tests", "benches", "examples", "test_utils"})

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


def is_rust_test_path(rel_path: str) -> bool:
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
            return text(c)
        if c.type == "scoped_identifier":
            return terminal_identifier(c)
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
        if n.type == "identifier" and text(n) == "test":
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


def collect_rust_test_skip_ids(ts_walk) -> frozenset[int]:
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
            if name_node is not None and text(name_node) in _RUST_TEST_MOD_NAMES:
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
                path_name = text(c)
                break
            if c.type == "scoped_identifier":
                path_name = terminal_identifier(c)
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
                        this_attr.append(text(c))
                    elif c.type == "scoped_identifier":
                        n = terminal_identifier(c)
                        if n:
                            this_attr.append(n)
                out = this_attr + out
        cursor = cursor.prev_named_sibling
    return out


def collect_rust_impl_method_targets(ts_walk: list) -> dict[int, str]:
    """Map each `function_item` ts_id sitting inside an `impl Foo { … }` block
    to the impl's target type name (`Foo`). Used by `resolve_rust_reference`
    to type-resolve `self.x` field accesses to the right struct's field
    when name-only lookup would be ambiguous."""
    out: dict[int, str] = {}
    for ts in ts_walk:
        if ts.type != "impl_item":
            continue
        type_field = ts.child_by_field_name("type")
        if type_field is None:
            continue
        target_name = terminal_identifier(type_field)
        if not target_name:
            continue
        body = ts.child_by_field_name("body")
        if body is None:
            continue
        for child in body.children:
            if child.type == "function_item":
                out[child.id] = target_name
    return out


def resolve_rust_reference(ts_node, rule, ctx: "SemanticContext") -> ResolvedReference | None:
    """Type-aware override for `self.x` field accesses inside `impl` methods.

    Catches the one pattern where AST context unambiguously identifies the
    target struct: `field_expression` whose `value` is the literal `self`
    keyword, inside a `function_item` that lives in some `impl Foo { … }`.
    The struct's `Foo.x` field is looked up via the per-scope name index.

    Other shapes (`obj.x`, `Self::x`, chained `self.inner().x`) need real
    type inference and are left to the default file_name_index path —
    which only resolves when the field name is unique in the file, so the
    no-false-positive guarantee is preserved.
    """
    if ts_node.type != "field_identifier":
        return None
    parent = ts_node.parent
    if parent is None or parent.type != "field_expression":
        return None
    field_child = parent.child_by_field_name("field")
    # tree-sitter Python returns fresh wrapper objects from child_by_field_name
    # each call, so `is`/`==` won't work — compare the underlying node id.
    if field_child is None or field_child.id != ts_node.id:
        return None
    value = parent.child_by_field_name("value")
    if value is None or value.type != "self":
        return None

    impl_method_target = ctx.scratch.get("impl_method_target", {})
    if not impl_method_target or ctx.module_def_id is None:
        return None

    cur = parent.parent
    while cur is not None:
        if cur.type == "function_item":
            target_name = impl_method_target.get(cur.id)
            if target_name is None:
                return None
            struct_did = ctx.defs_by_scope_and_name.get(
                (ctx.module_def_id, target_name)
            )
            if struct_did is None:
                return None
            field_did = ctx.defs_by_scope_and_name.get((struct_did, text(ts_node)))
            if field_did is None:
                return None
            return ResolvedReference(target_def_id=field_did, confidence="certain")
        cur = cur.parent
    return None


def rust_qualified_name_prefix(ts_node, ctx: "SemanticContext") -> str | None:
    """For a `function_item` whose AST grandparent is an `impl_item`, return
    the impl's target type as a qualified-name prefix segment (`Counter::new`
    → 'Counter'). impl_item is intentionally not a definition of its own
    (see configs/rust.yaml) so the prefix has to come from the AST, not the
    scope chain."""
    if ts_node.type != "function_item":
        return None
    parent = ts_node.parent
    if parent is None or parent.type != "declaration_list":
        return None
    gp = parent.parent
    if gp is None or gp.type != "impl_item":
        return None
    type_field = gp.child_by_field_name("type")
    if type_field is None:
        return None
    return terminal_identifier(type_field)


def synthesize_rust_inheritance(ts_walk: list, ctx: "SemanticContext") -> list[InheritanceEdge]:
    """Two synthetic shapes the YAML inheritance machinery can't express:
      • `impl Trait for Type { … }` → edge Type → Trait. The child here is
        the Type's existing struct/enum/union/type def, not the impl_item
        (which is intentionally not a definition).
      • `#[derive(Trait1, Trait2)]` on a struct/enum/union → one edge per
        derived trait. Tree-sitter never expands the macro, so the actual
        `impl Trait for Type { ... }` block rust-analyzer would see is
        invisible to us; we synthesize the edges best-effort. Confidence
        `inferred` reflects the heuristic nature of both shapes.
    """
    out: list[InheritanceEdge] = []
    if ctx.module_def_id is None:
        return out
    test_skip = ctx.scratch.get("test_skip_ts_ids", frozenset())
    defs_lookup = ctx.defs_by_scope_and_name
    module_def_id = ctx.module_def_id

    # impl Trait for Type → Type's def → Trait
    for ts in ts_walk:
        if ts.id in test_skip:
            continue
        if ts.type != "impl_item":
            continue
        type_field = ts.child_by_field_name("type")
        trait_field = ts.child_by_field_name("trait")
        if type_field is None or trait_field is None:
            continue
        target_name = terminal_identifier(type_field)
        trait_name = terminal_identifier(trait_field)
        if not target_name or not trait_name:
            continue
        target_def = defs_lookup.get((module_def_id, target_name))
        if target_def is None:
            # Type defined in another file — Phase 3 will not currently
            # link this since inherits cross-file resolution keys on
            # child_def_id. Skip silently.
            continue
        base_def = defs_lookup.get((module_def_id, trait_name))
        out.append(InheritanceEdge(target_def, trait_name, base_def, "inferred"))

    # #[derive(Trait1, Trait2, …)] above struct/enum/union
    for ts in ts_walk:
        if ts.id in test_skip:
            continue
        if ts.type not in ("struct_item", "enum_item", "union_item"):
            continue
        name_node = ts.child_by_field_name("name")
        if name_node is None:
            continue
        target_name = text(name_node)
        target_def = defs_lookup.get((module_def_id, target_name))
        if target_def is None:
            continue
        for trait_name in _collect_rust_derives(ts):
            base_def = defs_lookup.get((module_def_id, trait_name))
            out.append(InheritanceEdge(target_def, trait_name, base_def, "inferred"))

    return out
