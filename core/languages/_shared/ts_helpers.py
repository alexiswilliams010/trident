"""Tiny tree-sitter helpers reused across language handlers + the semantic resolver."""

from __future__ import annotations

from typing import Iterator


def text(ts_node) -> str:
    return ts_node.text.decode("utf-8", errors="replace")


def strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def dfs(root) -> Iterator:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for i in range(n.child_count - 1, -1, -1):
            stack.append(n.children[i])


def peel_expression(ts_node):
    """Solidity wraps many constructs in `expression`. Peel single-child wrappers."""
    while ts_node is not None and ts_node.type == "expression" and ts_node.child_count == 1:
        ts_node = ts_node.children[0]
    return ts_node


def terminal_identifier(ts_node) -> str | None:
    """Walk into wrapper nodes (user_defined_type, expression, attribute, …) to
    pull out the trailing identifier text. Used by inheritance extraction where
    the base name is wrapped in a type node."""
    node = peel_expression(ts_node)
    if node is None:
        return None
    if node.type in ("identifier", "type_identifier"):
        return text(node)
    if node.type == "user_defined_type":
        # Solidity: user_defined_type wraps the base identifier (or a dotted path).
        last_ident: str | None = None
        for c in node.children:
            if c.type in ("identifier", "type_identifier"):
                last_ident = text(c)
        return last_ident
    if node.type == "qualified_type":
        # Go: `pkg.Foo` — the type-side identifier is what we resolve against.
        name_node = node.child_by_field_name("name")
        return text(name_node) if name_node is not None else None
    if node.type == "type_elem":
        # Go interface embedding: a type_elem wraps either a bare
        # type_identifier (same-package embed) or a qualified_type
        # (cross-package embed). Both are unwrapped here.
        for c in node.children:
            if c.type == "type_identifier":
                return text(c)
            if c.type == "qualified_type":
                name_node = c.child_by_field_name("name")
                return text(name_node) if name_node is not None else None
        return None
    if node.type == "pointer_type":
        # Go embedded fields can be `*Header` — peel the pointer and recurse so
        # the inner type_identifier / qualified_type lookup applies uniformly.
        for c in node.children:
            if c.is_named:
                return terminal_identifier(c)
        return None
    if node.type == "class_heritage":
        # JS: `class Dog extends Animal` — class_heritage holds an `extends`
        # keyword and an `identifier` directly. TS: the same shape but the
        # identifier is wrapped in `extends_clause` (with field `value`); a
        # sibling `implements_clause` may also be present and is handled by a
        # separate inheritance rule. Here we surface only the extends side.
        for c in node.children:
            if c.type == "identifier":
                return text(c)
            if c.type == "extends_clause":
                v = c.child_by_field_name("value")
                return text(v) if v is not None else None
        return None
    if node.type == "attribute":
        prop = node.child_by_field_name("attribute")
        return text(prop) if prop is not None else None
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        return text(prop) if prop is not None else None
    if node.type == "scoped_type_identifier":
        # Rust: `std::fmt::Display` — the trailing `name` field is the trait.
        name_node = node.child_by_field_name("name")
        return text(name_node) if name_node is not None else None
    if node.type == "scoped_identifier":
        # Rust: `serde::Serialize` used in trait position via generic_type.
        name_node = node.child_by_field_name("name")
        return text(name_node) if name_node is not None else None
    if node.type == "generic_type":
        # Rust: `From<u32>` — drop the type arguments, recurse on the base.
        base = node.child_by_field_name("type")
        return terminal_identifier(base) if base is not None else None
    return None
