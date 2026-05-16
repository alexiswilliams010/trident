"""Rust-specific tree-sitter AST helpers shared between import extraction,
import resolution, and semantic-resolver hooks."""

from __future__ import annotations

from .._shared.ts_helpers import text


def rust_path_text(path_node) -> str:
    """Reconstruct a use-path string from a tree-sitter-rust path-shaped node.
    Handles identifier, type_identifier, crate / self / super, scoped_identifier,
    metavariable. Falls back to the node's raw text for anything else."""
    t = path_node.type
    if t in ("identifier", "type_identifier", "metavariable"):
        return text(path_node)
    if t in ("crate", "self", "super"):
        return text(path_node)
    if t == "scoped_identifier":
        p = path_node.child_by_field_name("path")
        n = path_node.child_by_field_name("name")
        prefix = rust_path_text(p) if p is not None else ""
        suffix = text(n) if n is not None else ""
        if prefix and suffix:
            return f"{prefix}::{suffix}"
        return prefix or suffix
    return text(path_node)


def flatten_rust_use(arg_node, prefix: str):
    """Yield (full_path, last_name_or_None) leaves from a use_declaration's
    argument. None signifies a wildcard (`use a::*` / `a::{*}`).

    Examples (top-level call uses prefix=''):
        use a::b::c;            → ('a::b::c', 'c')
        use a::{b, c};          → ('a::b', 'b'), ('a::c', 'c')
        use a::{b::c, d};       → ('a::b::c', 'c'), ('a::d', 'd')
        use a::*;               → ('a', None)
        use a::B as Bb;         → ('a::B', 'B')   # original name, not alias
    """
    t = arg_node.type
    if t == "use_as_clause":
        p = arg_node.child_by_field_name("path")
        if p is not None:
            yield from flatten_rust_use(p, prefix)
        return
    if t == "use_wildcard":
        # The path child (if any) is the only named child apart from the
        # implicit `*` token. `use foo::*;` parses path='foo'; bare `use *;` is
        # not legal Rust, so we always expect one.
        p = next((c for c in arg_node.children if c.is_named), None)
        if p is None:
            yield (prefix, None)
            return
        path_str = rust_path_text(p)
        full = f"{prefix}::{path_str}" if prefix and path_str else (path_str or prefix)
        yield (full, None)
        return
    if t == "scoped_use_list":
        p = arg_node.child_by_field_name("path")
        list_node = arg_node.child_by_field_name("list")
        path_str = rust_path_text(p) if p is not None else ""
        new_prefix = (
            f"{prefix}::{path_str}" if prefix and path_str
            else (path_str or prefix)
        )
        if list_node is None:
            return
        for c in list_node.children:
            if c.is_named:
                yield from flatten_rust_use(c, new_prefix)
        return
    # Leaf: identifier / type_identifier / scoped_identifier / crate / self / super.
    path_str = rust_path_text(arg_node)
    full = f"{prefix}::{path_str}" if prefix else path_str
    last = full.rsplit("::", 1)[-1] if "::" in full else full
    yield (full, last)


def rust_inside_test_subtree(ts) -> bool:
    """True if `ts` sits anywhere inside a `#[cfg(test)]` / `#[test]`-gated
    item or a `mod tests { … }` block. Mirrors the test-skip logic in
    semantic_resolver so import rows aren't created for test-only code.
    Walks the AST upward; returns on the first matching ancestor."""
    cur = ts.parent
    while cur is not None:
        if cur.type == "mod_item":
            name_node = cur.child_by_field_name("name")
            if name_node is not None and text(name_node) in ("tests", "test_utils"):
                return True
        if cur.type in (
            "mod_item", "function_item", "impl_item",
            "struct_item", "enum_item", "union_item",
            "trait_item", "type_item", "const_item",
            "static_item", "macro_definition",
        ):
            sib = cur.prev_named_sibling
            while sib is not None and sib.type == "attribute_item":
                attr = next((c for c in sib.children if c.type == "attribute"), None)
                if attr is not None:
                    path_name = None
                    for c in attr.children:
                        if c.type == "identifier":
                            path_name = text(c)
                            break
                        if c.type == "scoped_identifier":
                            n = c.child_by_field_name("name")
                            path_name = text(n) if n is not None else None
                            break
                    if path_name == "test":
                        return True
                    if path_name == "cfg":
                        args = attr.child_by_field_name("arguments")
                        if args is not None:
                            stack = [args]
                            while stack:
                                m = stack.pop()
                                if m.type == "identifier" and text(m) == "test":
                                    return True
                                for cc in m.children:
                                    stack.append(cc)
                sib = sib.prev_named_sibling
        cur = cur.parent
    return False
