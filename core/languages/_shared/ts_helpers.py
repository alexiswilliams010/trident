"""Tiny tree-sitter helpers reused across language handlers."""

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
