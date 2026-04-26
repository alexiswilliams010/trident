"""Helpers for the JS/TS heuristic resolver: tsconfig.json paths, package.json
package-name rollup, and Node-style relative-path probing with extension
fallbacks.

Kept out of `heuristic_resolver` so the JS/TS-specific path math doesn't
balloon that file. All functions are pure (no DB / asyncio).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


# Extension probe order matches what `bundler` / TS resolution does in
# practice: prefer TS sources over compiled JS when both happen to coexist.
NODE_EXTENSIONS: tuple[str, ...] = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


@dataclass(frozen=True)
class TsconfigPaths:
    """Parsed `compilerOptions.baseUrl` + `compilerOptions.paths` from
    tsconfig.json. `base_url_rel` is repo-relative posix (empty string when
    baseUrl is the repo root). `paths` preserves declaration order."""

    base_url_rel: str
    paths: tuple[tuple[str, tuple[str, ...]], ...]


# ────────────────────────────────────────────────────────────────────
# tsconfig.json
# ────────────────────────────────────────────────────────────────────


def load_tsconfig_paths(repo_root: Path, max_extends_depth: int = 3) -> TsconfigPaths | None:
    """Return the `paths` map declared at `<repo_root>/tsconfig.json`, with
    `extends` chain resolved up to `max_extends_depth` hops. Returns None if no
    tsconfig.json is present or the file is malformed.
    """
    primary = repo_root / "tsconfig.json"
    if not primary.is_file():
        return None

    chain: list[tuple[Path, dict]] = []
    cur = primary
    seen: set[Path] = set()
    for _ in range(max_extends_depth + 1):
        try:
            cur_resolved = cur.resolve()
        except (OSError, RuntimeError):
            break
        if cur_resolved in seen:
            break
        seen.add(cur_resolved)
        try:
            data = _parse_jsonc(cur.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        chain.append((cur.parent, data))
        ext = data.get("extends")
        if not isinstance(ext, str):
            break
        candidate = (cur.parent / ext)
        # tsconfig `extends` may omit the `.json` suffix.
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".json")
        if not candidate.is_file():
            break
        cur = candidate

    if not chain:
        return None

    # Merge compilerOptions bottom-up: the deepest `extends` provides defaults,
    # the primary file overrides.
    merged: dict = {}
    for _, data in reversed(chain):
        co = data.get("compilerOptions") or {}
        if isinstance(co, dict):
            merged.update(co)

    base_dir = chain[0][0]  # primary's directory
    base_url_raw = merged.get("baseUrl")
    if isinstance(base_url_raw, str):
        base_url_abs = (base_dir / base_url_raw).resolve()
    else:
        base_url_abs = base_dir.resolve()

    try:
        base_url_rel = base_url_abs.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        base_url_rel = ""
    if base_url_rel in (".", "/"):
        base_url_rel = ""

    paths_raw = merged.get("paths") or {}
    paths_list: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(paths_raw, dict):
        for pattern, targets in paths_raw.items():
            if not isinstance(pattern, str):
                continue
            if isinstance(targets, list):
                t = tuple(x for x in targets if isinstance(x, str))
            elif isinstance(targets, str):
                t = (targets,)
            else:
                continue
            if t:
                paths_list.append((pattern, t))

    return TsconfigPaths(base_url_rel=base_url_rel, paths=tuple(paths_list))


# String-or-comment alternation. Strings are matched first so comment-like
# sequences inside them (e.g. `"@app/*"`) don't get treated as `/* ... */`.
# The substitution callback keeps strings verbatim and drops comments.
_JSONC_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'  # JSON string (any escaped char or non-quote/backslash)
    r"|//[^\n]*"           # line comment
    r"|/\*.*?\*/",         # block comment
    re.DOTALL,
)
_JSONC_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _parse_jsonc(text: str) -> dict:
    """Tolerate JSON-with-comments syntax used by tsconfig.json files.

    A single alternation regex matches either a JSON string or a comment;
    the substitution callback preserves strings and drops comments, so
    comment-like sequences inside string values are left intact. After
    comment stripping, trailing commas before `}` / `]` are removed.
    """
    stripped = _JSONC_TOKEN_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        text,
    )
    stripped = _JSONC_TRAILING_COMMA_RE.sub(r"\1", stripped)
    return json.loads(stripped)


# ────────────────────────────────────────────────────────────────────
# Path matching / probing
# ────────────────────────────────────────────────────────────────────


def match_path_alias(pattern: str, spec: str) -> str | None:
    """If `pattern` matches `spec`, return the captured wildcard text (empty
    string if pattern has no `*`). Otherwise None.

    Patterns may contain at most one `*`, which captures one or more chars.
    """
    if "*" not in pattern:
        return "" if pattern == spec else None
    prefix, _, suffix = pattern.partition("*")
    if not spec.startswith(prefix) or not spec.endswith(suffix):
        return None
    end = len(spec) - len(suffix)
    if end <= len(prefix):
        return None
    return spec[len(prefix):end]


def node_candidates(base_rel: str) -> list[str]:
    """Yield the ordered list of repo-relative paths Node would probe when
    resolving `base_rel` (no extension assumed)."""
    out: list[str] = [base_rel]
    for ext in NODE_EXTENSIONS:
        out.append(base_rel + ext)
    for ext in NODE_EXTENSIONS:
        out.append(f"{base_rel}/index{ext}")
    return out


def normalize_relative_posix(path: str) -> str:
    """`a/b/../c/./d` → `a/c/d`, ignoring filesystem state."""
    parts: list[str] = []
    for seg in path.split("/"):
        if seg == "" or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def resolve_relative(
    importer_rel_path: str, spec: str, file_index: dict[str, int],
) -> int | None:
    """Resolve `./foo`/`../foo` against an in-memory file index.

    Returns the matched file_id or None.
    """
    importer_dir = PurePosixPath(importer_rel_path).parent
    joined = (importer_dir / spec).as_posix() if importer_dir.as_posix() != "." else spec
    base = normalize_relative_posix(joined)
    for cand in node_candidates(base):
        if cand in file_index:
            return file_index[cand]
    return None


def resolve_tsconfig_alias(
    spec: str, paths_cfg: TsconfigPaths, file_index: dict[str, int],
) -> int | None:
    """Apply tsconfig `paths` aliases to `spec`. Returns a file_id from
    `file_index` or None.
    """
    base_url = paths_cfg.base_url_rel
    for pattern, targets in paths_cfg.paths:
        captured = match_path_alias(pattern, spec)
        if captured is None:
            continue
        for target in targets:
            replaced = target.replace("*", captured) if "*" in target else target
            full_rel = f"{base_url}/{replaced}" if base_url else replaced
            full_rel = normalize_relative_posix(full_rel)
            for cand in node_candidates(full_rel):
                if cand in file_index:
                    return file_index[cand]
    return None


# ────────────────────────────────────────────────────────────────────
# Bare specifiers
# ────────────────────────────────────────────────────────────────────


def package_name_for_specifier(spec: str) -> str:
    """Roll up a bare specifier to its package portion:
       'react'              → 'react'
       'react/jsx-runtime'  → 'react'
       '@scope/pkg'         → '@scope/pkg'
       '@scope/pkg/sub'     → '@scope/pkg'
    """
    parts = spec.split("/")
    if spec.startswith("@") and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def is_relative_specifier(spec: str) -> bool:
    """Node-resolution `relative specifier` predicate: `./`, `../`, or just
    `.` / `..`."""
    return spec.startswith("./") or spec.startswith("../") or spec in (".", "..")


__all__ = [
    "NODE_EXTENSIONS",
    "TsconfigPaths",
    "is_relative_specifier",
    "load_tsconfig_paths",
    "match_path_alias",
    "node_candidates",
    "normalize_relative_posix",
    "package_name_for_specifier",
    "resolve_relative",
    "resolve_tsconfig_alias",
]
