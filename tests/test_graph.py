"""Tests for core/graph.py graph traversal functions."""

from __future__ import annotations

from pathlib import Path

from core.extractor import index_repo
from core.graph import (
    ancestors,
    callers_of,
    callees_of,
    entrypoints,
    file_dependents,
    file_imports,
    get_source,
    inheritance_tree,
    paths_between,
    reachable_from,
    resolve_definitions,
)
from core.heuristic_resolver import resolve_repo_imports
from core.semantic_resolver import resolve_repo


async def _seed(pool, repo_id: int, root: Path):
    await index_repo(pool, repo_id, root)
    await resolve_repo(pool, repo_id)
    await resolve_repo_imports(pool, repo_id)


# ────────────────────────────────────────────────────────────────────
# resolve_definitions
# ────────────────────────────────────────────────────────────────────


async def test_resolve_by_simple_name(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    defs = await resolve_definitions(pool, repo_id, "helper")
    assert len(defs) >= 1
    assert any(d.qualified_name == "utils.helper" for d in defs)


async def test_resolve_by_qualified_name(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    defs = await resolve_definitions(pool, repo_id, "main.Calculator.add")
    assert len(defs) == 1
    assert defs[0].kind == "function"


async def test_resolve_by_suffix(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    defs = await resolve_definitions(pool, repo_id, "Calculator.add")
    assert len(defs) == 1
    assert defs[0].qualified_name == "main.Calculator.add"


async def test_resolve_with_kind_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    defs = await resolve_definitions(pool, repo_id, "Calculator", kind="class")
    assert len(defs) == 1
    assert defs[0].kind == "class"


# ────────────────────────────────────────────────────────────────────
# callers_of / callees_of
# ────────────────────────────────────────────────────────────────────


async def test_callers_of(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    callers = await callers_of(pool, repo_id, "helper")
    qnames = {d.qualified_name for d in callers}
    assert "main.Calculator.add" in qnames or "utils.double" in qnames


async def test_callees_of(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    callees = await callees_of(pool, repo_id, "Calculator.add")
    qnames = {d.qualified_name for d in callees}
    assert "utils.helper" in qnames


# ────────────────────────────────────────────────────────────────────
# ancestors / reachable_from
# ────────────────────────────────────────────────────────────────────


async def test_ancestors_of_helper(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    anc = await ancestors(pool, repo_id, "helper")
    qnames = {d.qualified_name for d in anc}
    # helper is called by Calculator.add and double; Calculator.add is called by run
    assert "utils.double" in qnames or "main.Calculator.add" in qnames
    assert all(d.depth is not None and d.depth >= 1 for d in anc)


async def test_ancestors_with_max_depth(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    anc_1 = await ancestors(pool, repo_id, "helper", max_depth=1)
    anc_all = await ancestors(pool, repo_id, "helper")
    assert len(anc_1) <= len(anc_all)


async def test_reachable_from_run(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    reach = await reachable_from(pool, repo_id, "run")
    qnames = {d.qualified_name for d in reach}
    # run → Calculator.add → helper
    assert "main.Calculator.add" in qnames or "utils.helper" in qnames
    assert all(d.depth is not None and d.depth >= 1 for d in reach)


# ────────────────────────────────────────────────────────────────────
# paths_between
# ────────────────────────────────────────────────────────────────────


async def test_paths_between(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    result = await paths_between(pool, repo_id, "run", "helper")
    # run → Calculator.add → helper (or via double)
    assert len(result) >= 1
    for path in result:
        assert path[0].qualified_name == "main.run"
        assert path[-1].qualified_name == "utils.helper"


async def test_paths_between_no_path(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    result = await paths_between(pool, repo_id, "helper", "run")
    # helper does not call run — no path in forward direction
    assert len(result) == 0


# ────────────────────────────────────────────────────────────────────
# entrypoints
# ────────────────────────────────────────────────────────────────────


async def test_entrypoints(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    eps = await entrypoints(pool, repo_id, include_internal=True)
    qnames = {d.qualified_name for d in eps}
    # run and greet have no callers
    assert "main.run" in qnames
    assert "main.greet" in qnames
    # helper is called by add and double — should NOT be an entrypoint
    assert "utils.helper" not in qnames


async def test_entrypoints_kind_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    eps = await entrypoints(pool, repo_id, kind="function", include_internal=True)
    assert all(d.kind == "function" for d in eps)


async def test_entrypoints_excludes_overrides(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    all_eps = await entrypoints(pool, repo_id, include_internal=True)
    default_eps = await entrypoints(pool, repo_id)
    all_qnames = {d.qualified_name for d in all_eps}
    default_qnames = {d.qualified_name for d in default_eps}
    # StrictPolicy.is_active overrides BasePolicy.is_active — excluded by default
    has_override = any("StrictPolicy" in qn and "is_active" in qn for qn in all_qnames)
    if has_override:
        assert not any("StrictPolicy" in qn and "is_active" in qn for qn in default_qnames)


# ────────────────────────────────────────────────────────────────────
# get_source
# ────────────────────────────────────────────────────────────────────


async def test_get_source(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    defs = await get_source(pool, repo_id, "helper")
    assert len(defs) >= 1
    src = defs[0]
    assert src.source is not None
    assert "def helper" in src.source
    assert "return x + 1" in src.source


async def test_get_source_handles_multibyte_utf8(clean_repo, tmp_path: Path):
    """Regression: tree-sitter byte offsets index into UTF-8 bytes, not
    Python codepoints. If get_source slices the str directly by them,
    every multi-byte char preceding the def shifts the returned source
    forward — typically chopping off the `def` line and leaking the next
    statement's tail. The fix encodes raw_content to bytes before slicing
    and decodes the slice back to str."""
    pool, repo_id = clean_repo
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # Pile non-ASCII into the file ahead of the target so byte- and
    # codepoint-indexed offsets diverge by many characters.
    (pkg / "mod.py").write_text(
        '"""Привет мир — multibyte greeting 日本語 🚀."""\n'
        "# Ω + Ω = 2Ω — more multibyte filler\n"
        "leading = '日本語'\n"
        "\n"
        "def regression_target(x):\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    await _seed(pool, repo_id, tmp_path)
    defs = await get_source(pool, repo_id, "regression_target")
    assert len(defs) >= 1
    src = defs[0].source
    assert src is not None
    assert src.startswith("def regression_target")
    assert "return x + 1" in src


# ────────────────────────────────────────────────────────────────────
# file_imports / file_dependents
# ────────────────────────────────────────────────────────────────────


async def test_file_imports(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    imps = await file_imports(pool, repo_id)
    assert len(imps) >= 1
    paths = {i.import_path for i in imps}
    assert "mypackage.utils" in paths or any("utils" in p for p in paths)


async def test_file_imports_by_file(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    all_imps = await file_imports(pool, repo_id)
    main_files = [i for i in all_imps if "main" in i.file_path]
    if main_files:
        main_path = main_files[0].file_path
        filtered = await file_imports(pool, repo_id, file_path=main_path)
        assert all(i.file_path == main_path for i in filtered)


async def test_file_imports_by_dep_class(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    external = await file_imports(pool, repo_id, dep_class="external")
    assert all(i.dep_class == "external" for i in external)


async def test_file_dependents(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    all_imps = await file_imports(pool, repo_id, dep_class="intra_repo")
    resolved = [i for i in all_imps if i.resolved_file is not None]
    if resolved:
        target_path = resolved[0].resolved_file
        deps = await file_dependents(pool, repo_id, target_path)
        assert len(deps) >= 1


# ────────────────────────────────────────────────────────────────────
# inheritance_tree
# ────────────────────────────────────────────────────────────────────


async def test_inheritance_tree(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    tree = await inheritance_tree(pool, repo_id, "BasePolicy")
    qnames = {n.def_info.qualified_name for n in tree}
    assert "policies.BasePolicy" in qnames
    assert "policies.StrictPolicy" in qnames
    strict = [n for n in tree if "StrictPolicy" in n.def_info.qualified_name]
    if strict:
        assert any("BasePolicy" in b for b in strict[0].bases)


async def test_inheritance_tree_from_child(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _seed(pool, repo_id, python_fixture_root)
    tree = await inheritance_tree(pool, repo_id, "StrictPolicy")
    qnames = {n.def_info.qualified_name for n in tree}
    assert "policies.BasePolicy" in qnames
    assert "policies.StrictPolicy" in qnames
