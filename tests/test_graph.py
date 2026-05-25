"""Tests for core/graph.py graph traversal functions."""

from __future__ import annotations

from pathlib import Path

from core.extractor import index_repo
from core.graph import (
    ancestors,
    callers_of,
    callees_of,
    definitions_in_file,
    entrypoints,
    entrypoints_reaching,
    file_dependents,
    file_imports,
    get_source,
    inheritance_tree,
    is_reachable,
    paths_between,
    readers_of,
    reachable_from,
    resolve_definitions,
    taint_paths,
    writers_of,
)
from core.heuristic_resolver import resolve_branch_imports
from core.semantic_resolver import resolve_repo


async def _seed(pool, repo_id: int, branch_id: int, root: Path):
    await index_repo(pool, repo_id, branch_id, root)
    await resolve_repo(pool, repo_id, branch_id)
    await resolve_branch_imports(pool, repo_id, branch_id)


async def test_definitions_in_file_lists_all_functions(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)

    # Module-level functions (would not all be entrypoints) are listed.
    utils = await definitions_in_file(pool, branch_id, "utils.py", kind="function")
    names = {d.name for d in utils}
    assert {"helper", "double"} <= names

    # Suffix match works, and the kind filter narrows to functions only.
    main_all = await definitions_in_file(pool, branch_id, "mypackage/main.py")
    main_fns = await definitions_in_file(pool, branch_id, "main.py", kind="function")
    assert len(main_all) >= len(main_fns)
    main_names = {d.name for d in main_fns}
    # Includes the free function AND class methods (the enumeration entrypoints misses).
    assert {"run", "greet", "add"} <= main_names

    # Unknown file → empty, never an error.
    assert await definitions_in_file(pool, branch_id, "does_not_exist.py") == []


# ────────────────────────────────────────────────────────────────────
# resolve_definitions
# ────────────────────────────────────────────────────────────────────


async def test_resolve_by_simple_name(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    defs = await resolve_definitions(pool, branch_id, "helper")
    assert len(defs) >= 1
    assert any(d.qualified_name == "utils.helper" for d in defs)


async def test_resolve_by_qualified_name(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    defs = await resolve_definitions(pool, branch_id, "main.Calculator.add")
    assert len(defs) == 1
    assert defs[0].kind == "function"


async def test_resolve_by_suffix(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    defs = await resolve_definitions(pool, branch_id, "Calculator.add")
    assert len(defs) == 1
    assert defs[0].qualified_name == "main.Calculator.add"


async def test_resolve_with_kind_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    defs = await resolve_definitions(pool, branch_id, "Calculator", kind="class")
    assert len(defs) == 1
    assert defs[0].kind == "class"


# ────────────────────────────────────────────────────────────────────
# callers_of / callees_of
# ────────────────────────────────────────────────────────────────────


async def test_callers_of(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    callers = await callers_of(pool, branch_id, "helper")
    qnames = {d.qualified_name for d in callers}
    assert "main.Calculator.add" in qnames or "utils.double" in qnames


async def test_callees_of(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    callees = await callees_of(pool, branch_id, "Calculator.add")
    qnames = {d.qualified_name for d in callees}
    assert "utils.helper" in qnames


# ────────────────────────────────────────────────────────────────────
# ancestors / reachable_from
# ────────────────────────────────────────────────────────────────────


async def test_ancestors_of_helper(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    anc = await ancestors(pool, branch_id, "helper")
    qnames = {d.qualified_name for d in anc}
    assert "utils.double" in qnames or "main.Calculator.add" in qnames
    assert all(d.depth is not None and d.depth >= 1 for d in anc)


async def test_ancestors_with_max_depth(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    anc_1 = await ancestors(pool, branch_id, "helper", max_depth=1)
    anc_all = await ancestors(pool, branch_id, "helper")
    assert len(anc_1) <= len(anc_all)


async def test_reachable_from_run(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    reach = await reachable_from(pool, branch_id, "run")
    qnames = {d.qualified_name for d in reach}
    assert "main.Calculator.add" in qnames or "utils.helper" in qnames
    assert all(d.depth is not None and d.depth >= 1 for d in reach)


# ────────────────────────────────────────────────────────────────────
# paths_between
# ────────────────────────────────────────────────────────────────────


async def test_paths_between(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    result = await paths_between(pool, branch_id, "run", "helper")
    assert len(result) >= 1
    for path in result:
        assert path[0].qualified_name == "main.run"
        assert path[-1].qualified_name == "utils.helper"


async def test_paths_between_no_path(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    result = await paths_between(pool, branch_id, "helper", "run")
    assert len(result) == 0


# ────────────────────────────────────────────────────────────────────
# entrypoints
# ────────────────────────────────────────────────────────────────────


async def test_entrypoints(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    eps = await entrypoints(pool, branch_id, include_internal=True)
    qnames = {d.qualified_name for d in eps}
    assert "main.run" in qnames
    assert "main.greet" in qnames
    assert "utils.helper" not in qnames


async def test_entrypoints_kind_filter(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    eps = await entrypoints(pool, branch_id, kind="function", include_internal=True)
    assert all(d.kind == "function" for d in eps)


async def test_entrypoints_excludes_overrides(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    all_eps = await entrypoints(pool, branch_id, include_internal=True)
    default_eps = await entrypoints(pool, branch_id)
    all_qnames = {d.qualified_name for d in all_eps}
    default_qnames = {d.qualified_name for d in default_eps}
    has_override = any("StrictPolicy" in qn and "is_active" in qn for qn in all_qnames)
    if has_override:
        assert not any("StrictPolicy" in qn and "is_active" in qn for qn in default_qnames)


# ────────────────────────────────────────────────────────────────────
# get_source
# ────────────────────────────────────────────────────────────────────


async def test_get_source(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    defs = await get_source(pool, branch_id, "helper")
    assert len(defs) >= 1
    src = defs[0]
    assert src.source is not None
    assert "def helper" in src.source
    assert "return x + 1" in src.source


async def test_get_source_handles_multibyte_utf8(clean_repo, tmp_path: Path):
    pool, repo_id, branch_id = clean_repo
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(
        '"""Привет мир — multibyte greeting 日本語 🚀."""\n'
        "# Ω + Ω = 2Ω — more multibyte filler\n"
        "leading = '日本語'\n"
        "\n"
        "def regression_target(x):\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    await _seed(pool, repo_id, branch_id, tmp_path)
    defs = await get_source(pool, branch_id, "regression_target")
    assert len(defs) >= 1
    src = defs[0].source
    assert src is not None
    assert src.startswith("def regression_target")
    assert "return x + 1" in src


# ────────────────────────────────────────────────────────────────────
# file_imports / file_dependents
# ────────────────────────────────────────────────────────────────────


async def test_file_imports(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    imps = await file_imports(pool, branch_id)
    assert len(imps) >= 1
    paths = {i.import_path for i in imps}
    assert "mypackage.utils" in paths or any("utils" in p for p in paths)


async def test_file_imports_by_file(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    all_imps = await file_imports(pool, branch_id)
    main_files = [i for i in all_imps if "main" in i.file_path]
    if main_files:
        main_path = main_files[0].file_path
        filtered = await file_imports(pool, branch_id, file_path=main_path)
        assert all(i.file_path == main_path for i in filtered)


async def test_file_imports_by_dep_class(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    external = await file_imports(pool, branch_id, dep_class="external")
    assert all(i.dep_class == "external" for i in external)


async def test_file_dependents(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    all_imps = await file_imports(pool, branch_id, dep_class="intra_repo")
    resolved = [i for i in all_imps if i.resolved_file is not None]
    if resolved:
        target_path = resolved[0].resolved_file
        deps = await file_dependents(pool, branch_id, target_path)
        assert len(deps) >= 1


# ────────────────────────────────────────────────────────────────────
# inheritance_tree
# ────────────────────────────────────────────────────────────────────


async def test_inheritance_tree(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    tree = await inheritance_tree(pool, branch_id, "BasePolicy")
    qnames = {n.def_info.qualified_name for n in tree}
    assert "policies.BasePolicy" in qnames
    assert "policies.StrictPolicy" in qnames
    strict = [n for n in tree if "StrictPolicy" in n.def_info.qualified_name]
    if strict:
        assert any("BasePolicy" in b for b in strict[0].bases)


async def test_inheritance_tree_from_child(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    tree = await inheritance_tree(pool, branch_id, "StrictPolicy")
    qnames = {n.def_info.qualified_name for n in tree}
    assert "policies.BasePolicy" in qnames
    assert "policies.StrictPolicy" in qnames


# ────────────────────────────────────────────────────────────────────
# is_reachable
# ────────────────────────────────────────────────────────────────────


async def test_is_reachable_true_for_existing_call_chain(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    # run() → Calculator.add → helper  (this chain exists in the fixture)
    assert await is_reachable(pool, branch_id, "run", "helper") is True


async def test_is_reachable_false_when_no_path(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    # helper is a leaf — it doesn't call run.
    assert await is_reachable(pool, branch_id, "helper", "run") is False


async def test_is_reachable_false_when_name_unknown(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    assert await is_reachable(pool, branch_id, "no_such_fn", "helper") is False
    assert await is_reachable(pool, branch_id, "helper", "no_such_target") is False


# ────────────────────────────────────────────────────────────────────
# readers_of / writers_of (data_access)
# ────────────────────────────────────────────────────────────────────


async def test_readers_of_global_variable(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    # main.greet reads top-level GREETING (see fixture comment in main.py).
    readers = await readers_of(pool, branch_id, "GREETING")
    qnames = {d.qualified_name for d in readers}
    assert "main.greet" in qnames


async def test_writers_of_unwritten_constant_is_empty(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    # GREETING is only read, never written after init.
    writers = await writers_of(pool, branch_id, "GREETING")
    assert writers == []


# ────────────────────────────────────────────────────────────────────
# taint_paths (call_edges ∪ data_access, sanitizer-aware)
# ────────────────────────────────────────────────────────────────────


async def test_taint_paths_finds_call_only_route(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    # Pure call chain — should also be discoverable via taint_paths.
    result = await taint_paths(pool, branch_id, "run", "helper")
    assert len(result) >= 1
    assert all(
        path[0].qualified_name.endswith("run")
        and path[-1].qualified_name.endswith("helper")
        for path in result
    )


async def test_taint_paths_sanitizer_blocks_route(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    # Excluding Calculator.add (an intermediate node on run→helper) should
    # prune the only path in this fixture.
    blocked = await taint_paths(
        pool, branch_id, "run", "helper", sanitizer_names=["add"]
    )
    # Either zero paths or paths that don't traverse Calculator.add.
    for path in blocked:
        assert not any("Calculator.add" in d.qualified_name for d in path)


# ────────────────────────────────────────────────────────────────────
# entrypoints_reaching
# ────────────────────────────────────────────────────────────────────


async def test_entrypoints_reaching_includes_run(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    eps = await entrypoints_reaching(pool, branch_id, "helper")
    qnames = {d.qualified_name for d in eps}
    # `run` is a public entrypoint and reaches helper via Calculator.add.
    assert "main.run" in qnames


async def test_entrypoints_reaching_empty_for_unreachable_target(
    clean_repo, python_fixture_root: Path
):
    pool, repo_id, branch_id = clean_repo
    await _seed(pool, repo_id, branch_id, python_fixture_root)
    eps = await entrypoints_reaching(pool, branch_id, "no_such_target")
    assert eps == []
