"""Phase 1 acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.extractor import index_repo
from core.file_walker import WalkConfig, walk_dependency_files, walk_repo


# ────────────────────────────────────────────────────────────────────
# File walker tests (no DB required)
# ────────────────────────────────────────────────────────────────────


def test_walker_yields_python_files(python_fixture_root: Path) -> None:
    cfg = WalkConfig.with_defaults(python_fixture_root)
    found = sorted(d.rel_path for d in walk_repo(cfg))
    assert "mypackage/main.py" in found
    assert "mypackage/utils.py" in found
    assert "mypackage/relative_user.py" in found


def test_walker_skips_dependency_dirs(python_fixture_root: Path) -> None:
    """Files under .venv/ must NEVER be yielded by the initial walk."""
    cfg = WalkConfig.with_defaults(python_fixture_root)
    for d in walk_repo(cfg):
        assert ".venv" not in d.rel_path.split("/"), f"walker leaked into .venv: {d.rel_path}"


def test_walker_skips_solidity_lib_dir(solidity_fixture_root: Path) -> None:
    cfg = WalkConfig.with_defaults(solidity_fixture_root)
    found = [d.rel_path for d in walk_repo(cfg)]
    assert "src/Token.sol" in found
    assert "src/Vault.sol" in found
    # lib/ is a Foundry dependency dir — must be pruned.
    assert all("lib/" not in p for p in found), found


def test_walk_dependency_files_targeted_pass(solidity_fixture_root: Path) -> None:
    """The targeted second pass yields exactly the requested files inside dep dirs."""
    candidates = ["lib/forge-std/src/Test.sol"]
    found = list(walk_dependency_files(solidity_fixture_root, candidates))
    assert len(found) == 1
    assert found[0].rel_path == "lib/forge-std/src/Test.sol"
    assert found[0].language == "solidity"


def test_walk_dependency_files_rejects_path_escape(solidity_fixture_root: Path) -> None:
    """A relative path that escapes repo_root must be silently dropped."""
    found = list(walk_dependency_files(solidity_fixture_root, ["../../../etc/passwd"]))
    assert found == []


def test_walker_yields_go_files(go_fixture_root: Path) -> None:
    cfg = WalkConfig.with_defaults(go_fixture_root)
    found = sorted(d.rel_path for d in walk_repo(cfg))
    assert "cmd/main.go" in found
    assert "internal/utils/utils.go" in found
    assert "pkg/iface/composed.go" in found
    # vendor/ would be pruned if present; go.mod is not a .go file and is
    # ignored by the walker (resolver reads it directly via repos.root_path).
    assert all(p.endswith(".go") for p in found)


def test_walker_yields_node_files(node_fixture_root: Path) -> None:
    """JS, JSX, TS, and TSX files all surface from the walker with the right
    language tag. node_modules/ must be pruned."""
    cfg = WalkConfig.with_defaults(node_fixture_root)
    found = {d.rel_path: d.language for d in walk_repo(cfg)}
    assert found["src/index.js"] == "javascript"
    assert found["src/components/App.jsx"] == "javascript"
    assert found["src/lib.ts"] == "typescript"
    assert found["src/components/Button.tsx"] == "typescript"
    assert found["src/utils/helpers.ts"] == "typescript"
    # node_modules/ is in DEFAULT_DEP_PATHS for both languages.
    assert all("node_modules/" not in p for p in found), sorted(found)
    # package.json / tsconfig.json are not .js/.ts and should not be yielded.
    assert "package.json" not in found
    assert "tsconfig.json" not in found


# ────────────────────────────────────────────────────────────────────
# Extractor tests (require Postgres)
# ────────────────────────────────────────────────────────────────────


async def test_index_python_fixture(clean_repo, python_fixture_root: Path) -> None:
    pool, repo_id = clean_repo
    result = await index_repo(pool, repo_id, python_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "mypackage/main.py" in rel_paths
    assert "mypackage/utils.py" in rel_paths
    assert "mypackage/relative_user.py" in rel_paths
    # __init__.py is empty but should still be parsed.
    assert "mypackage/__init__.py" in rel_paths
    # .venv/dummy.py must not have been parsed.
    assert all(".venv" not in p for p in rel_paths)

    async with pool.acquire() as conn:
        # Every file we parsed has nodes.
        rows = await conn.fetch(
            "SELECT f.path, COUNT(n.id) AS n_count "
            "FROM files f LEFT JOIN nodes n ON n.file_id=f.id "
            "WHERE f.repo_id=$1 GROUP BY f.path",
            repo_id,
        )
        by_path = {r["path"]: r["n_count"] for r in rows}
        assert by_path["mypackage/main.py"] > 30
        assert by_path["mypackage/utils.py"] > 10

        # Root node (module) for main.py should be present.
        main_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path=$2",
            repo_id,
            "mypackage/main.py",
        )
        root_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_id=$1 AND parent_id IS NULL",
            main_id,
        )
        assert root_count == 1

        # Function definitions exist in the parsed CST.
        func_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_id=$1 AND node_type='function_definition'",
            main_id,
        )
        assert func_count >= 3  # __init__, add, double_it, run

        # Leaf nodes have text; named branch nodes do not.
        leaf_with_text = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_id=$1 AND text IS NOT NULL",
            main_id,
        )
        assert leaf_with_text > 0


async def test_index_solidity_fixture(clean_repo, solidity_fixture_root: Path) -> None:
    pool, repo_id = clean_repo
    result = await index_repo(pool, repo_id, solidity_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "src/Token.sol" in rel_paths
    assert "src/Vault.sol" in rel_paths
    # lib/ files must be skipped.
    assert all("lib/" not in p for p in rel_paths)

    async with pool.acquire() as conn:
        # Vault.sol has a contract_declaration.
        vault_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path=$2",
            repo_id,
            "src/Vault.sol",
        )
        contracts = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_id=$1 AND node_type='contract_declaration'",
            vault_id,
        )
        assert contracts == 1

        # Function definitions exist (deposit, withdraw, constructor).
        functions = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_id=$1 AND node_type IN ('function_definition', 'constructor_definition')",
            vault_id,
        )
        assert functions >= 2


async def test_index_go_fixture(clean_repo, go_fixture_root: Path) -> None:
    pool, repo_id = clean_repo
    result = await index_repo(pool, repo_id, go_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "cmd/main.go" in rel_paths
    assert "internal/utils/utils.go" in rel_paths
    assert "pkg/iface/composed.go" in rel_paths

    async with pool.acquire() as conn:
        # index_repo records repo_root so Phase 3 can find go.mod.
        root_path = await conn.fetchval(
            "SELECT root_path FROM repos WHERE id=$1", repo_id,
        )
        assert root_path is not None
        assert root_path.endswith("go_fixture")

        composed_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path=$2",
            repo_id, "pkg/iface/composed.go",
        )
        # interface_type wraps two embedded names + one method_elem in FullIO,
        # plus the two type_elem in ReadWriter — so at least 4 type_elem total.
        type_elems = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_id=$1 AND node_type='type_elem'", composed_id,
        )
        assert type_elems >= 4

        method_elems = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_id=$1 AND node_type='method_elem'", composed_id,
        )
        assert method_elems >= 1


async def test_index_node_fixture(clean_repo, node_fixture_root: Path) -> None:
    pool, repo_id = clean_repo
    result = await index_repo(pool, repo_id, node_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "src/index.js" in rel_paths
    assert "src/lib.ts" in rel_paths
    assert "src/pets.ts" in rel_paths
    assert "src/components/App.jsx" in rel_paths
    assert "src/components/Button.tsx" in rel_paths
    # node_modules/ stays out.
    assert all("node_modules" not in p for p in rel_paths), rel_paths

    async with pool.acquire() as conn:
        # `.tsx` parses via the tsx sub-grammar — the JSX-specific node types
        # should appear in Button.tsx's CST.
        button_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path=$2",
            repo_id, "src/components/Button.tsx",
        )
        jsx_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_id=$1 "
            "AND node_type IN ('jsx_element', 'jsx_self_closing_element', "
            "                  'jsx_opening_element', 'jsx_closing_element')",
            button_id,
        )
        assert jsx_count >= 2, "expected JSX nodes in .tsx file"

        # Same check for .jsx via the JS grammar.
        app_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path=$2",
            repo_id, "src/components/App.jsx",
        )
        jsx_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_id=$1 "
            "AND node_type IN ('jsx_element', 'jsx_self_closing_element', "
            "                  'jsx_opening_element', 'jsx_closing_element')",
            app_id,
        )
        assert jsx_count >= 1, "expected JSX nodes in .jsx file"


async def test_incremental_indexing_skips_unchanged(clean_repo, python_fixture_root: Path) -> None:
    pool, repo_id = clean_repo
    first = await index_repo(pool, repo_id, python_fixture_root)
    assert len(first.indexed) > 0
    assert len(first.skipped) == 0
    assert first.deleted == []

    second = await index_repo(pool, repo_id, python_fixture_root)
    # Second run: every file should be skipped because the content hash matches,
    # and nothing should be pruned (the walker yields the same set both times).
    assert len(second.indexed) == 0
    assert len(second.skipped) == len(first.indexed)
    assert second.deleted == []


async def test_indexing_prunes_moved_and_deleted_files(clean_repo, tmp_path: Path) -> None:
    """A file moved or deleted on disk between runs must have its DB rows
    removed by the next index_repo call. Unchanged siblings must not be
    affected."""
    pool, repo_id = clean_repo

    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n")
    (tmp_path / "moves.py").write_text("def moves():\n    return 2\n")
    (tmp_path / "deleted.py").write_text("def deleted():\n    return 3\n")

    first = await index_repo(pool, repo_id, tmp_path)
    assert {r.rel_path for r in first.indexed} == {"keep.py", "moves.py", "deleted.py"}
    assert first.deleted == []

    # Move moves.py → sub/moved.py and remove deleted.py entirely.
    (tmp_path / "sub").mkdir()
    (tmp_path / "moves.py").rename(tmp_path / "sub" / "moved.py")
    (tmp_path / "deleted.py").unlink()

    second = await index_repo(pool, repo_id, tmp_path)
    # The moved file shows up at its new path; keep.py is unchanged.
    assert {r.rel_path for r in second.indexed} == {"sub/moved.py"}
    assert {r.rel_path for r in second.skipped} == {"keep.py"}
    assert sorted(second.deleted) == ["deleted.py", "moves.py"]

    # DB confirms the old paths are gone and the new path exists.
    async with pool.acquire() as conn:
        remaining = await conn.fetch(
            "SELECT path FROM files WHERE repo_id=$1 ORDER BY path", repo_id,
        )
    assert [r["path"] for r in remaining] == ["keep.py", "sub/moved.py"]
