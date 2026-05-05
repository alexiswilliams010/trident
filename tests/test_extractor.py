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
    assert all(p.endswith(".go") for p in found)


def test_walker_yields_rust_files(rust_fixture_root: Path) -> None:
    cfg = WalkConfig.with_defaults(rust_fixture_root)
    found = sorted(d.rel_path for d in walk_repo(cfg))
    assert "src/lib.rs" in found
    assert "src/main.rs" in found
    assert "src/utils.rs" in found
    assert "src/relative_user.rs" in found
    assert all(p.endswith(".rs") for p in found)


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
    assert all("node_modules/" not in p for p in found), sorted(found)
    assert "package.json" not in found
    assert "tsconfig.json" not in found


# ────────────────────────────────────────────────────────────────────
# Extractor tests (require Postgres)
# ────────────────────────────────────────────────────────────────────


async def _file_version_id_for_path(conn, branch_id: int, path: str) -> int:
    """Helper: look up the file_version mapped at `path` in this branch."""
    return await conn.fetchval(
        "SELECT file_version_id FROM branch_files WHERE branch_id=$1 AND path=$2",
        branch_id, path,
    )


async def test_index_python_fixture(clean_repo, python_fixture_root: Path) -> None:
    pool, repo_id, branch_id = clean_repo
    result = await index_repo(pool, repo_id, branch_id, python_fixture_root)

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
            """
            SELECT bf.path, COUNT(n.id) AS n_count
            FROM branch_files bf
            JOIN file_versions fv ON fv.id = bf.file_version_id
            LEFT JOIN nodes n ON n.file_version_id = fv.id
            WHERE bf.branch_id=$1
            GROUP BY bf.path
            """,
            branch_id,
        )
        by_path = {r["path"]: r["n_count"] for r in rows}
        assert by_path["mypackage/main.py"] > 30
        assert by_path["mypackage/utils.py"] > 10

        main_id = await _file_version_id_for_path(conn, branch_id, "mypackage/main.py")
        # Root node (module) for main.py should be present.
        root_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 AND parent_id IS NULL",
            main_id,
        )
        assert root_count == 1

        # Function definitions exist in the parsed CST.
        func_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_version_id=$1 AND node_type='function_definition'",
            main_id,
        )
        assert func_count >= 3  # __init__, add, double_it, run

        # Leaf nodes have text; named branch nodes do not.
        leaf_with_text = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 AND text IS NOT NULL",
            main_id,
        )
        assert leaf_with_text > 0


async def test_index_solidity_fixture(clean_repo, solidity_fixture_root: Path) -> None:
    pool, repo_id, branch_id = clean_repo
    result = await index_repo(pool, repo_id, branch_id, solidity_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "src/Token.sol" in rel_paths
    assert "src/Vault.sol" in rel_paths
    assert all("lib/" not in p for p in rel_paths)

    async with pool.acquire() as conn:
        vault_id = await _file_version_id_for_path(conn, branch_id, "src/Vault.sol")
        contracts = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_version_id=$1 AND node_type='contract_declaration'",
            vault_id,
        )
        assert contracts == 1

        functions = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_version_id=$1 AND node_type IN ('function_definition', 'constructor_definition')",
            vault_id,
        )
        assert functions >= 2


async def test_index_go_fixture(clean_repo, go_fixture_root: Path) -> None:
    pool, repo_id, branch_id = clean_repo
    result = await index_repo(pool, repo_id, branch_id, go_fixture_root)

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

        composed_id = await _file_version_id_for_path(conn, branch_id, "pkg/iface/composed.go")
        type_elems = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_version_id=$1 AND node_type='type_elem'", composed_id,
        )
        assert type_elems >= 4

        method_elems = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes "
            "WHERE file_version_id=$1 AND node_type='method_elem'", composed_id,
        )
        assert method_elems >= 1


async def test_index_node_fixture(clean_repo, node_fixture_root: Path) -> None:
    pool, repo_id, branch_id = clean_repo
    result = await index_repo(pool, repo_id, branch_id, node_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "src/index.js" in rel_paths
    assert "src/lib.ts" in rel_paths
    assert "src/pets.ts" in rel_paths
    assert "src/components/App.jsx" in rel_paths
    assert "src/components/Button.tsx" in rel_paths
    assert all("node_modules" not in p for p in rel_paths), rel_paths

    async with pool.acquire() as conn:
        button_id = await _file_version_id_for_path(conn, branch_id, "src/components/Button.tsx")
        jsx_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 "
            "AND node_type IN ('jsx_element', 'jsx_self_closing_element', "
            "                  'jsx_opening_element', 'jsx_closing_element')",
            button_id,
        )
        assert jsx_count >= 2, "expected JSX nodes in .tsx file"

        app_id = await _file_version_id_for_path(conn, branch_id, "src/components/App.jsx")
        jsx_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 "
            "AND node_type IN ('jsx_element', 'jsx_self_closing_element', "
            "                  'jsx_opening_element', 'jsx_closing_element')",
            app_id,
        )
        assert jsx_count >= 1, "expected JSX nodes in .jsx file"


async def test_index_rust_fixture(clean_repo, rust_fixture_root: Path) -> None:
    pool, repo_id, branch_id = clean_repo
    result = await index_repo(pool, repo_id, branch_id, rust_fixture_root)

    rel_paths = {r.rel_path for r in result.indexed}
    assert "src/lib.rs" in rel_paths
    assert "src/utils.rs" in rel_paths
    assert "src/main.rs" in rel_paths
    assert "src/relative_user.rs" in rel_paths
    assert all(not p.startswith("target/") for p in rel_paths)

    async with pool.acquire() as conn:
        utils_id = await _file_version_id_for_path(conn, branch_id, "src/utils.rs")
        struct_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 AND node_type='struct_item'",
            utils_id,
        )
        impl_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 AND node_type='impl_item'",
            utils_id,
        )
        attr_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1 AND node_type='attribute_item'",
            utils_id,
        )
        assert struct_count == 1
        assert impl_count == 1
        assert attr_count >= 1


async def test_incremental_indexing_skips_unchanged(clean_repo, python_fixture_root: Path) -> None:
    pool, repo_id, branch_id = clean_repo
    first = await index_repo(pool, repo_id, branch_id, python_fixture_root)
    assert len(first.indexed) > 0
    assert len(first.skipped) == 0
    assert first.deleted == []

    second = await index_repo(pool, repo_id, branch_id, python_fixture_root)
    # Second run: every file's content_hash already maps in branch_files, so
    # the indexer skips parsing entirely.
    assert len(second.indexed) == 0
    assert len(second.skipped) == len(first.indexed)
    assert second.deleted == []


async def test_in_place_modify_then_revert_reuses_file_version(clean_repo, tmp_path: Path) -> None:
    """Edit a file in place, re-index → new file_version is created. Revert the
    file to its original content, re-index → the original file_version row is
    reused (no new parse), and the path's mapping points back at it.

    This exercises the content-hash dedup path: file_versions are keyed by
    (repo_id, content_hash), so reverting to a previously seen content gets
    the existing row for free without re-emitting nodes / definitions.
    """
    pool, repo_id, branch_id = clean_repo

    src = tmp_path / "module.py"
    original = "def hello():\n    return 1\n"
    modified = "def hello():\n    return 2\n"
    src.write_text(original)

    # Initial index → one file_version, one branch_files mapping.
    await index_repo(pool, repo_id, branch_id, tmp_path)
    async with pool.acquire() as conn:
        fv_initial = await conn.fetchval(
            "SELECT file_version_id FROM branch_files WHERE branch_id=$1 AND path=$2",
            branch_id, "module.py",
        )
        fv_count_initial = await conn.fetchval(
            "SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id,
        )
    assert fv_count_initial == 1

    # Modify in place → re-index. A new file_version row appears (different
    # content_hash); the branch_files mapping repoints to it.
    src.write_text(modified)
    await index_repo(pool, repo_id, branch_id, tmp_path)
    async with pool.acquire() as conn:
        fv_after_modify = await conn.fetchval(
            "SELECT file_version_id FROM branch_files WHERE branch_id=$1 AND path=$2",
            branch_id, "module.py",
        )
        fv_count_after_modify = await conn.fetchval(
            "SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id,
        )
    assert fv_after_modify != fv_initial, "modified content should map to a new file_version"
    assert fv_count_after_modify == 2, "new content_hash adds a second file_version row"

    # Revert in place → re-index. The original content_hash hits the existing
    # file_version (kept across the modify pass because gc is a separate step),
    # so no new row is created and the mapping points back at fv_initial.
    src.write_text(original)
    await index_repo(pool, repo_id, branch_id, tmp_path)
    async with pool.acquire() as conn:
        fv_after_revert = await conn.fetchval(
            "SELECT file_version_id FROM branch_files WHERE branch_id=$1 AND path=$2",
            branch_id, "module.py",
        )
        fv_count_after_revert = await conn.fetchval(
            "SELECT COUNT(*) FROM file_versions WHERE repo_id=$1", repo_id,
        )
    assert fv_after_revert == fv_initial, "revert should reuse the original file_version row"
    assert fv_count_after_revert == 2, (
        "revert reuses an existing row; the modified file_version is now an orphan "
        "but lingers until `make gc` reclaims it"
    )

    # Sanity: the original's nodes are still attached to the original file_version.
    async with pool.acquire() as conn:
        n_count = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes WHERE file_version_id=$1", fv_initial,
        )
    assert n_count > 0


async def test_indexing_prunes_moved_and_deleted_files(clean_repo, tmp_path: Path) -> None:
    """A file moved or deleted on disk between runs must have its branch_files
    mapping removed. Shared content (file_versions) is preserved — only `gc`
    reclaims unreferenced ones."""
    pool, repo_id, branch_id = clean_repo

    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n")
    (tmp_path / "moves.py").write_text("def moves():\n    return 2\n")
    (tmp_path / "deleted.py").write_text("def deleted():\n    return 3\n")

    first = await index_repo(pool, repo_id, branch_id, tmp_path)
    assert {r.rel_path for r in first.indexed} == {"keep.py", "moves.py", "deleted.py"}
    assert first.deleted == []

    # Move moves.py → sub/moved.py and remove deleted.py entirely.
    (tmp_path / "sub").mkdir()
    (tmp_path / "moves.py").rename(tmp_path / "sub" / "moved.py")
    (tmp_path / "deleted.py").unlink()

    second = await index_repo(pool, repo_id, branch_id, tmp_path)
    # The moved file shows up at its new path; keep.py is unchanged.
    # moves.py's content shows up at sub/moved.py — same content_hash hits the
    # existing file_version so the indexer reports it as skipped (no parse).
    indexed_paths = {r.rel_path for r in second.indexed}
    skipped_paths = {r.rel_path for r in second.skipped}
    assert "sub/moved.py" in (indexed_paths | skipped_paths)
    assert "keep.py" in skipped_paths
    assert sorted(second.deleted) == ["deleted.py", "moves.py"]

    # branch_files confirms the old paths are gone and the new path exists.
    async with pool.acquire() as conn:
        remaining = await conn.fetch(
            "SELECT path FROM branch_files WHERE branch_id=$1 ORDER BY path", branch_id,
        )
    assert [r["path"] for r in remaining] == ["keep.py", "sub/moved.py"]
