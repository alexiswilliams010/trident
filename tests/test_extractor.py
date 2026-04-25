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


# ────────────────────────────────────────────────────────────────────
# Extractor tests (require Postgres)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_incremental_indexing_skips_unchanged(clean_repo, python_fixture_root: Path) -> None:
    pool, repo_id = clean_repo
    first = await index_repo(pool, repo_id, python_fixture_root)
    assert len(first.indexed) > 0
    assert len(first.skipped) == 0

    second = await index_repo(pool, repo_id, python_fixture_root)
    # Second run: every file should be skipped because the content hash matches.
    assert len(second.indexed) == 0
    assert len(second.skipped) == len(first.indexed)
