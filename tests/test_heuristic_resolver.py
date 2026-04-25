"""Phase 3 acceptance tests."""

from __future__ import annotations

from pathlib import Path

from core.extractor import index_repo
from core.heuristic_resolver import resolve_repo_imports
from core.semantic_resolver import resolve_repo


async def _full_pipeline(pool, repo_id: int, root: Path):
    await index_repo(pool, repo_id, root)
    await resolve_repo(pool, repo_id)
    return await resolve_repo_imports(pool, repo_id)


# ────────────────────────────────────────────────────────────────────
# Python
# ────────────────────────────────────────────────────────────────────


async def test_python_imports_classified(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    stats = await _full_pipeline(pool, repo_id, python_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT f.path, i.import_path, i.dep_class, i.resolved_file_id, i.imported_names "
            "FROM imports i JOIN files f ON f.id=i.file_id "
            "WHERE f.repo_id=$1 ORDER BY f.path, i.id",
            repo_id,
        )
        idx = {(r["path"], r["import_path"]): r for r in rows}

        # `from .utils import helper` (relative_user.py)
        rel = idx[("mypackage/relative_user.py", ".utils")]
        assert rel["dep_class"] == "intra_repo"
        # The resolved file should be utils.py.
        utils_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path='mypackage/utils.py'",
            repo_id,
        )
        assert rel["resolved_file_id"] == utils_id
        assert "helper" in rel["imported_names"]

        # `from mypackage.utils import double, helper` (main.py — absolute)
        main_abs = idx[("mypackage/main.py", "mypackage.utils")]
        assert main_abs["dep_class"] == "intra_repo"
        assert main_abs["resolved_file_id"] == utils_id
        assert set(main_abs["imported_names"]) == {"double", "helper"}

        # `import requests` (main.py — external)
        ext = idx[("mypackage/main.py", "requests")]
        assert ext["dep_class"] == "external"

        # external_dependencies row exists.
        deps = await conn.fetch(
            "SELECT package_name, language FROM external_dependencies WHERE repo_id=$1",
            repo_id,
        )
        assert any(r["package_name"] == "requests" for r in deps)


async def test_python_cross_file_call_edges_upgraded(clean_repo, python_fixture_root: Path):
    """Phase 2 left `Calculator.add → helper(...)` as uncertain; Phase 3
    should re-link it to utils.helper with confidence='certain'."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, python_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller, callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='main.Calculator.add'
            """,
            repo_id,
        )
        # Should now contain a (caller, helper) edge with certain confidence.
        helpers = [r for r in rows if r["callee"] == "utils.helper"]
        assert helpers, f"expected Calculator.add → utils.helper edge, got {rows}"
        assert helpers[0]["confidence"] == "certain"

        # Calculator.double_it → utils.double should also be linked.
        rows = await conn.fetch(
            """
            SELECT callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='main.Calculator.double_it'
              AND callee.qualified_name='utils.double'
            """,
            repo_id,
        )
        assert rows and rows[0]["confidence"] == "certain"


async def test_python_cross_file_references_resolved(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, python_fixture_root)

    async with pool.acquire() as conn:
        # The `helper` reference at the call site in main.py should now resolve
        # cross-file to utils.helper.
        rows = await conn.fetch(
            """
            SELECT r.name, target.qualified_name AS target
            FROM "references" r
            JOIN files f ON f.id=r.file_id
            JOIN definitions target ON target.id=r.target_def_id
            WHERE f.repo_id=$1 AND f.path='mypackage/main.py'
              AND r.name='helper' AND target.qualified_name='utils.helper'
            """,
            repo_id,
        )
        assert rows


# ────────────────────────────────────────────────────────────────────
# Solidity
# ────────────────────────────────────────────────────────────────────


async def test_solidity_imports_classified(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, solidity_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT f.path, i.import_path, i.dep_class "
            "FROM imports i JOIN files f ON f.id=i.file_id "
            "WHERE f.repo_id=$1 ORDER BY f.path, i.id",
            repo_id,
        )
        cls = {(r["path"], r["import_path"]): r["dep_class"] for r in rows}

        # Vault.sol relative import to Token.sol
        assert cls[("src/Vault.sol", "./Token.sol")] == "intra_repo"

        # External.sol mix
        assert cls[("src/External.sol", "./Token.sol")] == "intra_repo"
        assert cls[("src/External.sol", "@openzeppelin/contracts/access/Ownable.sol")] == "external"
        assert cls[("src/External.sol", "lib/forge-std/src/Test.sol")] == "external"
        assert cls[("src/External.sol", "forge-std/Test.sol")] == "external"

        # external_dependencies populated
        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE repo_id=$1 ORDER BY package_name",
            repo_id,
        )
        names = {r["package_name"] for r in deps}
        assert "@openzeppelin/contracts" in names
        assert "forge-std" in names


async def test_solidity_cross_file_call_edges(clean_repo, solidity_fixture_root: Path):
    """Vault.deposit calls `token.transfer(...)`. `transfer` isn't itself an
    imported_name (`{Token}` is), but `transfer` resolves uniquely to
    Token.transfer within Vault.sol's set of imported files → tier-B inferred
    cross-file link.
    """
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, solidity_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller, callee.qualified_name AS callee,
                   ce.confidence, ce.callee_name
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='Vault.Vault.deposit'
              AND ce.callee_name='transfer'
            """,
            repo_id,
        )
        assert rows
        assert rows[0]["callee"] == "Token.Token.transfer"
        assert rows[0]["confidence"] == "inferred"
