"""Phase 3 acceptance tests."""

from __future__ import annotations

from pathlib import Path

from core.extractor import index_repo
from core.heuristic_resolver import resolve_branch_imports
from core.semantic_resolver import resolve_repo


async def _full_pipeline(pool, repo_id: int, branch_id: int, root: Path):
    await index_repo(pool, repo_id, branch_id, root)
    await resolve_repo(pool, repo_id, branch_id)
    return await resolve_branch_imports(pool, repo_id, branch_id)


async def _file_version_id(conn, branch_id: int, path: str) -> int | None:
    return await conn.fetchval(
        "SELECT file_version_id FROM branch_files WHERE branch_id=$1 AND path=$2",
        branch_id, path,
    )


# ────────────────────────────────────────────────────────────────────
# Python
# ────────────────────────────────────────────────────────────────────


async def test_python_imports_classified(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, python_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT bf.path, i.import_path, i.dep_class,
                   i.resolved_file_version_id, i.imported_names
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1
            ORDER BY bf.path, i.id
            """,
            branch_id,
        )
        idx = {(r["path"], r["import_path"]): r for r in rows}

        # `from .utils import helper` (relative_user.py)
        rel = idx[("mypackage/relative_user.py", ".utils")]
        assert rel["dep_class"] == "intra_repo"
        utils_fv = await _file_version_id(conn, branch_id, "mypackage/utils.py")
        assert rel["resolved_file_version_id"] == utils_fv
        assert "helper" in rel["imported_names"]

        # `from mypackage.utils import double, helper` (main.py — absolute)
        main_abs = idx[("mypackage/main.py", "mypackage.utils")]
        assert main_abs["dep_class"] == "intra_repo"
        assert main_abs["resolved_file_version_id"] == utils_fv
        assert set(main_abs["imported_names"]) == {"double", "helper"}

        # `import requests` (main.py — external)
        ext = idx[("mypackage/main.py", "requests")]
        assert ext["dep_class"] == "external"

        # external_dependencies row exists for this branch.
        deps = await conn.fetch(
            "SELECT package_name, language FROM external_dependencies WHERE branch_id=$1",
            branch_id,
        )
        assert any(r["package_name"] == "requests" for r in deps)


async def test_python_cross_file_call_edges_upgraded(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, python_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller, callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='main.Calculator.add'
            """,
            branch_id,
        )
        helpers = [r for r in rows if r["callee"] == "utils.helper"]
        assert helpers, f"expected Calculator.add → utils.helper edge, got {rows}"
        assert helpers[0]["confidence"] == "certain"

        rows = await conn.fetch(
            """
            SELECT callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='main.Calculator.double_it'
              AND callee.qualified_name='utils.double'
            """,
            branch_id,
        )
        assert rows and rows[0]["confidence"] == "certain"


async def test_python_cross_file_references_resolved(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, python_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT r.name, target.qualified_name AS target
            FROM "references" r
            JOIN branch_files bf
                ON bf.file_version_id=r.file_version_id AND bf.branch_id=r.branch_id
            JOIN definitions target ON target.id=r.target_def_id
            WHERE r.branch_id=$1 AND bf.path='mypackage/main.py'
              AND r.name='helper' AND target.qualified_name='utils.helper'
            """,
            branch_id,
        )
        assert rows


# ────────────────────────────────────────────────────────────────────
# Solidity
# ────────────────────────────────────────────────────────────────────


async def test_solidity_imports_classified(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, solidity_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT bf.path, i.import_path, i.dep_class
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1
            ORDER BY bf.path, i.id
            """,
            branch_id,
        )
        cls = {(r["path"], r["import_path"]): r["dep_class"] for r in rows}

        assert cls[("src/Vault.sol", "./Token.sol")] == "intra_repo"
        assert cls[("src/External.sol", "./Token.sol")] == "intra_repo"
        assert cls[("src/External.sol", "@openzeppelin/contracts/access/Ownable.sol")] == "external"
        assert cls[("src/External.sol", "lib/forge-std/src/Test.sol")] == "external"
        assert cls[("src/External.sol", "forge-std/Test.sol")] == "external"

        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE branch_id=$1 ORDER BY package_name",
            branch_id,
        )
        names = {r["package_name"] for r in deps}
        assert "@openzeppelin/contracts" in names
        assert "forge-std" in names


async def test_solidity_cross_file_call_edges(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, solidity_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller, callee.qualified_name AS callee,
                   ce.confidence, ce.callee_name
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='Vault.Vault.deposit'
              AND ce.callee_name='transfer'
            """,
            branch_id,
        )
        assert rows
        assert rows[0]["callee"] == "Token.Token.transfer"
        assert rows[0]["confidence"] == "inferred"


# ────────────────────────────────────────────────────────────────────
# Go
# ────────────────────────────────────────────────────────────────────


async def test_go_imports_classified(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, go_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT bf.path, i.import_path, i.dep_class, i.resolved_file_version_id
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1
            ORDER BY bf.path, i.id
            """,
            branch_id,
        )
        cls = {(r["path"], r["import_path"]): r for r in rows}

        intra = cls[("cmd/main.go", "github.com/example/myapp/internal/utils")]
        assert intra["dep_class"] == "intra_repo"
        # The intra-repo import resolves to one of the utils package's files.
        utils_paths = {r["path"] for r in await conn.fetch(
            """
            SELECT bf.path FROM branch_files bf
            WHERE bf.branch_id=$1 AND bf.path LIKE 'internal/utils/%'
            """,
            branch_id,
        )}
        # Look up the resolved file_version's path in this branch.
        resolved_path = await conn.fetchval(
            """
            SELECT bf.path FROM branch_files bf
            WHERE bf.branch_id=$1 AND bf.file_version_id=$2
            """,
            branch_id, intra["resolved_file_version_id"],
        )
        assert resolved_path in utils_paths

        assert cls[("cmd/main.go", "fmt")]["dep_class"] == "external"
        assert cls[("cmd/main.go", "github.com/pkg/errors")]["dep_class"] == "external"

        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE branch_id=$1 AND language='go'",
            branch_id,
        )
        names = {r["package_name"] for r in deps}
        assert "fmt" in names
        assert "github.com/pkg/errors" in names


async def test_go_cross_file_call_edges_upgraded(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, go_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller,
                   callee.qualified_name AS callee,
                   ce.confidence, ce.callee_name
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1
              AND caller.qualified_name='main.Calculator.DoubleIt'
              AND ce.callee_name='Double'
            """,
            branch_id,
        )
        assert rows, "expected DoubleIt → Double edge"
        assert rows[0]["callee"] == "utils.Double"


async def test_go_method_qualified_name_uses_receiver(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, go_fixture_root)

    async with pool.acquire() as conn:
        names = await conn.fetch(
            """
            SELECT qualified_name FROM definitions d
            JOIN branch_files bf ON bf.file_version_id=d.file_version_id
            WHERE bf.branch_id=$1 AND d.kind='method' AND d.name='Add'
            """,
            branch_id,
        )
        assert any(r["qualified_name"] == "main.Calculator.Add" for r in names), [r["qualified_name"] for r in names]


async def test_go_multi_name_var_emits_two_defs(clean_repo, go_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, go_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.name FROM definitions d
            JOIN branch_files bf ON bf.file_version_id=d.file_version_id
            WHERE bf.branch_id=$1 AND d.kind='var' AND bf.path='internal/utils/utils.go'
            """,
            branch_id,
        )
        names = {r["name"] for r in rows}
        assert {"X", "Y"} <= names, names


# ────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript
# ────────────────────────────────────────────────────────────────────


async def test_node_imports_classified(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, node_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT bf.path, i.import_path, i.dep_class,
                   rbf.path AS resolved_path
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            LEFT JOIN branch_files rbf
                ON rbf.file_version_id=i.resolved_file_version_id AND rbf.branch_id=i.branch_id
            WHERE i.branch_id=$1
            """,
            branch_id,
        )
        idx = {(r["path"], r["import_path"]): r for r in rows}

        rel_lib = idx[("src/index.js", "./lib")]
        assert rel_lib["dep_class"] == "intra_repo"
        assert rel_lib["resolved_path"] == "src/lib.ts"

        rel_pets = idx[("src/index.js", "./pets")]
        assert rel_pets["dep_class"] == "intra_repo"
        assert rel_pets["resolved_path"] == "src/pets.ts"

        rel_button = idx[("src/components/App.jsx", "./Button")]
        assert rel_button["dep_class"] == "intra_repo"
        assert rel_button["resolved_path"] == "src/components/Button.tsx"

        alias = idx[("src/pets.ts", "@app/lib")]
        assert alias["dep_class"] == "intra_repo"
        assert alias["resolved_path"] == "src/lib.ts"
        alias2 = idx[("src/pets.ts", "@app/utils/helpers")]
        assert alias2["resolved_path"] == "src/utils/helpers.ts"

        ext_react = idx[("src/components/Button.tsx", "react")]
        assert ext_react["dep_class"] == "external"

        req_leftpad = idx[("src/index.js", "leftpad")]
        assert req_leftpad["dep_class"] == "external"

        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE branch_id=$1",
            branch_id,
        )
        names = {r["package_name"] for r in deps}
        assert "react" in names
        assert "leftpad" in names
        assert "@scoped/pkg" in names


async def test_node_bare_specifier_subpath_rollup(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, node_fixture_root)

    async with pool.acquire() as conn:
        n_react = await conn.fetchval(
            "SELECT COUNT(*) FROM external_dependencies "
            "WHERE branch_id=$1 AND package_name='react'",
            branch_id,
        )
        assert n_react == 1


async def test_node_cross_file_call_edges(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, node_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller,
                   callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='index.run'
              AND ce.callee_name='describeAll'
            """,
            branch_id,
        )
        assert rows, "expected run → describeAll call edge"
        assert rows[0]["callee"] == "helpers.describeAll"
        assert rows[0]["confidence"] == "certain"


async def test_node_cross_file_references_resolved(
    clean_repo, node_fixture_root: Path,
):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, node_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT target.qualified_name AS target
            FROM "references" r
            JOIN branch_files bf
                ON bf.file_version_id=r.file_version_id AND bf.branch_id=r.branch_id
            LEFT JOIN definitions target ON target.id=r.target_def_id
            WHERE r.branch_id=$1 AND bf.path='src/index.js' AND r.name='Animal'
              AND target.qualified_name='lib.Animal'
            """,
            branch_id,
        )
        assert rows, "expected `Animal` reference in index.js to resolve to lib.Animal"


# ────────────────────────────────────────────────────────────────────
# Rust
# ────────────────────────────────────────────────────────────────────


async def test_rust_imports_classified(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT bf.path, i.import_path, i.dep_class,
                   i.resolved_file_version_id, i.imported_names
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1
            ORDER BY bf.path, i.id
            """,
            branch_id,
        )
        idx = {(r["path"], r["import_path"]): r for r in rows}

        utils_fv = await _file_version_id(conn, branch_id, "src/utils.rs")

        rel = idx[("src/relative_user.rs", "super::utils::helper")]
        assert rel["dep_class"] == "intra_repo"
        assert rel["resolved_file_version_id"] == utils_fv
        assert "helper" in rel["imported_names"]

        for path in ("myapp::utils::double", "myapp::utils::helper"):
            row = idx[("src/main.rs", path)]
            assert row["dep_class"] == "intra_repo"
            assert row["resolved_file_version_id"] == utils_fv

        ext = idx[("src/main.rs", "serde::Serialize")]
        assert ext["dep_class"] == "external"

        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE branch_id=$1",
            branch_id,
        )
        assert any(r["package_name"] == "serde" for r in deps)


async def test_rust_cross_file_call_edges_upgraded(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='main.run'
            ORDER BY callee.qualified_name
            """,
            branch_id,
        )
        callees = {(r["callee"], r["confidence"]) for r in rows}
        assert ("utils.double", "certain") in callees
        assert ("utils.helper", "certain") in callees


async def test_rust_impl_method_qualified_names(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT qualified_name FROM definitions d
            JOIN branch_files bf ON bf.file_version_id=d.file_version_id
            WHERE bf.branch_id=$1 AND d.kind='function'
              AND d.qualified_name LIKE 'utils.Counter.%'
            ORDER BY qualified_name
            """,
            branch_id,
        )
        names = {r["qualified_name"] for r in rows}
        assert names == {"utils.Counter.new", "utils.Counter.increment"}


async def test_rust_workspace_cross_crate_imports(clean_repo, rust_workspace_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_workspace_fixture_root)

    async with pool.acquire() as conn:
        helpers_fv = await _file_version_id(conn, branch_id, "programs/core/src/helpers.rs")
        state_fv = await _file_version_id(conn, branch_id, "programs/core/src/state.rs")
        assert helpers_fv is not None and state_fv is not None

        rows = await conn.fetch(
            """
            SELECT i.import_path, i.dep_class, i.resolved_file_version_id
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1 AND bf.path='programs/app/src/runner.rs'
            ORDER BY i.id
            """,
            branch_id,
        )
        idx = {r["import_path"]: r for r in rows}

        helpers_row = idx["core_lib::helpers::shared"]
        assert helpers_row["dep_class"] == "intra_repo"
        assert helpers_row["resolved_file_version_id"] == helpers_fv

        state_row = idx["core_lib::state::Counter"]
        assert state_row["dep_class"] == "intra_repo"
        assert state_row["resolved_file_version_id"] == state_fv

        ext = idx["serde::Serialize"]
        assert ext["dep_class"] == "external"


async def test_rust_workspace_relative_within_crate(clean_repo, rust_workspace_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_workspace_fixture_root)

    async with pool.acquire() as conn:
        runner_fv = await _file_version_id(conn, branch_id, "programs/app/src/runner.rs")
        row = await conn.fetchrow(
            """
            SELECT i.dep_class, i.resolved_file_version_id
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1 AND bf.path='programs/app/src/nested/deep.rs'
              AND i.import_path='super::super::runner'
            """,
            branch_id,
        )
        assert row is not None
        assert row["dep_class"] == "intra_repo"
        assert row["resolved_file_version_id"] == runner_fv


async def test_rust_workspace_crate_paths_resolve(clean_repo, rust_workspace_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_workspace_fixture_root)

    async with pool.acquire() as conn:
        deep_fv = await _file_version_id(conn, branch_id, "programs/app/src/nested/deep.rs")
        row = await conn.fetchrow(
            """
            SELECT i.dep_class, i.resolved_file_version_id
            FROM imports i
            JOIN branch_files bf
                ON bf.file_version_id=i.file_version_id AND bf.branch_id=i.branch_id
            WHERE i.branch_id=$1 AND bf.path='programs/app/src/runner.rs'
              AND i.import_path='crate::nested::deep::nested_helper'
            """,
            branch_id,
        )
        assert row is not None
        assert row["dep_class"] == "intra_repo"
        assert row["resolved_file_version_id"] == deep_fv


async def test_rust_derive_emits_inferred_inherits(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await _full_pipeline(pool, repo_id, branch_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.qualified_name AS child, ie.base_name, ie.confidence
            FROM inherits_edges ie
            JOIN definitions d ON d.id=ie.child_def_id
            WHERE ie.branch_id=$1
            ORDER BY child, base_name
            """,
            branch_id,
        )
        edges = {(r["child"], r["base_name"], r["confidence"]) for r in rows}
        assert ("utils.Counter", "Clone", "inferred") in edges
        assert ("utils.Counter", "Debug", "inferred") in edges
        assert ("utils.Counter", "Default", "inferred") in edges
        assert ("main.Report", "Serialize", "inferred") in edges
