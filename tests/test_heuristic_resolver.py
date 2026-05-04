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


# ────────────────────────────────────────────────────────────────────
# Go
# ────────────────────────────────────────────────────────────────────


async def test_go_imports_classified(clean_repo, go_fixture_root: Path):
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, go_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT f.path, i.import_path, i.dep_class, i.resolved_file_id "
            "FROM imports i JOIN files f ON f.id=i.file_id "
            "WHERE f.repo_id=$1 ORDER BY f.path, i.id",
            repo_id,
        )
        cls = {(r["path"], r["import_path"]): r for r in rows}

        # Intra-repo: cmd/main.go imports github.com/example/myapp/internal/utils.
        intra = cls[("cmd/main.go", "github.com/example/myapp/internal/utils")]
        assert intra["dep_class"] == "intra_repo"
        utils_files = {r["path"] for r in await conn.fetch(
            "SELECT path FROM files WHERE repo_id=$1 AND path LIKE 'internal/utils/%'",
            repo_id,
        )}
        resolved = await conn.fetchval(
            "SELECT path FROM files WHERE id=$1", intra["resolved_file_id"],
        )
        assert resolved in utils_files

        # Stdlib: fmt → external, package_name=fmt.
        assert cls[("cmd/main.go", "fmt")]["dep_class"] == "external"

        # Third-party: github.com/pkg/errors → external, package_name=github.com/pkg/errors.
        assert cls[("cmd/main.go", "github.com/pkg/errors")]["dep_class"] == "external"

        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE repo_id=$1 AND language='go'",
            repo_id,
        )
        names = {r["package_name"] for r in deps}
        assert "fmt" in names
        assert "github.com/pkg/errors" in names


async def test_go_cross_file_call_edges_upgraded(clean_repo, go_fixture_root: Path):
    """Calculator.DoubleIt calls utils.Double — the selector resolves cross-file
    via the imported `utils` package and lands as a certain edge.
    """
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, go_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller,
                   callee.qualified_name AS callee,
                   ce.confidence, ce.callee_name
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1
              AND caller.qualified_name='main.Calculator.DoubleIt'
              AND ce.callee_name='Double'
            """,
            repo_id,
        )
        assert rows, "expected DoubleIt → Double edge"
        assert rows[0]["callee"] == "utils.Double"


async def test_go_method_qualified_name_uses_receiver(clean_repo, go_fixture_root: Path):
    """`func (c *Calculator) Add(...)` should produce qualified_name
    `<file>.Calculator.Add`, not `<file>.Add`."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, go_fixture_root)

    async with pool.acquire() as conn:
        names = await conn.fetch(
            """
            SELECT qualified_name FROM definitions d
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND d.kind='method' AND d.name='Add'
            """,
            repo_id,
        )
        assert any(r["qualified_name"] == "main.Calculator.Add" for r in names), [r["qualified_name"] for r in names]


async def test_go_multi_name_var_emits_two_defs(clean_repo, go_fixture_root: Path):
    """`var X, Y int` in utils.go should produce two `var` definitions."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, go_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.name FROM definitions d
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND d.kind='var' AND f.path='internal/utils/utils.go'
            """,
            repo_id,
        )
        names = {r["name"] for r in rows}
        assert {"X", "Y"} <= names, names


# ────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript
# ────────────────────────────────────────────────────────────────────


async def test_node_imports_classified(clean_repo, node_fixture_root: Path):
    """Coverage matrix:
       - relative import with extension probing (.ts, .tsx, .jsx)
       - tsconfig `paths` alias `@app/*` rewritten to `src/*`
       - bare specifier with subpath rolled up to package_name (`react`)
       - scoped package external (`@scoped/pkg`)
       - CommonJS `require("…")` resolution + classification
    """
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, node_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT f.path, i.import_path, i.dep_class,
                   resolved.path AS resolved_path
            FROM imports i JOIN files f ON f.id=i.file_id
            LEFT JOIN files resolved ON resolved.id=i.resolved_file_id
            WHERE f.repo_id=$1
            """,
            repo_id,
        )
        idx = {(r["path"], r["import_path"]): r for r in rows}

        # 1) ESM relative import where the source is a .ts file: extension
        # probing must reach `src/lib.ts`.
        rel_lib = idx[("src/index.js", "./lib")]
        assert rel_lib["dep_class"] == "intra_repo"
        assert rel_lib["resolved_path"] == "src/lib.ts"

        rel_pets = idx[("src/index.js", "./pets")]
        assert rel_pets["dep_class"] == "intra_repo"
        assert rel_pets["resolved_path"] == "src/pets.ts"

        # 2) JSX → TSX cross-language relative.
        rel_button = idx[("src/components/App.jsx", "./Button")]
        assert rel_button["dep_class"] == "intra_repo"
        assert rel_button["resolved_path"] == "src/components/Button.tsx"

        # 3) tsconfig path alias.
        alias = idx[("src/pets.ts", "@app/lib")]
        assert alias["dep_class"] == "intra_repo"
        assert alias["resolved_path"] == "src/lib.ts"
        alias2 = idx[("src/pets.ts", "@app/utils/helpers")]
        assert alias2["resolved_path"] == "src/utils/helpers.ts"

        # 4) Bare specifiers — external. `react` from Button.tsx.
        ext_react = idx[("src/components/Button.tsx", "react")]
        assert ext_react["dep_class"] == "external"

        # 5) CommonJS require. Even though leftpad lives under node_modules/
        # (not indexed in v1), the import row exists and is classified
        # external with package_name=leftpad.
        req_leftpad = idx[("src/index.js", "leftpad")]
        assert req_leftpad["dep_class"] == "external"

        # 6) Scoped package: package_name should roll up to `@scoped/pkg`,
        # not just the first segment.
        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE repo_id=$1",
            repo_id,
        )
        names = {r["package_name"] for r in deps}
        assert "react" in names
        assert "leftpad" in names
        assert "@scoped/pkg" in names


async def test_node_bare_specifier_subpath_rollup(clean_repo, node_fixture_root: Path):
    """`react/jsx-runtime` and `react` should share one external_dependencies
    row keyed by `react`. (Synthesizes the scenario via direct INSERT to keep
    the fixture small.)"""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, node_fixture_root)

    async with pool.acquire() as conn:
        # One row per package_name regardless of subpath usage.
        n_react = await conn.fetchval(
            "SELECT COUNT(*) FROM external_dependencies "
            "WHERE repo_id=$1 AND package_name='react'",
            repo_id,
        )
        assert n_react == 1


async def test_node_cross_file_call_edges(clean_repo, node_fixture_root: Path):
    """`describeAll([...])` in index.js resolves to `helpers.describeAll`
    cross-file. Tier-A direct hit: `describeAll` is in `imported_names` of
    the `./utils/helpers` import row, so the cross-file linker upgrades the
    call-edge confidence from uncertain to certain."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, node_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT caller.qualified_name AS caller,
                   callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='index.run'
              AND ce.callee_name='describeAll'
            """,
            repo_id,
        )
        assert rows, "expected run → describeAll call edge"
        assert rows[0]["callee"] == "helpers.describeAll"
        assert rows[0]["confidence"] == "certain"


async def test_node_cross_file_references_resolved(
    clean_repo, node_fixture_root: Path,
):
    """Tier-A direct linking should also upgrade plain identifier references:
    `new Animal(...)` is a `new_expression` (not a call), but the inner
    `Animal` identifier is captured as a reference and gets re-targeted."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, node_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT target.qualified_name AS target
            FROM "references" r
            JOIN files f ON f.id=r.file_id
            LEFT JOIN definitions target ON target.id=r.target_def_id
            WHERE f.repo_id=$1 AND f.path='src/index.js' AND r.name='Animal'
              AND target.qualified_name='lib.Animal'
            """,
            repo_id,
        )
        assert rows, "expected `Animal` reference in index.js to resolve to lib.Animal"


# ────────────────────────────────────────────────────────────────────
# Rust
# ────────────────────────────────────────────────────────────────────


async def test_rust_imports_classified(clean_repo, rust_fixture_root: Path):
    """Three intra-repo paths (one `super::`, two `myapp::`) all resolve to
    utils.rs; one external (`serde`) gets a row in external_dependencies."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT f.path, i.import_path, i.dep_class, i.resolved_file_id, i.imported_names "
            "FROM imports i JOIN files f ON f.id=i.file_id "
            "WHERE f.repo_id=$1 ORDER BY f.path, i.id",
            repo_id,
        )
        idx = {(r["path"], r["import_path"]): r for r in rows}

        utils_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path='src/utils.rs'",
            repo_id,
        )

        # `use super::utils::helper` from relative_user.rs — relative resolution.
        rel = idx[("src/relative_user.rs", "super::utils::helper")]
        assert rel["dep_class"] == "intra_repo"
        assert rel["resolved_file_id"] == utils_id
        assert "helper" in rel["imported_names"]

        # `use myapp::utils::{double, helper}` from main.rs flattens to two
        # entries (one per leaf) — both should resolve to utils.rs.
        for path in ("myapp::utils::double", "myapp::utils::helper"):
            row = idx[("src/main.rs", path)]
            assert row["dep_class"] == "intra_repo"
            assert row["resolved_file_id"] == utils_id

        # `use serde::Serialize` — external.
        ext = idx[("src/main.rs", "serde::Serialize")]
        assert ext["dep_class"] == "external"

        deps = await conn.fetch(
            "SELECT package_name FROM external_dependencies WHERE repo_id=$1",
            repo_id,
        )
        assert any(r["package_name"] == "serde" for r in deps)


async def test_rust_cross_file_call_edges_upgraded(clean_repo, rust_fixture_root: Path):
    """`run()` in main.rs calls `helper(1)` and `double(2)` — both imported
    from utils.rs. After Phase 3 the call_edges should point at utils.helper
    / utils.double with confidence='certain' (Tier-A direct linking)."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT callee.qualified_name AS callee, ce.confidence
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            JOIN files f ON f.id=caller.file_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE f.repo_id=$1 AND caller.qualified_name='main.run'
            ORDER BY callee.qualified_name
            """,
            repo_id,
        )
        callees = {(r["callee"], r["confidence"]) for r in rows}
        assert ("utils.double", "certain") in callees
        assert ("utils.helper", "certain") in callees


async def test_rust_impl_method_qualified_names(clean_repo, rust_fixture_root: Path):
    """Methods defined inside `impl Counter { … }` should pick up `Counter`
    as their qualified-name prefix despite impl_item not being a definition."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT qualified_name FROM definitions d JOIN files f ON f.id=d.file_id "
            "WHERE f.repo_id=$1 AND d.kind='function' "
            "  AND d.qualified_name LIKE 'utils.Counter.%' "
            "ORDER BY qualified_name",
            repo_id,
        )
        names = {r["qualified_name"] for r in rows}
        assert names == {"utils.Counter.new", "utils.Counter.increment"}


async def test_rust_derive_emits_inferred_inherits(clean_repo, rust_fixture_root: Path):
    """`#[derive(Clone, Debug, Default)]` on Counter and `#[derive(Serialize)]`
    on Report should emit one inferred-confidence inherits edge per derived
    trait. The base names land in inherits_edges.base_name; base_def_id is
    NULL for traits whose definition lives outside the repo."""
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, rust_fixture_root)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.qualified_name AS child, ie.base_name, ie.confidence
            FROM inherits_edges ie
            JOIN definitions d ON d.id=ie.child_def_id
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1
            ORDER BY child, base_name
            """,
            repo_id,
        )
        edges = {(r["child"], r["base_name"], r["confidence"]) for r in rows}
        assert ("utils.Counter", "Clone", "inferred") in edges
        assert ("utils.Counter", "Debug", "inferred") in edges
        assert ("utils.Counter", "Default", "inferred") in edges
        assert ("main.Report", "Serialize", "inferred") in edges

