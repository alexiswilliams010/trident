"""Phase 2 acceptance tests."""

from __future__ import annotations

from pathlib import Path

from core.config_loader import load_language_config
from core.extractor import index_repo
from core.semantic_resolver import resolve_repo


# ────────────────────────────────────────────────────────────────────
# Config-loader smoke tests (no DB)
# ────────────────────────────────────────────────────────────────────


def test_python_yaml_loads():
    cfg = load_language_config("python")
    assert cfg.language == "python"
    kinds = {r.kind for r in cfg.definitions}
    assert {"function", "class", "variable"} <= kinds
    assert "module" in kinds or cfg.module_node_type == "module"
    # Reference rule for identifier exists with expected exclusions.
    ident = next(r for r in cfg.references if r.node_type == "identifier")
    assert "function_definition.name" in ident.exclude_parent_field
    # Call rule for `call` exists.
    assert any(r.node_type == "call" for r in cfg.calls)


def test_solidity_yaml_loads():
    cfg = load_language_config("solidity")
    assert cfg.language == "solidity"
    kinds = {r.kind for r in cfg.definitions}
    assert {"contract", "function", "state_variable", "event"} <= kinds
    assert any(r.node_type == "call_expression" for r in cfg.calls)
    assert any(r.node_type == "emit_statement" for r in cfg.calls)


def test_javascript_yaml_loads():
    cfg = load_language_config("javascript")
    assert cfg.language == "javascript"
    assert cfg.module_node_type == "program"
    kinds = {r.kind for r in cfg.definitions}
    assert {"function", "class", "method", "variable"} <= kinds
    assert any(r.node_type == "call_expression" for r in cfg.calls)
    # Single inheritance rule for `class extends`.
    assert len(cfg.inheritance) == 1
    assert "class_declaration" in cfg.inheritance[0].parent_node_types


def test_rust_yaml_loads():
    cfg = load_language_config("rust")
    assert cfg.language == "rust"
    assert cfg.module_node_type == "source_file"
    kinds = {r.kind for r in cfg.definitions}
    assert {"function", "type", "trait", "module", "macro"} <= kinds
    # impl_item is intentionally NOT a definition — see configs/rust.yaml.
    assert all(d.node_type != "impl_item" for d in cfg.definitions)
    # Macro invocations participate as call edges, with the macro name field.
    assert any(c.node_type == "macro_invocation" for c in cfg.calls)
    assert any(c.node_type == "call_expression" for c in cfg.calls)
    # use_declaration is registered as the import node.
    assert cfg.imports is not None
    assert "use_declaration" in cfg.imports.node_types


def test_typescript_yaml_loads():
    cfg = load_language_config("typescript")
    assert cfg.language == "typescript"
    kinds = {r.kind for r in cfg.definitions}
    # Superset of JS plus interface / type / enum.
    assert {"function", "class", "method", "variable", "interface", "type", "enum"} <= kinds
    # Three inheritance rules: extends, implements, interface-extends.
    assert len(cfg.inheritance) == 3
    impl_rule = next(
        r for r in cfg.inheritance if r.child_node_type == "implements_clause"
    )
    assert impl_rule.child_iterate_identifiers is True
    iface_rule = next(
        r for r in cfg.inheritance if r.child_node_type == "extends_type_clause"
    )
    assert iface_rule.child_iterate_identifiers is True


# ────────────────────────────────────────────────────────────────────
# End-to-end: extractor + resolver on the Python fixture
# ────────────────────────────────────────────────────────────────────


async def test_resolve_python_fixture_definitions(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, python_fixture_root)
    results = await resolve_repo(pool, repo_id)
    assert any(r.n_definitions > 0 for r in results)

    async with pool.acquire() as conn:
        # main.py has Calculator (class) and free functions run, greet.
        rows = await conn.fetch(
            """
            SELECT d.kind, d.name, d.qualified_name
            FROM definitions d JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND f.path='mypackage/main.py'
            ORDER BY d.id
            """,
            repo_id,
        )
        by_q = {r["qualified_name"]: (r["kind"], r["name"]) for r in rows}
        assert by_q["main"] == ("module", "main")
        assert by_q["main.Calculator"] == ("class", "Calculator")
        assert by_q["main.Calculator.__init__"] == ("function", "__init__")
        assert by_q["main.Calculator.add"] == ("function", "add")
        assert by_q["main.Calculator.double_it"] == ("function", "double_it")
        assert by_q["main.run"] == ("function", "run")
        assert by_q["main.greet"] == ("function", "greet")
        # Top-level variable is captured.
        assert by_q["main.GREETING"] == ("variable", "GREETING")

        # Sanity: Calculator's scope_id points at the module def.
        scope_chain = await conn.fetch(
            """
            SELECT d.qualified_name, parent.qualified_name AS parent_qn
            FROM definitions d
            LEFT JOIN definitions parent ON parent.id=d.scope_id
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND f.path='mypackage/main.py'
              AND d.qualified_name='main.Calculator.add'
            """,
            repo_id,
        )
        assert scope_chain[0]["parent_qn"] == "main.Calculator"


async def test_resolve_python_call_edges(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, python_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        # `Calculator(10)` inside run → call to Calculator class (certain).
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn
            FROM call_edges ce
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN definitions caller ON caller.id=ce.caller_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='main.run'
              AND callee.qualified_name='main.Calculator'
            """,
            repo_id,
        )
        assert row is not None
        assert row["confidence"] == "certain"

        # `calc.add(5)` inside run → attribute call → inferred (resolves by
        # name match to main.Calculator.add since `add` is unique in the file).
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='main.run'
              AND callee.qualified_name='main.Calculator.add'
            """,
            repo_id,
        )
        assert row is not None
        assert row["confidence"] == "inferred"


async def test_resolve_python_data_access(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, python_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        # greet() reads GREETING.
        row = await conn.fetchrow(
            """
            SELECT da.access_type, target.qualified_name AS target_qn
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            JOIN files f ON f.id=accessor.file_id
            WHERE f.repo_id=$1 AND accessor.qualified_name='main.greet'
              AND target.qualified_name='main.GREETING'
            """,
            repo_id,
        )
        assert row is not None
        assert row["access_type"] == "read"


# ────────────────────────────────────────────────────────────────────
# End-to-end: Solidity fixture
# ────────────────────────────────────────────────────────────────────


async def test_resolve_solidity_definitions(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, solidity_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.kind, d.qualified_name
            FROM definitions d JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND f.path='src/Vault.sol'
            ORDER BY d.id
            """,
            repo_id,
        )
        kinds = {r["qualified_name"]: r["kind"] for r in rows}
        assert kinds["Vault.Vault"] == "contract"
        assert kinds["Vault.Vault.deposit"] == "function"
        assert kinds["Vault.Vault.withdraw"] == "function"
        # state variables (token, balances) and events (Deposit, Withdraw).
        assert kinds.get("Vault.Vault.token") == "state_variable"
        assert kinds.get("Vault.Vault.balances") == "state_variable"
        assert kinds.get("Vault.Vault.Deposit") == "event"
        assert kinds.get("Vault.Vault.Withdraw") == "event"


async def test_resolve_solidity_data_access(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, solidity_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        # withdraw() writes balances (`balances[msg.sender] -= amount`).
        rows = await conn.fetch(
            """
            SELECT da.access_type
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            JOIN files f ON f.id=accessor.file_id
            WHERE f.repo_id=$1 AND accessor.qualified_name='Vault.Vault.withdraw'
              AND target.qualified_name='Vault.Vault.balances'
            """,
            repo_id,
        )
        access_types = {r["access_type"] for r in rows}
        assert "write" in access_types  # `balances[msg.sender] -= amount`
        assert "read" in access_types   # `balances[msg.sender] >= amount`


async def test_resolve_solidity_emit_call_edge(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, solidity_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        # `emit Deposit(...)` inside Vault.deposit → call edge to event Deposit.
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn, callee.kind
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='Vault.Vault.deposit'
              AND callee.qualified_name='Vault.Vault.Deposit'
            """,
            repo_id,
        )
        assert row is not None
        assert row["kind"] == "event"


# ────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript
# ────────────────────────────────────────────────────────────────────


async def test_resolve_javascript_definitions(clean_repo, node_fixture_root: Path):
    """Top-level JS definitions: class with methods, function, top-level
    variable_declarator (require() bindings count as such)."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, node_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.kind, d.qualified_name FROM definitions d
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND f.path='src/index.js' AND d.kind <> 'module'
            ORDER BY d.qualified_name
            """,
            repo_id,
        )
        kinds_by_qn = {r["qualified_name"]: r["kind"] for r in rows}
        assert kinds_by_qn["index.Calculator"] == "class"
        assert kinds_by_qn["index.Calculator.constructor"] == "method"
        assert kinds_by_qn["index.Calculator.describe"] == "method"
        assert kinds_by_qn["index.run"] == "function"
        # require() bound names are top-level lexical declarations and surface
        # as `variable` definitions.
        assert kinds_by_qn["index.leftpad"] == "variable"
        assert kinds_by_qn["index.scoped"] == "variable"
        # `const a = ...` inside Calculator.describe is INSIDE a method scope
        # and must be filtered out by require_enclosing_scope_kind: [module].
        assert "index.Calculator.describe.a" not in kinds_by_qn


async def test_resolve_typescript_definitions(clean_repo, node_fixture_root: Path):
    """TS-only constructs: interface, type alias, enum, and method_signature
    inside an interface body."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, node_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.kind, d.qualified_name FROM definitions d
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND d.kind IN ('interface','type','enum','method')
            ORDER BY d.qualified_name
            """,
            repo_id,
        )
        kinds_by_qn = {r["qualified_name"]: r["kind"] for r in rows}
        assert kinds_by_qn["lib.Greeter"] == "interface"
        assert kinds_by_qn["lib.Bilingual"] == "interface"
        assert kinds_by_qn["helpers.Closer"] == "interface"
        assert kinds_by_qn["helpers.Pair"] == "type"
        assert kinds_by_qn["helpers.Status"] == "enum"
        # Interface methods (`method_signature`) become method defs so
        # override generation can connect implementing classes to them.
        assert kinds_by_qn["lib.Greeter.greet"] == "method"
        assert kinds_by_qn["helpers.Closer.close"] == "method"


async def test_resolve_rust_definitions(clean_repo, rust_fixture_root: Path):
    """Methods defined inside `impl Counter { … }` get the impl's target type
    as their qualified-name prefix even though impl_item itself is not a
    definition."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, rust_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.kind, d.qualified_name FROM definitions d
            JOIN files f ON f.id=d.file_id
            WHERE f.repo_id=$1 AND f.path='src/utils.rs' AND d.kind <> 'module'
            ORDER BY d.qualified_name
            """,
            repo_id,
        )
        by_q = {r["qualified_name"]: r["kind"] for r in rows}
        assert by_q["utils.Counter"] == "type"
        assert by_q["utils.helper"] == "function"
        assert by_q["utils.double"] == "function"
        # Impl methods carry the type prefix.
        assert by_q["utils.Counter.new"] == "function"
        assert by_q["utils.Counter.increment"] == "function"
        # impl_item itself does NOT produce a definition row.
        impl_rows = await conn.fetch(
            "SELECT 1 FROM definitions d JOIN files f ON f.id=d.file_id "
            "WHERE f.repo_id=$1 AND d.kind='impl'",
            repo_id,
        )
        assert impl_rows == []


async def test_resolve_rust_skips_inline_test_module(clean_repo, rust_fixture_root: Path):
    """`#[cfg(test)] mod tests { … }` at the bottom of utils.rs must not
    contribute any definitions to the graph. The non-test items in the
    same file are still emitted normally."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, rust_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT d.qualified_name FROM definitions d "
            "JOIN files f ON f.id=d.file_id "
            "WHERE f.repo_id=$1 AND f.path='src/utils.rs' "
            "ORDER BY d.qualified_name",
            repo_id,
        )
        names = {r["qualified_name"] for r in rows}

        # Production code is still indexed.
        assert "utils.helper" in names
        assert "utils.double" in names
        assert "utils.Counter" in names
        assert "utils.Counter.new" in names

        # Nothing from `mod tests { … }` should be present — neither the
        # mod itself, nor any of its functions (with or without #[test]).
        assert all(not n.startswith("utils.tests") for n in names), names
        assert "utils.test_helper_increments" not in names
        assert "utils.test_double_doubles_helper" not in names
        assert "utils.test_only_helper" not in names


async def test_resolve_rust_skips_integration_tests_dir(clean_repo, rust_fixture_root: Path):
    """A file under `tests/` is a separate cargo compilation unit. Its
    definitions and references must be skipped entirely; the file row
    in `files` is still present (Tier-1 indexes it) but no semantic rows
    point at it."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, rust_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        # Tier-1 still parsed the file.
        integration_id = await conn.fetchval(
            "SELECT id FROM files WHERE repo_id=$1 AND path='tests/integration.rs'",
            repo_id,
        )
        assert integration_id is not None
        # Tier-2 did not emit any defs / refs / call_edges for it.
        n_defs = await conn.fetchval(
            "SELECT COUNT(*) FROM definitions WHERE file_id=$1", integration_id,
        )
        n_refs = await conn.fetchval(
            'SELECT COUNT(*) FROM "references" WHERE file_id=$1', integration_id,
        )
        assert n_defs == 0
        assert n_refs == 0


async def test_resolve_rust_within_file_calls(clean_repo, rust_fixture_root: Path):
    """`Counter::increment` calls `helper(...)` inside utils.rs — should
    resolve to utils.helper via in-file scope chain."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, rust_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN files f ON f.id=caller.file_id
            WHERE f.repo_id=$1 AND caller.qualified_name='utils.Counter.increment'
              AND callee.qualified_name='utils.helper'
            """,
            repo_id,
        )
        assert row is not None
        assert row["confidence"] == "certain"


async def test_resolve_node_intra_file_inheritance(clean_repo, node_fixture_root: Path):
    """`interface Bilingual extends Greeter` is intra-file in lib.ts and must
    resolve to Greeter via the `extends_type_clause` rule."""
    pool, repo_id = clean_repo
    await index_repo(pool, repo_id, node_fixture_root)
    await resolve_repo(pool, repo_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT child.qualified_name AS child, ie.base_name,
                   base.qualified_name AS base, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id=ie.child_def_id
            LEFT JOIN definitions base ON base.id=ie.base_def_id
            JOIN files f ON f.id=child.file_id
            WHERE f.repo_id=$1 AND child.qualified_name='lib.Bilingual'
            """,
            repo_id,
        )
        assert len(rows) == 1
        assert rows[0]["base_name"] == "Greeter"
        assert rows[0]["base"] == "lib.Greeter"
        assert rows[0]["confidence"] == "certain"
