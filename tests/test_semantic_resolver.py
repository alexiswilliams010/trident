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
