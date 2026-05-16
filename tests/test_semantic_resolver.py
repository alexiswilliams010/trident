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
    ident = next(r for r in cfg.references if r.node_type == "identifier")
    assert "function_definition.name" in ident.exclude_parent_field
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
    assert len(cfg.inheritance) == 1
    assert "class_declaration" in cfg.inheritance[0].parent_node_types


def test_rust_yaml_loads():
    cfg = load_language_config("rust")
    assert cfg.language == "rust"
    assert cfg.module_node_type == "source_file"
    kinds = {r.kind for r in cfg.definitions}
    assert {"function", "type", "trait", "module", "macro"} <= kinds
    assert all(d.node_type != "impl_item" for d in cfg.definitions)
    assert any(c.node_type == "macro_invocation" for c in cfg.calls)
    assert any(c.node_type == "call_expression" for c in cfg.calls)
    assert cfg.imports is not None
    assert "use_declaration" in cfg.imports.node_types


def test_typescript_yaml_loads():
    cfg = load_language_config("typescript")
    assert cfg.language == "typescript"
    kinds = {r.kind for r in cfg.definitions}
    assert {"function", "class", "method", "variable", "interface", "type", "enum"} <= kinds
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


# SELECT-fragment used by every per-branch test. Joins the path lookup
# through branch_files so file_path filters still work, while keeping the
# branch_id WHERE clause as a single $1 parameter.
_BRANCH_FILE_JOIN = (
    "JOIN branch_files bf ON bf.file_version_id = d.file_version_id "
    "WHERE bf.branch_id=$1 AND bf.path"
)


async def test_resolve_python_fixture_definitions(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, python_fixture_root)
    results = await resolve_repo(pool, repo_id, branch_id)
    assert any(r.n_definitions > 0 for r in results)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.kind, d.name, d.qualified_name
            FROM definitions d
            {_BRANCH_FILE_JOIN}='mypackage/main.py'
            ORDER BY d.id
            """,
            branch_id,
        )
        by_q = {r["qualified_name"]: (r["kind"], r["name"]) for r in rows}
        assert by_q["main"] == ("module", "main")
        assert by_q["main.Calculator"] == ("class", "Calculator")
        assert by_q["main.Calculator.__init__"] == ("function", "__init__")
        assert by_q["main.Calculator.add"] == ("function", "add")
        assert by_q["main.Calculator.double_it"] == ("function", "double_it")
        assert by_q["main.run"] == ("function", "run")
        assert by_q["main.greet"] == ("function", "greet")
        assert by_q["main.GREETING"] == ("variable", "GREETING")

        scope_chain = await conn.fetch(
            f"""
            SELECT d.qualified_name, parent.qualified_name AS parent_qn
            FROM definitions d
            LEFT JOIN definitions parent ON parent.id=d.scope_id
            {_BRANCH_FILE_JOIN}='mypackage/main.py'
              AND d.qualified_name='main.Calculator.add'
            """,
            branch_id,
        )
        assert scope_chain[0]["parent_qn"] == "main.Calculator"


async def test_resolve_python_call_edges(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, python_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn
            FROM call_edges ce
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            JOIN definitions caller ON caller.id=ce.caller_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='main.run'
              AND callee.qualified_name='main.Calculator'
            """,
            branch_id,
        )
        assert row is not None
        assert row["confidence"] == "certain"

        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='main.run'
              AND callee.qualified_name='main.Calculator.add'
            """,
            branch_id,
        )
        assert row is not None
        assert row["confidence"] == "inferred"


async def test_resolve_python_data_access(clean_repo, python_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, python_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT da.access_type, target.qualified_name AS target_qn
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1 AND accessor.qualified_name='main.greet'
              AND target.qualified_name='main.GREETING'
            """,
            branch_id,
        )
        assert row is not None
        assert row["access_type"] == "read"


# ────────────────────────────────────────────────────────────────────
# End-to-end: Solidity fixture
# ────────────────────────────────────────────────────────────────────


async def test_resolve_solidity_definitions(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, solidity_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.kind, d.qualified_name
            FROM definitions d
            {_BRANCH_FILE_JOIN}='src/Vault.sol'
            ORDER BY d.id
            """,
            branch_id,
        )
        kinds = {r["qualified_name"]: r["kind"] for r in rows}
        assert kinds["Vault.Vault"] == "contract"
        assert kinds["Vault.Vault.deposit"] == "function"
        assert kinds["Vault.Vault.withdraw"] == "function"
        assert kinds.get("Vault.Vault.token") == "state_variable"
        assert kinds.get("Vault.Vault.balances") == "state_variable"
        assert kinds.get("Vault.Vault.Deposit") == "event"
        assert kinds.get("Vault.Vault.Withdraw") == "event"


async def test_resolve_solidity_data_access(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, solidity_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT da.access_type
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1 AND accessor.qualified_name='Vault.Vault.withdraw'
              AND target.qualified_name='Vault.Vault.balances'
            """,
            branch_id,
        )
        access_types = {r["access_type"] for r in rows}
        assert "write" in access_types
        assert "read" in access_types


async def test_resolve_solidity_emit_call_edge(clean_repo, solidity_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, solidity_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn, callee.kind
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='Vault.Vault.deposit'
              AND callee.qualified_name='Vault.Vault.Deposit'
            """,
            branch_id,
        )
        assert row is not None
        assert row["kind"] == "event"


# ────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript
# ────────────────────────────────────────────────────────────────────


async def test_resolve_javascript_definitions(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, node_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.kind, d.qualified_name FROM definitions d
            {_BRANCH_FILE_JOIN}='src/index.js' AND d.kind <> 'module'
            ORDER BY d.qualified_name
            """,
            branch_id,
        )
        kinds_by_qn = {r["qualified_name"]: r["kind"] for r in rows}
        assert kinds_by_qn["index.Calculator"] == "class"
        assert kinds_by_qn["index.Calculator.constructor"] == "method"
        assert kinds_by_qn["index.Calculator.describe"] == "method"
        assert kinds_by_qn["index.run"] == "function"
        assert kinds_by_qn["index.leftpad"] == "variable"
        assert kinds_by_qn["index.scoped"] == "variable"
        assert "index.Calculator.describe.a" not in kinds_by_qn


async def test_resolve_typescript_definitions(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, node_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.kind, d.qualified_name FROM definitions d
            JOIN branch_files bf ON bf.file_version_id = d.file_version_id
            WHERE bf.branch_id=$1 AND d.kind IN ('interface','type','enum','method')
            ORDER BY d.qualified_name
            """,
            branch_id,
        )
        kinds_by_qn = {r["qualified_name"]: r["kind"] for r in rows}
        assert kinds_by_qn["lib.Greeter"] == "interface"
        assert kinds_by_qn["lib.Bilingual"] == "interface"
        assert kinds_by_qn["helpers.Closer"] == "interface"
        assert kinds_by_qn["helpers.Pair"] == "type"
        assert kinds_by_qn["helpers.Status"] == "enum"
        assert kinds_by_qn["lib.Greeter.greet"] == "method"
        assert kinds_by_qn["helpers.Closer.close"] == "method"


async def test_resolve_rust_definitions(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, rust_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.kind, d.qualified_name FROM definitions d
            {_BRANCH_FILE_JOIN}='src/utils.rs' AND d.kind <> 'module'
            ORDER BY d.qualified_name
            """,
            branch_id,
        )
        by_q = {r["qualified_name"]: r["kind"] for r in rows}
        assert by_q["utils.Counter"] == "type"
        assert by_q["utils.helper"] == "function"
        assert by_q["utils.double"] == "function"
        assert by_q["utils.Counter.new"] == "function"
        assert by_q["utils.Counter.increment"] == "function"
        impl_rows = await conn.fetch(
            "SELECT 1 FROM definitions d "
            "JOIN branch_files bf ON bf.file_version_id = d.file_version_id "
            "WHERE bf.branch_id=$1 AND d.kind='impl'",
            branch_id,
        )
        assert impl_rows == []


async def test_resolve_rust_skips_inline_test_module(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, rust_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.qualified_name FROM definitions d
            {_BRANCH_FILE_JOIN}='src/utils.rs'
            ORDER BY d.qualified_name
            """,
            branch_id,
        )
        names = {r["qualified_name"] for r in rows}

        assert "utils.helper" in names
        assert "utils.double" in names
        assert "utils.Counter" in names
        assert "utils.Counter.new" in names

        assert all(not n.startswith("utils.tests") for n in names), names
        assert "utils.test_helper_increments" not in names
        assert "utils.test_double_doubles_helper" not in names
        assert "utils.test_only_helper" not in names


async def test_resolve_rust_skips_integration_tests_dir(clean_repo, rust_fixture_root: Path):
    """A file under `tests/` is a separate cargo compilation unit. Its
    definitions and references must be skipped entirely; the file_version
    row is still created (Tier-1 indexes it) but no semantic rows point at it.
    """
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, rust_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        # Tier-1 still parsed the file — branch_files maps its path to a
        # file_version row with nodes attached.
        integration_fv = await conn.fetchval(
            "SELECT file_version_id FROM branch_files "
            "WHERE branch_id=$1 AND path='tests/integration.rs'",
            branch_id,
        )
        assert integration_fv is not None
        # Tier-2 emitted no defs / refs for it.
        n_defs = await conn.fetchval(
            "SELECT COUNT(*) FROM definitions WHERE file_version_id=$1", integration_fv,
        )
        n_refs = await conn.fetchval(
            'SELECT COUNT(*) FROM "references" WHERE branch_id=$1 AND file_version_id=$2',
            branch_id, integration_fv,
        )
        assert n_defs == 0
        assert n_refs == 0


async def test_resolve_rust_within_file_calls(clean_repo, rust_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, rust_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ce.confidence, callee.qualified_name AS callee_qn
            FROM call_edges ce
            JOIN definitions caller ON caller.id=ce.caller_def_id
            LEFT JOIN definitions callee ON callee.id=ce.callee_def_id
            WHERE ce.branch_id=$1 AND caller.qualified_name='utils.Counter.increment'
              AND callee.qualified_name='utils.helper'
            """,
            branch_id,
        )
        assert row is not None
        assert row["confidence"] == "certain"


async def test_resolve_rust_struct_field_data_access(clean_repo, rust_fixture_root: Path):
    """`Counter.value` should appear as a `field` def, and `Counter::increment`
    should have a write + read on it (from `self.value = helper(self.value + by)`).
    Field-reference resolution uses the inferred-confidence file_name_index
    path: since `value` is unique in utils.rs, refs resolve to the field def.
    """
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, rust_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        field_kind = await conn.fetchval(
            f"""
            SELECT d.kind FROM definitions d
            {_BRANCH_FILE_JOIN}='src/utils.rs' AND d.qualified_name='utils.Counter.value'
            """,
            branch_id,
        )
        assert field_kind == "field"

        rows = await conn.fetch(
            """
            SELECT da.access_type
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1
              AND accessor.qualified_name='utils.Counter.increment'
              AND target.qualified_name='utils.Counter.value'
            """,
            branch_id,
        )
        access_types = {r["access_type"] for r in rows}
        assert "write" in access_types
        assert "read" in access_types


async def test_resolve_rust_field_data_access_skips_ambiguous_names(
    clean_repo, tmp_path: Path
):
    """When two structs in the same file share a field name AND the
    accessor is a plain function (no `impl` context), neither resolution
    path can pick the right field — the inferred lookup refuses on
    ambiguity and the impl-aware hook abstains because there's no `self`.
    No data_access rows should appear: the no-false-positive guarantee.
    """
    pool, repo_id, branch_id = clean_repo
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname="ambig"\nversion="0.0.0"\nedition="2021"\n'
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        "pub struct A { pub count: u32 }\n"
        "pub struct B { pub count: u32 }\n"
        "pub fn bump(a: &mut A) { a.count = a.count + 1; }\n"
    )
    await index_repo(pool, repo_id, branch_id, tmp_path)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT da.access_type
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1 AND target.name='count'
            """,
            branch_id,
        )
        assert rows == []


async def test_resolve_rust_impl_aware_field_resolution(
    clean_repo, tmp_path: Path
):
    """Multi-struct file where the field name is ambiguous, but the
    accessor is `self.count` inside `impl A`. The type-aware handler hook
    must pick `A.count` (not `B.count`) and tag the ref as `certain`
    rather than the rule's blanket `inferred`.
    """
    pool, repo_id, branch_id = clean_repo
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname="impl_aware"\nversion="0.0.0"\nedition="2021"\n'
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        "pub struct A { pub count: u32 }\n"
        "pub struct B { pub count: u32 }\n"
        "impl A {\n"
        "    pub fn bump(&mut self) { self.count = self.count + 1; }\n"
        "}\n"
        "impl B {\n"
        "    pub fn read(&self) -> u32 { self.count }\n"
        "}\n"
    )
    await index_repo(pool, repo_id, branch_id, tmp_path)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        a_rows = await conn.fetch(
            """
            SELECT da.access_type
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1
              AND accessor.qualified_name='lib.A.bump'
              AND target.qualified_name='lib.A.count'
            """,
            branch_id,
        )
        a_types = {r["access_type"] for r in a_rows}
        assert "write" in a_types
        assert "read" in a_types

        b_rows = await conn.fetch(
            """
            SELECT da.access_type
            FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1
              AND accessor.qualified_name='lib.B.read'
              AND target.qualified_name='lib.B.count'
            """,
            branch_id,
        )
        assert {r["access_type"] for r in b_rows} == {"read"}

        # Cross-check: B.read must NOT have touched A.count, and A.bump
        # must NOT have touched B.count.
        cross = await conn.fetchval(
            """
            SELECT COUNT(*) FROM data_access da
            JOIN definitions accessor ON accessor.id=da.accessor_def_id
            JOIN definitions target   ON target.id=da.target_def_id
            WHERE da.branch_id=$1 AND (
              (accessor.qualified_name='lib.A.bump' AND target.qualified_name='lib.B.count') OR
              (accessor.qualified_name='lib.B.read' AND target.qualified_name='lib.A.count')
            )
            """,
            branch_id,
        )
        assert cross == 0

        # Confidence got upgraded from the rule's blanket `inferred` to
        # `certain` because the impl-aware hook returned a typed answer.
        ref_conf = await conn.fetchval(
            """
            SELECT MAX(r.resolution_confidence) FROM "references" r
            JOIN definitions target ON target.id=r.target_def_id
            WHERE r.branch_id=$1 AND target.qualified_name='lib.A.count'
            """,
            branch_id,
        )
        assert ref_conf == 1.0


async def test_resolve_node_intra_file_inheritance(clean_repo, node_fixture_root: Path):
    pool, repo_id, branch_id = clean_repo
    await index_repo(pool, repo_id, branch_id, node_fixture_root)
    await resolve_repo(pool, repo_id, branch_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT child.qualified_name AS child, ie.base_name,
                   base.qualified_name AS base, ie.confidence
            FROM inherits_edges ie
            JOIN definitions child ON child.id=ie.child_def_id
            LEFT JOIN definitions base ON base.id=ie.base_def_id
            WHERE ie.branch_id=$1 AND child.qualified_name='lib.Bilingual'
            """,
            branch_id,
        )
        assert len(rows) == 1
        assert rows[0]["base_name"] == "Greeter"
        assert rows[0]["base"] == "lib.Greeter"
        assert rows[0]["confidence"] == "certain"
