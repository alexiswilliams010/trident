"""Phase 4a acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.chunk_assembler import (
    GRANULARITY_FUNCTION,
    HARD_OUTPUT_CAP,
    _degrade_if_oversize,
    assemble_chunks,
    count_tokens,
)
from core.extractor import index_repo
from core.heuristic_resolver import resolve_repo_imports
from core.semantic_resolver import resolve_repo


async def _full_pipeline(pool, repo_id: int, root: Path):
    await index_repo(pool, repo_id, root)
    await resolve_repo(pool, repo_id)
    await resolve_repo_imports(pool, repo_id)
    return await assemble_chunks(pool, repo_id)


# ────────────────────────────────────────────────────────────────────
# Token counting smoke
# ────────────────────────────────────────────────────────────────────


def test_count_tokens_basic():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


# ────────────────────────────────────────────────────────────────────
# Oversize degradation
# ────────────────────────────────────────────────────────────────────


def test_degrade_keeps_body_when_enrichment_overflows():
    """When the assembled chunk overflows the cap but the body alone fits,
    we should keep the body and shed the enrichment context. The previous
    behavior of replacing the entire chunk with a metadata stub destroyed
    retrieval signal and clustered every degraded chunk in the same corner
    of embedding space."""
    cap = HARD_OUTPUT_CAP[GRANULARITY_FUNCTION]
    metadata = {
        "anchor": "Foo.move",
        "kind": "function",
        "language": "solidity",
        "file": "src/Foo.sol",
        "granularity": GRANULARITY_FUNCTION,
    }
    body = "function move() external {\n    require(true);\n    state = 1;\n}"
    bloat = " ".join(["bloat"] * (cap * 2))
    full_content = f"/* preamble */\n{body}\n\n# extras\n{bloat}"
    assert count_tokens(full_content) > cap

    new_meta, new_content, tc = _degrade_if_oversize(
        GRANULARITY_FUNCTION, metadata, full_content,
        anchor_label="Foo.move", file_path="src/Foo.sol",
        body_only=body,
    )
    assert new_meta["degraded"] == "enrichment_shed"
    assert "function move()" in new_content
    assert "bloat" not in new_content
    assert tc <= cap


def test_degrade_trims_metadata_incrementally_least_important_first(monkeypatch):
    """When (full preamble + body) overflows because the metadata is bulky,
    drop bulky JSON fields one at a time in least-→most-important order
    until it fits. `callers` is asymmetric (not derivable from the body)
    and should survive longest; `dependencies` is partially redundant with
    the body's own call sites and should be shed earlier.

    Pins a small per-test cap so the assertion isn't coupled to the
    production cap, which is tuned to the embedding model and will shift
    over time.
    """
    test_cap = 300
    monkeypatch.setitem(HARD_OUTPUT_CAP, GRANULARITY_FUNCTION, test_cap)

    body = "function move() external {\n    doThing();\n}"
    # Big `dependencies` list — alone enough to push the preamble past the
    # pinned 300-token cap. Keep `callers` small so we can verify it
    # survives the trim.
    big_deps = [f"Pkg.Contract.dep_{i:04d}" for i in range(60)]
    metadata = {
        "anchor": "Pkg.Contract.move",
        "kind": "function",
        "language": "solidity",
        "file": "src/Pkg/Contract.sol",
        "dependencies": big_deps,
        "callers": ["Pkg.Contract.attack", "Pkg.Contract.defend"],
        "external_deps": ["foo", "bar"],
        "inheritance_chain": ["BaseContract"],
        "overrides": "BaseContract.move",
        "granularity": GRANULARITY_FUNCTION,
    }
    bloat = " ".join(["bloat"] * (test_cap * 2))
    full_content = f"/* preamble */\n{body}\n\n# extras\n{bloat}"

    new_meta, new_content, tc = _degrade_if_oversize(
        GRANULARITY_FUNCTION, metadata, full_content,
        anchor_label="Pkg.Contract.move", file_path="src/Pkg/Contract.sol",
        body_only=body,
    )
    assert new_meta["degraded"] == "metadata_trimmed"
    assert "function move()" in new_content
    assert tc <= test_cap
    # `dependencies` is the bulky, partially-redundant field — should be the
    # first thing dropped, and dropping just it should be enough here.
    assert "dependencies" not in new_meta
    # Higher-value enrichment that wasn't required to fit must survive.
    assert new_meta.get("callers") == ["Pkg.Contract.attack", "Pkg.Contract.defend"]
    assert new_meta.get("overrides") == "BaseContract.move"


def test_degrade_falls_back_to_stub_when_body_alone_overflows():
    """Pathological case: even the body alone is bigger than the cap (e.g.
    minified vendored code). Fall through to the existing stub behavior."""
    cap = HARD_OUTPUT_CAP[GRANULARITY_FUNCTION]
    metadata = {
        "anchor": "Vendor.blob",
        "kind": "function",
        "language": "javascript",
        "file": "dist/bundle.js",
        "granularity": GRANULARITY_FUNCTION,
    }
    huge_body = " ".join(["x"] * (cap * 3))
    full_content = f"/* preamble */\n{huge_body}"
    assert count_tokens(huge_body) > cap

    new_meta, new_content, _ = _degrade_if_oversize(
        GRANULARITY_FUNCTION, metadata, full_content,
        anchor_label="Vendor.blob", file_path="dist/bundle.js",
        body_only=huge_body,
    )
    assert new_meta["degraded"] == "oversize"
    assert "(degraded): Vendor.blob" in new_content


def test_degrade_passthrough_when_under_cap():
    """No-op when the chunk is under cap — metadata and content unchanged."""
    metadata = {"anchor": "Foo.bar", "granularity": GRANULARITY_FUNCTION}
    content = "/* preamble */\nfunction bar() {}"
    new_meta, new_content, tc = _degrade_if_oversize(
        GRANULARITY_FUNCTION, metadata, content,
        anchor_label="Foo.bar", file_path="src/Foo.sol",
        body_only="function bar() {}",
    )
    assert new_meta is metadata
    assert new_content == content
    assert tc == count_tokens(content)


# ────────────────────────────────────────────────────────────────────
# Python fixture
# ────────────────────────────────────────────────────────────────────


async def test_assemble_python_chunks(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    stats = await _full_pipeline(pool, repo_id, python_fixture_root)

    # Each function gets a function-level chunk; each module gets a module chunk.
    assert stats.n_function > 0
    assert stats.n_module >= 4  # 4 Python files including __init__.py

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.granularity, c.token_count, c.metadata, c.content,
                   d.qualified_name
            FROM chunks c JOIN files f ON f.id=c.file_id
            LEFT JOIN definitions d ON d.id=c.anchor_def_id
            WHERE f.repo_id=$1
            ORDER BY c.id
            """,
            repo_id,
        )
        by_anchor: dict[tuple[str, str], dict] = {}
        for r in rows:
            by_anchor[(r["qualified_name"], r["granularity"])] = r

        # Calculator.add (function) chunk should mention helper as a callee.
        add_fn = by_anchor[("main.Calculator.add", "function")]
        assert "helper" in add_fn["content"]
        # Metadata is well-formed JSON inside the chunk.
        assert "tsgrep-meta" in add_fn["content"]
        meta = json.loads(add_fn["metadata"])
        assert meta["anchor"] == "main.Calculator.add"
        assert meta["language"] == "python"
        assert meta["granularity"] == "function"

        # Cross-module chunk for run() should reference Calculator (its callee).
        run_xm = by_anchor.get(("main.run", "cross-module"))
        assert run_xm is not None
        assert "Calculator" in run_xm["content"]
        assert run_xm["token_count"] <= 2048

        # Module chunk for main.py should list its imports + dependencies.
        main_module = by_anchor[("main", "module")]
        main_meta = json.loads(main_module["metadata"])
        assert "main.Calculator" in main_meta["dependencies"]
        assert "imports" in main_module["content"]


async def test_chunk_metadata_lists_external_deps(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    await _full_pipeline(pool, repo_id, python_fixture_root)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT metadata FROM chunks c "
            "JOIN files f ON f.id=c.file_id "
            "JOIN definitions d ON d.id=c.anchor_def_id "
            "WHERE f.repo_id=$1 AND d.qualified_name='main' AND c.granularity='module'",
            repo_id,
        )
        assert row is not None
        meta = json.loads(row["metadata"])
        # `import requests` is the external import in main.py.
        assert "requests" in meta["external_deps"]


# ────────────────────────────────────────────────────────────────────
# Solidity fixture
# ────────────────────────────────────────────────────────────────────


async def test_assemble_solidity_chunks(clean_repo, solidity_fixture_root: Path):
    pool, repo_id = clean_repo
    stats = await _full_pipeline(pool, repo_id, solidity_fixture_root)
    assert stats.n_function > 0

    async with pool.acquire() as conn:
        # Vault.deposit (function) — mentions balances + Token.transfer.
        row = await conn.fetchrow(
            """
            SELECT c.content, c.metadata
            FROM chunks c JOIN definitions d ON d.id=c.anchor_def_id
            JOIN files f ON f.id=c.file_id
            WHERE f.repo_id=$1 AND d.qualified_name='Vault.Vault.deposit'
              AND c.granularity='function'
            """,
            repo_id,
        )
        assert row is not None
        content = row["content"]
        assert "balances" in content              # state-var dependency
        assert "transfer" in content              # callee signature/body
        meta = json.loads(row["metadata"])
        assert "Vault.Vault.balances" in meta["dependencies"]


async def test_chunk_idempotency(clean_repo, python_fixture_root: Path):
    pool, repo_id = clean_repo
    first = await _full_pipeline(pool, repo_id, python_fixture_root)
    second = await assemble_chunks(pool, repo_id)
    # Re-running must not insert duplicates and must mostly hit the unchanged path.
    assert first.total == second.total
    assert second.n_inserted == 0
    assert second.n_unchanged == first.total
