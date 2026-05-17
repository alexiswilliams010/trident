"""Orchestrate one benchmark run: ensure DB → index/embed → run both agents.

Reuses the same pipeline as `python -m cli.index`. The only swap is the
embedder: we substitute `TrackingEmbedder` so we can capture the token usage
the gateway returns on each call.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg

from cli._repo import resolve_repo_and_branch
from core.chunk_assembler import assemble_chunks
from core.embedder import EmbedderConfig, embed_branch_chunks
from core.extractor import index_repo
from core.file_walker import WalkConfig, read_tridentignore
from core.heuristic_resolver import resolve_branch_imports
from core.semantic_resolver import resolve_repo
from db.connection import apply_migrations, pool_ctx

from benchmark.agents import AgentResult, run_baseline_agent, run_trident_agent
from benchmark.cost import embedding_cost_usd
from benchmark.embed_tracker import TrackingEmbedder


def default_db_name(repo_path: Path) -> str:
    h = hashlib.sha1(str(repo_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"trident_bench_{h}"


@dataclass
class IndexStats:
    db_name: str
    repo_name: str
    branch: str
    indexed_files: int
    skipped_files: int
    deleted_files: int
    chunks_seen: int
    chunks_embedded: int
    chunks_skipped: int
    embed_model: str
    embed_tokens: int
    embed_api_calls: int
    cost_usd: float | None


@dataclass
class BenchmarkResult:
    repo_path: str
    repo_name: str
    branch: str
    question: str
    model: str
    index: IndexStats
    trident_agent: AgentResult
    baseline_agent: AgentResult
    wall_ms: int = 0


async def _ensure_database(db_name: str, dsn_for_admin: str | None = None) -> str:
    """Create the target Postgres DB if it doesn't exist. Returns the DSN
    pointing at it. Connects to the `postgres` admin DB to issue CREATE."""
    import getpass
    import os

    user = os.environ.get("PGUSER") or getpass.getuser()
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    admin_dsn = dsn_for_admin or f"postgresql://{user}@{host}:{port}/postgres"
    target_dsn = f"postgresql://{user}@{host}:{port}/{db_name}"

    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name,
        )
        if not exists:
            print(f"[db] creating database '{db_name}'")
            safe = db_name.replace('"', '""')
            await conn.execute(f'CREATE DATABASE "{safe}"')
        else:
            print(f"[db] reusing existing database '{db_name}'")
    finally:
        await conn.close()
    return target_dsn


async def _index_repo(
    pool: asyncpg.Pool,
    *,
    repo_path: Path,
    repo_name: str,
    branch: str,
    skip_embed: bool,
    force_rehash: bool,
) -> IndexStats:
    print(f"[repo] resolving repo='{repo_name}' branch='{branch}'")
    repo_id, branch_id = await resolve_repo_and_branch(
        pool, repo_name=repo_name, branch_name=branch, create=True,
        root_path=str(repo_path.resolve()),
    )
    print(f"[repo] repo_id={repo_id} branch_id={branch_id}")

    walk_cfg = WalkConfig.with_defaults(repo_path)
    ignore_patterns = read_tridentignore(repo_path)
    if ignore_patterns:
        walk_cfg.exclude_patterns = tuple(list(walk_cfg.exclude_patterns) + ignore_patterns)
        print(f"[walk] loaded {len(ignore_patterns)} pattern(s) from .tridentignore")

    print(f"[Tier 1] extracting AST from {repo_path}...")
    extract_result = await index_repo(
        pool, repo_id, branch_id, repo_path,
        walk_config=walk_cfg, force_rehash=force_rehash,
    )
    print(
        f"[Tier 1] indexed={len(extract_result.indexed)} "
        f"skipped={len(extract_result.skipped)} (unchanged) "
        f"pruned={len(extract_result.deleted)} (stale)"
    )

    file_version_ids = [r.file_version_id for r in extract_result.indexed]
    if file_version_ids:
        print(f"[Tier 2] resolving semantics for {len(file_version_ids)} changed file(s)...")
        await resolve_repo(
            pool, repo_id, branch_id, only_file_version_ids=file_version_ids,
        )
    else:
        print("[Tier 2] no changed files; skipping semantic resolution")

    print("[Tier 2 imports] linking cross-file references...")
    await resolve_branch_imports(pool, repo_id, branch_id)

    print("[Tier 3 chunks] assembling chunks...")
    cstats = await assemble_chunks(pool, repo_id, branch_id)
    print(
        f"[Tier 3 chunks] total={cstats.total} "
        f"inserted={cstats.n_inserted} updated={cstats.n_updated} unchanged={cstats.n_unchanged}"
    )

    tracker_tokens = 0
    tracker_calls = 0
    chunks_embedded = 0
    chunks_skipped = 0
    embed_model = "(skipped)"
    if not skip_embed:
        cfg = EmbedderConfig.from_env()
        tracker = TrackingEmbedder(cfg)
        embed_model = cfg.model
        print(f"[Tier 3 embed] starting embedder model={embed_model}...")
        estats = await embed_branch_chunks(
            pool, branch_id, tracker.embed, embed_model, dim=cfg.dim,
        )
        tracker_tokens = tracker.total_input_tokens
        tracker_calls = tracker.api_calls
        chunks_embedded = estats.embedded
        chunks_skipped = estats.skipped
        chunks_seen_from_estats = estats.chunks_seen
        print(
            f"[Tier 3 embed] embedded={chunks_embedded}/{chunks_seen_from_estats} "
            f"api_calls={tracker_calls} tokens={tracker_tokens}"
        )
    else:
        print("[Tier 3 embed] skipped (--no-embed)")
        chunks_seen_from_estats = cstats.total

    return IndexStats(
        db_name="",  # filled in by caller
        repo_name=repo_name,
        branch=branch,
        indexed_files=len(extract_result.indexed),
        skipped_files=len(extract_result.skipped),
        deleted_files=len(extract_result.deleted),
        chunks_seen=chunks_seen_from_estats,
        chunks_embedded=chunks_embedded,
        chunks_skipped=chunks_skipped,
        embed_model=embed_model,
        embed_tokens=tracker_tokens,
        embed_api_calls=tracker_calls,
        cost_usd=embedding_cost_usd(tracker_tokens),
    )


async def run_benchmark(
    *,
    repo_path: Path,
    question: str,
    repo_name: str,
    branch: str = "main",
    db_name: str | None = None,
    model: str = "claude-sonnet-4-6",
    skip_embed: bool = False,
    force_rehash: bool = False,
    verbose: bool = False,
) -> BenchmarkResult:
    import time

    db_name = db_name or default_db_name(repo_path)
    print(f"[benchmark] repo_path={repo_path}  db={db_name}  model={model}")
    target_dsn = await _ensure_database(db_name)

    wall_start = time.monotonic()
    async with pool_ctx(target_dsn) as pool:
        async with pool.acquire() as conn:
            print("[db] applying migrations (idempotent)...")
            applied = await apply_migrations(conn)
            if applied:
                print(f"[db] applied migrations: {', '.join(applied)}")
            else:
                print("[db] schema up to date")
        idx = await _index_repo(
            pool,
            repo_path=repo_path,
            repo_name=repo_name,
            branch=branch,
            skip_embed=skip_embed,
            force_rehash=force_rehash,
        )
    idx.db_name = db_name

    print()
    print(f"[agents] dispatching both agents concurrently (model={model})...")
    print("[agents]   • trident  — cwd=trident repo, skills=['trident']")
    print(f"[agents]   • baseline — cwd={repo_path}, skills=[] (suppressed)")
    agents_start = time.monotonic()

    async def _run_with_log(label: str, coro):
        print(f"[agents] [{label}] started")
        result = await coro
        elapsed = time.monotonic() - agents_start
        status = "ERROR" if result.is_error else "ok"
        cost = f"${result.total_cost_usd:.4f}" if result.total_cost_usd is not None else "n/a"
        print(
            f"[agents] [{label}] finished ({status}) "
            f"turns={result.num_turns} cost={cost} after {elapsed:.1f}s"
        )
        return result

    trident_task = asyncio.create_task(_run_with_log(
        "trident",
        run_trident_agent(
            question=question, repo_name=repo_name, db_name=db_name,
            branch=branch, repo_path=repo_path, model=model, verbose=verbose,
        ),
    ))
    baseline_task = asyncio.create_task(_run_with_log(
        "baseline",
        run_baseline_agent(
            question=question, repo_path=repo_path, model=model, verbose=verbose,
        ),
    ))
    trident_result, baseline_result = await asyncio.gather(trident_task, baseline_task)

    wall_ms = int((time.monotonic() - wall_start) * 1000)
    return BenchmarkResult(
        repo_path=str(repo_path.resolve()),
        repo_name=repo_name,
        branch=branch,
        question=question,
        model=model,
        index=idx,
        trident_agent=trident_result,
        baseline_agent=baseline_result,
        wall_ms=wall_ms,
    )
