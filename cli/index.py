"""CLI: index a repository (Tier 1 extractor + Tier 2 semantic resolver).

Usage:
    python -m cli.index <repo_path> --repo-name myrepo [--dsn postgresql://...]
                        [--init-schema] [--no-resolve]
                        [--exclude PATTERN ...] [--no-tsgrepignore]

Exclusion patterns combine `.tsgrepignore` (auto-loaded from repo root) with
any `--exclude` flags. Patterns without `/` match any path component;
patterns with `/` match the full repo-relative path. Examples:
    --exclude '*.t.sol'     # Foundry test files anywhere
    --exclude test          # any directory or file named `test`
    --exclude src/legacy    # exact relative path
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from cli._repo import resolve_repo_id
from core.chunk_assembler import assemble_chunks
from core.embedder import EmbedderConfig, OpenAICompatibleEmbedder, embed_repo_chunks, make_fake_embedder
from core.extractor import index_repo
from core.file_walker import WalkConfig, read_tsgrepignore
from core.heuristic_resolver import resolve_repo_imports
from core.semantic_resolver import resolve_repo
from db.connection import apply_migrations, pool_ctx


async def _run(
    repo_path: Path,
    repo_name: str,
    dsn: str | None,
    init_schema: bool,
    do_resolve: bool,
    do_imports: bool,
    do_chunks: bool,
    do_embed: str | None,
    exclude_patterns: tuple[str, ...],
    use_tsgrepignore: bool,
) -> int:
    async with pool_ctx(dsn) as pool:
        if init_schema:
            async with pool.acquire() as conn:
                applied = await apply_migrations(conn)
                if applied:
                    print(f"Applied migrations: {', '.join(applied)}")

        repo_id = await resolve_repo_id(
            pool, name=repo_name, root_path=str(repo_path.resolve()), create=True,
        )
        print(f"[repo] {repo_name} → repo_id={repo_id}")

        # Build the walk config: defaults + .tsgrepignore (if present) + --exclude flags.
        walk_cfg = WalkConfig.with_defaults(repo_path)
        all_excludes: list[str] = []
        if use_tsgrepignore:
            ignore_patterns = read_tsgrepignore(repo_path)
            if ignore_patterns:
                all_excludes.extend(ignore_patterns)
                print(f"[walk] loaded {len(ignore_patterns)} pattern(s) from .tsgrepignore")
        all_excludes.extend(exclude_patterns)
        if all_excludes:
            walk_cfg.exclude_patterns = tuple(all_excludes)
            print(f"[walk] excluding: {', '.join(all_excludes)}")

        extract_result = await index_repo(pool, repo_id, repo_path, walk_config=walk_cfg)
        print(
            f"[Tier 1] Indexed {len(extract_result.indexed)} files "
            f"({extract_result.total_nodes} nodes); skipped {len(extract_result.skipped)} unchanged"
        )

        if do_resolve:
            file_ids = [r.file_id for r in extract_result.indexed]
            if file_ids:
                resolve_results = await resolve_repo(pool, repo_id, only_file_ids=file_ids)
                tot_defs = sum(r.n_definitions for r in resolve_results)
                tot_refs = sum(r.n_references for r in resolve_results)
                tot_calls = sum(r.n_call_edges for r in resolve_results)
                tot_da = sum(r.n_data_access for r in resolve_results)
                print(
                    f"[Tier 2] {len(resolve_results)} files: "
                    f"{tot_defs} definitions, {tot_refs} references, "
                    f"{tot_calls} call_edges, {tot_da} data_access"
                )

        if do_imports:
            stats = await resolve_repo_imports(pool, repo_id)
            cls = stats.by_class
            total = sum(cls.values()) or 1
            print(
                f"[Tier 2 imports] {sum(cls.values())} imports — "
                f"intra_repo={cls.get('intra_repo', 0)} "
                f"external={cls.get('external', 0)} "
                f"unresolved={cls.get('unresolved', 0)} "
                f"({100 * cls.get('intra_repo', 0) // total}% intra) "
                f"| linked {stats.cross_file_refs_resolved} refs, "
                f"{stats.cross_file_calls_resolved} call_edges, "
                f"{stats.cross_file_inherits_resolved} inherits, "
                f"{stats.overrides_inserted} overrides"
            )

        if do_chunks:
            cstats = await assemble_chunks(pool, repo_id)
            print(
                f"[Tier 3 chunks] {cstats.total} chunks "
                f"(function={cstats.n_function} module={cstats.n_module} "
                f"cross-module={cstats.n_cross_module}) "
                f"— inserted={cstats.n_inserted} updated={cstats.n_updated} unchanged={cstats.n_unchanged}"
            )

        if do_embed:
            if do_embed == "fake":
                embed_fn, model_name = make_fake_embedder()
            else:  # "real"
                cfg = EmbedderConfig.from_env()
                embed_fn = OpenAICompatibleEmbedder(cfg).embed
                model_name = cfg.model
            estats = await embed_repo_chunks(pool, repo_id, embed_fn, model_name)
            print(f"[Tier 3 embed] {estats.embedded}/{estats.chunks_seen} chunks embedded ({model_name})")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsgrep — index a repo (Tier 1 + Tier 2)")
    parser.add_argument("repo_path", type=Path, help="Path to the repo to index")
    parser.add_argument("--repo-name", type=str, required=True,
                        help="Human-readable repo name (created on first use)")
    parser.add_argument("--dsn", type=str, default=None, help="Postgres DSN (defaults to DATABASE_URL env)")
    parser.add_argument("--init-schema", action="store_true", help="Apply migrations before indexing (idempotent)")
    parser.add_argument("--no-resolve", action="store_true", help="Skip Tier 2 semantic resolution")
    parser.add_argument("--no-imports", action="store_true", help="Skip Phase 3 import resolution / cross-file linking")
    parser.add_argument("--no-chunks", action="store_true", help="Skip Tier 3 chunk assembly")
    parser.add_argument("--embed", choices=["real", "fake"], default=None,
                        help="Run embedding step. 'real' uses EMBEDDING_BASE_URL/API_KEY/MODEL; 'fake' is the deterministic stub.")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        type=lambda s: [p.strip() for p in s.split(",") if p.strip()],
                        help="Glob(s) to skip. Repeatable; comma-separates also fine. "
                             "No-slash patterns match any path component; "
                             "with-slash patterns match the full repo-relative path.")
    parser.add_argument("--no-tsgrepignore", action="store_true",
                        help="Don't auto-load .tsgrepignore from the repo root.")
    args = parser.parse_args(argv)

    if not args.repo_path.is_dir():
        print(f"error: {args.repo_path} is not a directory", file=sys.stderr)
        return 2

    return asyncio.run(
        _run(
            args.repo_path,
            args.repo_name,
            args.dsn,
            args.init_schema,
            not args.no_resolve,
            not args.no_imports,
            not args.no_chunks,
            args.embed,
            tuple(p for group in args.exclude for p in group),
            not args.no_tsgrepignore,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
