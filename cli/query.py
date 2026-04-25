"""CLI: retrieve chunks from an indexed repo.

    python -m cli.query --repo-id 1 --semantic "how does helper resolve?"
    python -m cli.query --repo-id 1 --structural withdraw --depth 2
    python -m cli.query --repo-id 1 --hybrid "reentrancy guard usage"
    python -m cli.query --repo-id 1 --semantic "..." --fake   # uses deterministic stub embedder

By default the semantic / hybrid modes use the OpenAI-compatible gateway
configured by EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL.
Pass --fake to use the deterministic stub embedder (handy for sanity checks
before signing up for an API key).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from core.embedder import EmbedderConfig, OpenAICompatibleEmbedder, make_fake_embedder
from core.retrieval import (
    RetrievedChunk,
    assemble_context,
    hybrid_query,
    semantic_query,
    structural_query,
)
from db.connection import pool_ctx


def _print_chunks(chunks: list[RetrievedChunk], show_content: bool, content_chars: int) -> None:
    for i, c in enumerate(chunks, 1):
        print(f"[{i:02d}] score={c.score:.3f}  {c.granularity:13s}  "
              f"{c.qualified_name or '<module>'}  ({c.file_path}, {c.token_count} tok)")
        if show_content:
            snippet = c.content if len(c.content) <= content_chars else c.content[:content_chars] + "…"
            print(snippet)
            print("─" * 80)


async def _run(args: argparse.Namespace) -> int:
    if args.fake:
        embed_fn, _ = make_fake_embedder()
    elif args.semantic or args.hybrid:
        try:
            cfg = EmbedderConfig.from_env()
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            print("hint: pass --fake for a deterministic stub embedder.", file=sys.stderr)
            return 2
        embed_fn = OpenAICompatibleEmbedder(cfg).embed
    else:
        embed_fn = None

    async with pool_ctx(args.dsn) as pool:
        if args.structural:
            chunks = await structural_query(
                pool, args.repo_id, args.structural,
                depth=args.depth, granularity=args.granularity,
            )
        elif args.semantic:
            assert embed_fn is not None
            chunks = await semantic_query(
                pool, args.repo_id, args.semantic, embed_fn,
                top_k=args.top_k,
                granularities=tuple(args.granularity.split(",")) if args.granularity else None,
            )
        elif args.hybrid:
            assert embed_fn is not None
            chunks = await hybrid_query(pool, args.repo_id, args.hybrid, embed_fn, top_k=args.top_k)
        else:
            print("error: must pass one of --semantic / --structural / --hybrid", file=sys.stderr)
            return 2

    if not chunks:
        print("no results")
        return 0

    _print_chunks(chunks, args.show_content, args.content_chars)

    if args.context_budget > 0:
        ctx = assemble_context(chunks, args.context_budget)
        ctx_tokens = sum(c.token_count for c in chunks if c.content in ctx)
        print()
        print(f"--- assembled context ({ctx_tokens} tok, ≤{args.context_budget}) ---")
        print(ctx)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsgrep — retrieve chunks")
    parser.add_argument("--repo-id", type=int, required=True)
    parser.add_argument("--dsn", type=str, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--semantic", type=str, help="natural-language query")
    mode.add_argument("--structural", type=str, help="definition name or qualified name")
    mode.add_argument("--hybrid", type=str, help="natural-language query (semantic + graph expand)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2, help="structural traversal depth")
    parser.add_argument("--granularity", type=str, default="function",
                        help="comma-separated for semantic; single for structural (default: function)")
    parser.add_argument("--fake", action="store_true",
                        help="use deterministic stub embedder (no API key)")
    parser.add_argument("--show-content", action="store_true", help="print chunk bodies")
    parser.add_argument("--content-chars", type=int, default=600,
                        help="truncate each chunk body at N chars")
    parser.add_argument("--context-budget", type=int, default=0,
                        help="if >0, also print the assembled-context output under this token budget")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
