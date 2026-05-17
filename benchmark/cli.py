"""CLI: `python -m benchmark <repo_path> --question "..." --repo-name <name>`

See `benchmark/__init__.py` for the high-level goal.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from benchmark.report import _timestamp, print_summary, write_json, write_markdown
from benchmark.runner import default_db_name, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Compare a trident-enabled Claude Code agent against a baseline agent.",
    )
    parser.add_argument("repo_path", type=Path, help="Absolute path to the repo to benchmark.")
    parser.add_argument("--question", required=True, help="The question both agents will answer.")
    parser.add_argument(
        "--repo-name", required=True,
        help="Trident's name for the repo (created on first run).",
    )
    parser.add_argument("--branch", default="main", help="Branch name (default: main).")
    parser.add_argument(
        "--db", default=None,
        help="Postgres DB name. Default: trident_bench_<sha8(repo_path)>.",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Model ID passed to claude-agent-sdk for both agents.",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_runs", type=Path,
        help="Directory to write the JSON report (default: ./benchmark_runs).",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Skip the embedding step (queries that need vectors will fail).",
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Pass force_rehash=True to the extractor; useful when content hash didn't change.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Stream every tool call and tool result from both agents to stdout.",
    )
    args = parser.parse_args(argv)

    if not args.repo_path.is_dir():
        print(f"error: {args.repo_path} is not a directory", file=sys.stderr)
        return 2

    db_name = args.db or default_db_name(args.repo_path)

    result = asyncio.run(
        run_benchmark(
            repo_path=args.repo_path,
            question=args.question,
            repo_name=args.repo_name,
            branch=args.branch,
            db_name=db_name,
            model=args.model,
            skip_embed=args.no_embed,
            force_rehash=args.force_reindex,
            verbose=args.verbose,
        )
    )
    print_summary(result)
    stamp = _timestamp()
    json_path = write_json(result, args.output_dir, stamp=stamp)
    md_path = write_markdown(result, args.output_dir, stamp=stamp)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
