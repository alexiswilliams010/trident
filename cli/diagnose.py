"""CLI: print resolution stats for a repo.

Usage:
    python -m cli.diagnose --repo-name myrepo
    python -m cli.diagnose --repo-name myrepo --unresolved        # list unresolved imports
    python -m cli.diagnose --repo-name myrepo --top-callers 10    # top callers by # of edges
"""

from __future__ import annotations

import argparse
import asyncio

from db.connection import pool_ctx


async def _print_stats(pool, repo_id: int, show_unresolved: bool, top_callers: int) -> int:
    async with pool.acquire() as conn:
        files = await conn.fetchval("SELECT COUNT(*) FROM files WHERE repo_id=$1", repo_id)
        if not files:
            print(f"no files for repo_id={repo_id}; nothing to diagnose")
            return 1
        nodes = await conn.fetchval(
            "SELECT COUNT(*) FROM nodes n JOIN files f ON f.id=n.file_id WHERE f.repo_id=$1",
            repo_id,
        )
        defs = await conn.fetchval(
            "SELECT COUNT(*) FROM definitions d JOIN files f ON f.id=d.file_id WHERE f.repo_id=$1",
            repo_id,
        )
        refs_total = await conn.fetchval(
            'SELECT COUNT(*) FROM "references" r JOIN files f ON f.id=r.file_id WHERE f.repo_id=$1',
            repo_id,
        )
        refs_resolved = await conn.fetchval(
            'SELECT COUNT(*) FROM "references" r JOIN files f ON f.id=r.file_id '
            "WHERE f.repo_id=$1 AND r.target_def_id IS NOT NULL",
            repo_id,
        )

        imp_classes = await conn.fetch(
            "SELECT i.dep_class, COUNT(*) AS n "
            "FROM imports i JOIN files f ON f.id=i.file_id "
            "WHERE f.repo_id=$1 GROUP BY i.dep_class",
            repo_id,
        )
        imp_by = {r["dep_class"]: r["n"] for r in imp_classes}
        imp_total = sum(imp_by.values()) or 1

        ce_total = await conn.fetchval(
            "SELECT COUNT(*) FROM call_edges ce "
            "JOIN definitions caller ON caller.id=ce.caller_def_id "
            "JOIN files f ON f.id=caller.file_id WHERE f.repo_id=$1",
            repo_id,
        )
        ce_by_conf = await conn.fetch(
            "SELECT ce.confidence, COUNT(*) AS n FROM call_edges ce "
            "JOIN definitions caller ON caller.id=ce.caller_def_id "
            "JOIN files f ON f.id=caller.file_id WHERE f.repo_id=$1 "
            "GROUP BY ce.confidence",
            repo_id,
        )
        ce_conf = {r["confidence"]: r["n"] for r in ce_by_conf}

        print(f"=== repo_id={repo_id} ===")
        print(f"files:        {files}")
        print(f"nodes:        {nodes}")
        print(f"definitions:  {defs}")
        ref_pct = 100 * (refs_resolved or 0) // (refs_total or 1)
        print(f"references:   {refs_total}  ({refs_resolved} resolved, {ref_pct}%)")
        print(
            f"imports:      {sum(imp_by.values())}  "
            f"intra_repo={imp_by.get('intra_repo', 0)} "
            f"external={imp_by.get('external', 0)} "
            f"unresolved={imp_by.get('unresolved', 0)} "
            f"({100 * imp_by.get('intra_repo', 0) // imp_total}% intra-repo)"
        )
        print(
            f"call_edges:   {ce_total}  "
            f"certain={ce_conf.get('certain', 0)} "
            f"inferred={ce_conf.get('inferred', 0)} "
            f"uncertain={ce_conf.get('uncertain', 0)}"
        )

        ext = await conn.fetch(
            "SELECT package_name, language FROM external_dependencies "
            "WHERE repo_id=$1 ORDER BY language, package_name",
            repo_id,
        )
        if ext:
            print("external_dependencies:")
            for row in ext:
                print(f"  {row['language']:10s}  {row['package_name']}")

        if show_unresolved:
            unresolved = await conn.fetch(
                "SELECT f.path, i.import_path FROM imports i "
                "JOIN files f ON f.id=i.file_id "
                "WHERE f.repo_id=$1 AND i.dep_class='unresolved' "
                "ORDER BY f.path, i.import_path",
                repo_id,
            )
            print(f"unresolved imports ({len(unresolved)}):")
            for row in unresolved:
                print(f"  {row['path']} → {row['import_path']}")

        if top_callers > 0:
            rows = await conn.fetch(
                """
                SELECT caller.qualified_name AS caller, COUNT(*) AS n
                FROM call_edges ce
                JOIN definitions caller ON caller.id=ce.caller_def_id
                JOIN files f ON f.id=caller.file_id
                WHERE f.repo_id=$1
                GROUP BY caller.qualified_name
                ORDER BY n DESC LIMIT $2
                """,
                repo_id,
                top_callers,
            )
            if rows:
                print(f"top {top_callers} callers by edge count:")
                for row in rows:
                    print(f"  {row['n']:4d}  {row['caller']}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="trident — diagnostics for an indexed repo")
    parser.add_argument("--repo-name", type=str, required=True,
                        help="Repo name (must already be indexed)")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--unresolved", action="store_true", help="List unresolved imports")
    parser.add_argument("--top-callers", type=int, default=0, help="Show N most-active callers")
    args = parser.parse_args(argv)

    async def _run() -> int:
        from cli._repo import resolve_repo_id
        async with pool_ctx(args.dsn) as pool:
            repo_id = await resolve_repo_id(pool, name=args.repo_name, create=False)
            return await _print_stats(pool, repo_id, args.unresolved, args.top_callers)

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
