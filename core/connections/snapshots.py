"""Capture committed Git trees without trusting mutable branch indexes or disk files."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import UUID

import asyncpg

from .contracts import RepoRevision, Snapshot


def fingerprint(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RepositoryInput:
    repo_id: int
    root: Path
    revision: str = "HEAD"


@dataclass(frozen=True)
class CapturedFile:
    path: str
    mode: str
    oid: str
    content: bytes

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], input=input, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(
            f"Git snapshot read failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def read_repository(repo: RepositoryInput) -> tuple[str, list[CapturedFile]]:
    revision = (
        _git(repo.root, "rev-parse", "--verify", "--end-of-options", f"{repo.revision}^{{commit}}")
        .decode()
        .strip()
    )
    entries = []
    for entry in _git(repo.root, "ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        entries.append((path.decode("utf-8"), mode, kind, oid))
    # Batch by object ID, not filename: tabs, spaces, and newlines in paths cannot
    # change the command syntax. Symlinks are read as link text, never followed.
    oids = sorted({oid for _, _, kind, oid in entries if kind == "blob"})
    data = _git(
        repo.root, "cat-file", "--batch", input="".join(f"{oid}\n" for oid in oids).encode()
    )
    blobs, offset = {}, 0
    for expected in oids:
        end = data.index(b"\n", offset)
        oid, kind, size = data[offset:end].decode("ascii").split()
        if oid != expected or kind != "blob":
            raise ValueError("Unexpected Git object in snapshot")
        offset = end + 1
        size = int(size)
        blobs[oid] = data[offset : offset + size]
        if len(blobs[oid]) != size or data[offset + size : offset + size + 1] != b"\n":
            raise ValueError("Truncated Git object in snapshot")
        offset += size + 1
    # Gitlinks retain the pinned submodule commit, not its working-tree contents.
    return revision, [
        CapturedFile(path, mode, oid, blobs[oid] if kind == "blob" else oid.encode())
        for path, mode, kind, oid in entries
    ]


async def capture_snapshot(
    pool: asyncpg.Pool,
    repositories: tuple[RepositoryInput, ...],
    *,
    context: dict,
) -> Snapshot:
    """Capture exact commits, including config/build files and binary blobs.

    Context is fingerprinted, not persisted verbatim. Working-tree changes and
    untracked files are excluded. Callers must match parser inputs by content hash.
    """
    ids = [repo.repo_id for repo in repositories]
    if len(set(ids)) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Snapshot capture requires at least two distinct repositories, once each")
    if len({repo.root.resolve() for repo in repositories}) != len(repositories):
        raise ValueError("Different repository IDs cannot refer to the same checkout")
    repositories = tuple(sorted(repositories, key=lambda r: r.repo_id))
    common_dirs = await asyncio.gather(
        *(
            asyncio.to_thread(
                _git, repo.root, "rev-parse", "--path-format=absolute", "--git-common-dir"
            )
            for repo in repositories
        )
    )
    if len({Path(value.decode().strip()).resolve() for value in common_dirs}) != len(repositories):
        raise ValueError(
            "Worktrees of the same Git repository do not qualify as distinct repositories"
        )
    captured = await asyncio.gather(
        *(asyncio.to_thread(read_repository, repo) for repo in repositories)
    )
    members = tuple(
        RepoRevision(repo.repo_id, revision) for repo, (revision, _) in zip(repositories, captured)
    )
    context_hash = fingerprint(context)
    key = fingerprint(
        {"members": [(m.repo_id, m.revision) for m in members], "context": context_hash}
    )
    async with pool.acquire() as conn, conn.transaction():
        # Canonical database identity must not be silently assigned to another checkout.
        for repo in repositories:
            root = await conn.fetchval("SELECT root_path FROM repos WHERE id=$1", repo.repo_id)
            if root is None or Path(root).resolve() != repo.root.resolve():
                raise ValueError("Repository root does not match its registered canonical ID")
        sid = await conn.fetchval(
            """
            INSERT INTO connection_snapshots(context_fingerprint, snapshot_key)
            VALUES ($1,$2) ON CONFLICT (snapshot_key) DO NOTHING RETURNING id
        """,
            context_hash,
            key,
        )
        if sid is None:
            row = await conn.fetchrow(
                "SELECT id, sealed_at FROM connection_snapshots WHERE snapshot_key=$1", key
            )
            if row["sealed_at"] is None:
                raise ValueError("An unsealed snapshot already owns this capture identity")
            return Snapshot(str(row["id"]), members, context_hash)
        for repo, (revision, files) in zip(repositories, captured):
            await conn.execute(
                "INSERT INTO connection_snapshot_members VALUES ($1,$2,$3)",
                sid,
                repo.repo_id,
                revision,
            )
            blobs = {file.content_hash: file.content for file in files}
            await conn.executemany(
                """
                INSERT INTO connection_source_blobs VALUES ($1,$2)
                ON CONFLICT (content_hash) DO NOTHING
            """,
                blobs.items(),
            )
            await conn.executemany(
                """
                INSERT INTO connection_snapshot_files
                    (snapshot_id, repo_id, path, mode, git_oid, content_hash)
                VALUES ($1,$2,$3,$4,$5,$6)
            """,
                [(sid, repo.repo_id, f.path, f.mode, f.oid, f.content_hash) for f in files],
            )
        await conn.execute("UPDATE connection_snapshots SET sealed_at=now() WHERE id=$1", sid)
    return Snapshot(str(sid), members, context_hash)


async def load_snapshot(conn: asyncpg.Connection, snapshot_id: str | UUID) -> Snapshot:
    sid = UUID(str(snapshot_id))
    row = await conn.fetchrow("SELECT * FROM connection_snapshots WHERE id=$1", sid)
    if row is None or row["sealed_at"] is None:
        raise ValueError("A sealed captured snapshot is required")
    members = await conn.fetch(
        """
        SELECT repo_id, revision FROM connection_snapshot_members
        WHERE snapshot_id=$1 ORDER BY repo_id
    """,
        sid,
    )
    snapshot = Snapshot(
        str(sid),
        tuple(RepoRevision(r["repo_id"], r["revision"]) for r in members),
        row["context_fingerprint"],
    )
    snapshot.require_cross_repo()
    return snapshot


async def read_source(
    conn: asyncpg.Connection, snapshot_id: str | UUID, repo_id: int, path: str
) -> bytes:
    await load_snapshot(conn, snapshot_id)
    raw = await conn.fetchval(
        """
        SELECT b.content FROM connection_snapshot_files f
        JOIN connection_source_blobs b USING (content_hash)
        WHERE f.snapshot_id=$1 AND f.repo_id=$2 AND f.path=$3 AND f.mode <> '160000'
    """,
        UUID(str(snapshot_id)),
        repo_id,
        path,
    )
    if raw is None:
        raise ValueError("Source file is absent from the captured snapshot")
    return bytes(raw)
