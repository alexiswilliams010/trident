"""Tier 3b: chunk embedding via an OpenAI-compatible HTTP gateway.

Configurable via env vars so any compatible gateway (LiteLLM, vLLM, Voyage's
OpenAI-compat endpoint, OpenAI itself) can be slotted in:

    EMBEDDING_BASE_URL   e.g. https://api.openai.com/v1
    EMBEDDING_API_KEY
    EMBEDDING_MODEL      e.g. text-embedding-3-large, voyage-code-3, nomic-embed-code
    EMBEDDING_DIM        defaults to 1024 (matches schema vector(1024))

Branch model: `chunk_embeddings` is keyed by chunk content_hash. Two
chunks (across branches) with byte-identical content share one embedding
row. When indexing a feature branch whose content is mostly identical to
main, the SELECT below finds existing embeddings and the API call count
drops to zero for the unchanged chunks.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

import asyncpg

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - dev deps installed by uv
    AsyncOpenAI = None  # type: ignore

try:
    from pgvector.asyncpg import register_vector
except ImportError:  # pragma: no cover
    register_vector = None  # type: ignore


# Either an Embedder instance or a plain async callable that maps texts → vectors.
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class EmbedderConfig:
    base_url: str
    api_key: str
    model: str
    dim: int = 1024
    batch_size: int = 64

    @classmethod
    def from_env(cls) -> "EmbedderConfig":
        base_url = os.environ.get("EMBEDDING_BASE_URL")
        api_key = os.environ.get("EMBEDDING_API_KEY")
        model = os.environ.get("EMBEDDING_MODEL")
        if not base_url or not api_key or not model:
            raise RuntimeError(
                "Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL to enable embedding."
            )
        dim = int(os.environ.get("EMBEDDING_DIM", "1024"))
        batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
        return cls(base_url=base_url, api_key=api_key, model=model, dim=dim, batch_size=batch_size)


class OpenAICompatibleEmbedder:
    """Calls `client.embeddings.create(model=..., input=batch)` per the OpenAI HTTP shape."""

    def __init__(self, config: EmbedderConfig):
        if AsyncOpenAI is None:
            raise RuntimeError("openai SDK not installed")
        self.config = config
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    @property
    def model_name(self) -> str:
        return self.config.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict = {"model": self.config.model, "input": texts}
        # OpenAI v3 supports `dimensions`; gateways that ignore it just return native dim.
        if "openai.com" in self.config.base_url or self.config.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.config.dim
        resp = await self._client.embeddings.create(**kwargs)
        # SDK returns Pydantic models with .data[i].embedding (list[float]).
        return [list(item.embedding) for item in resp.data]


# ────────────────────────────────────────────────────────────────────
# Persistence loop
# ────────────────────────────────────────────────────────────────────


@dataclass
class EmbedStats:
    chunks_seen: int = 0
    embedded: int = 0
    skipped: int = 0
    skipped_oversize: list[tuple[int, int]] = None  # (chunk_id, token_count)

    def __post_init__(self):
        if self.skipped_oversize is None:
            self.skipped_oversize = []


# Outer ceiling that filters chunks before sending to the embedder gateway.
DEFAULT_MAX_INPUT_TOKENS = int(os.environ.get("EMBEDDING_MAX_INPUT_TOKENS", "16000"))

# How many embedding API calls to keep in flight concurrently. API latency
# (200–500 ms per batch) dominates throughput, so 2–4 concurrent calls
# typically give a near-linear speedup without tripping rate limits.
DEFAULT_EMBEDDING_CONCURRENCY = int(os.environ.get("EMBEDDING_CONCURRENCY", "3"))


def _vector_literal(values: list[float]) -> str:
    """pgvector accepts a string of the form '[v1,v2,...]'."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


async def embed_branch_chunks(
    pool: asyncpg.Pool,
    branch_id: int,
    embed_fn: EmbedFn,
    model_name: str,
    *,
    dim: int = 4096,
    batch_size: int = 64,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> EmbedStats:
    """Embed every chunk in `branch_id` whose content_hash isn't yet in
    chunk_embeddings.

    `chunk_embeddings` is keyed by content_hash (not chunk_id), so two
    chunks with the same content text — typical across branches that share
    most files — produce a single embedding row. Re-running this on a
    feature branch identical to main yields zero API calls.

    `embed_fn(texts) -> vectors` is the abstraction; production passes
    `OpenAICompatibleEmbedder(...).embed`, tests pass a deterministic stub.

    Chunks whose `token_count` exceeds `max_input_tokens` are skipped rather
    than sent to the gateway: most embedding endpoints reject inputs above
    their context window with HTTP 422.
    """
    stats = EmbedStats()
    async with pool.acquire() as conn:
        # DISTINCT ON content_hash so each unique chunk text is embedded once
        # even if multiple chunks in this branch share it (rare — typically
        # only the trivial-empty case — but easy to handle).
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (c.content_hash)
                   c.id, c.content, c.token_count, c.content_hash
            FROM chunks c
            LEFT JOIN chunk_embeddings ce ON ce.content_hash = c.content_hash
            WHERE c.branch_id = $1 AND ce.id IS NULL
            ORDER BY c.content_hash, c.id
            """,
            branch_id,
        )
        stats.chunks_seen = len(rows)
        if not rows:
            return stats

        # Filter oversize chunks up front. We still report each one so the
        # operator can see what's being dropped (typically vendored bundles
        # like *.min.js or pathological generated code).
        eligible: list = []
        for r in rows:
            tc = r["token_count"] or 0
            if tc > max_input_tokens:
                stats.skipped_oversize.append((r["id"], tc))
                stats.skipped += 1
            else:
                eligible.append(r)
        if stats.skipped_oversize:
            print(
                f"[Tier 3 embed] skipping {len(stats.skipped_oversize)} oversize "
                f"chunk(s) > {max_input_tokens} tokens "
                f"(largest: {max(t for _, t in stats.skipped_oversize)})"
            )

        sem = asyncio.Semaphore(DEFAULT_EMBEDDING_CONCURRENCY)

        async def _embed_one(batch: list) -> tuple[list, list[list[float]]]:
            async with sem:
                texts = [r["content"] for r in batch]
                vectors = await embed_fn(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embed_fn returned {len(vectors)} vectors for batch of {len(batch)}"
                )
            for r, vec in zip(batch, vectors):
                if len(vec) != dim:
                    raise RuntimeError(
                        f"embedding dim mismatch: got {len(vec)}, expected {dim} (chunk {r['id']})"
                    )
            return batch, vectors

        tasks = [
            asyncio.create_task(_embed_one(eligible[i : i + batch_size]))
            for i in range(0, len(eligible), batch_size)
        ]

        all_hashes: list[str] = []
        all_vectors: list[str] = []
        try:
            for coro in asyncio.as_completed(tasks):
                batch, vectors = await coro
                for r, vec in zip(batch, vectors):
                    all_hashes.append(r["content_hash"])
                    all_vectors.append(_vector_literal(vec))
                stats.embedded += len(batch)
        except BaseException:
            # On any failure, cancel the remaining in-flight API calls so we
            # don't keep burning the gateway after a fatal validation error.
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

        if all_hashes:
            await conn.execute(
                """
                INSERT INTO chunk_embeddings (content_hash, embedding, model_name)
                SELECT content_hash, embedding::vector, $3
                FROM UNNEST($1::text[], $2::text[]) AS t(content_hash, embedding)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                all_hashes,
                all_vectors,
                model_name,
            )
    return stats


# Backwards-compatible alias for the old name.
embed_repo_chunks = embed_branch_chunks


def embed_branch_sync(
    branch_id: int,
    embed_fn: EmbedFn,
    model_name: str,
    *,
    dim: int = 4096,
    dsn: str | None = None,
) -> EmbedStats:
    from db.connection import pool_ctx

    async def _run():
        async with pool_ctx(dsn) as pool:
            return await embed_branch_chunks(pool, branch_id, embed_fn, model_name, dim=dim)

    return asyncio.run(_run())


# Backwards-compatible alias.
embed_repo_sync = embed_branch_sync


# ────────────────────────────────────────────────────────────────────
# Test helper: deterministic fake embedder (no network)
# ────────────────────────────────────────────────────────────────────


def make_fake_embedder(dim: int = 4096) -> tuple[EmbedFn, str]:
    """Hash-based deterministic embedder. Used by Phase 4 retrieval tests so
    the suite doesn't need an API key. Same input always → same vector."""

    import hashlib

    def _vec_from_text(text: str) -> list[float]:
        # Use SHA-256 to seed a pseudo-random sequence; spread across `dim`.
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
        out: list[float] = []
        i = 0
        while len(out) < dim:
            chunk = h[i : i + 4]
            if len(chunk) < 4:
                # Re-hash to extend.
                h = hashlib.sha256(h).digest()
                i = 0
                continue
            v = int.from_bytes(chunk, "big") / 2**32  # 0..1
            out.append(v - 0.5)  # centre at 0
            i += 4
        # Normalise so cosine similarity is well-defined.
        norm = sum(x * x for x in out) ** 0.5 or 1.0
        return [x / norm for x in out]

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [_vec_from_text(t) for t in texts]

    return _embed, "fake-deterministic"
