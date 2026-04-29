"""Tier 3b: chunk embedding via an OpenAI-compatible HTTP gateway.

Configurable via env vars so any compatible gateway (LiteLLM, vLLM, Voyage's
OpenAI-compat endpoint, OpenAI itself) can be slotted in:

    EMBEDDING_BASE_URL   e.g. https://api.openai.com/v1
    EMBEDDING_API_KEY
    EMBEDDING_MODEL      e.g. text-embedding-3-large, voyage-code-3, nomic-embed-code
    EMBEDDING_DIM        defaults to 1024 (matches schema vector(1024))

For models with native dim != 1024, the gateway/SDK is expected to truncate
or project to 1024 (OpenAI v3 supports `dimensions=1024`; Voyage code 3 is
natively 1024).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

import asyncpg

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dev deps installed by uv
    OpenAI = None  # type: ignore

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
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed")
        self.config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

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
        # The OpenAI client is sync; offload to a thread.
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, lambda: self._client.embeddings.create(**kwargs))
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
# Sized to comfortably exceed our per-granularity `HARD_OUTPUT_CAP` (largest
# is cross-module at 8000) while staying under typical model context windows
# (qwen3-embedding-8b: 32k, OpenAI v3: 8191, Voyage: 32k). Chunks above this
# are skipped rather than failing the whole run; in practice they come from
# vendored minified bundles or pathological generated code that has no
# semantic value to index anyway. Override via `EMBEDDING_MAX_INPUT_TOKENS`
# for models with smaller context (e.g. OpenAI v3 at 8191).
DEFAULT_MAX_INPUT_TOKENS = int(os.environ.get("EMBEDDING_MAX_INPUT_TOKENS", "16000"))


def _vector_literal(values: list[float]) -> str:
    """pgvector accepts a string of the form '[v1,v2,...]'."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


async def embed_repo_chunks(
    pool: asyncpg.Pool,
    repo_id: int,
    embed_fn: EmbedFn,
    model_name: str,
    *,
    dim: int = 4096,
    batch_size: int = 64,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> EmbedStats:
    """Embed every chunk in `repo_id` that doesn't already have an embedding.

    `embed_fn(texts) -> vectors` is the abstraction; production passes
    `OpenAICompatibleEmbedder(...).embed`, tests pass a deterministic stub.

    Chunks whose `token_count` exceeds `max_input_tokens` are skipped rather
    than sent to the gateway: most embedding endpoints reject inputs above
    their context window with HTTP 422, which would otherwise abort the
    whole run. Skipped chunks are reported on `EmbedStats.skipped_oversize`.
    """
    stats = EmbedStats()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.content, c.token_count
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
            WHERE f.repo_id = $1 AND ce.id IS NULL
            ORDER BY c.id
            """,
            repo_id,
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

        for i in range(0, len(eligible), batch_size):
            batch = eligible[i : i + batch_size]
            texts = [r["content"] for r in batch]
            vectors = await embed_fn(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embed_fn returned {len(vectors)} vectors for batch of {len(batch)}"
                )
            insert_rows = []
            for r, vec in zip(batch, vectors):
                if len(vec) != dim:
                    raise RuntimeError(
                        f"embedding dim mismatch: got {len(vec)}, expected {dim} (chunk {r['id']})"
                    )
                insert_rows.append((r["id"], _vector_literal(vec), model_name))
            await conn.executemany(
                "INSERT INTO chunk_embeddings (chunk_id, embedding, model_name) "
                "VALUES ($1, $2::vector, $3)",
                insert_rows,
            )
            stats.embedded += len(batch)
    return stats


def embed_repo_sync(
    repo_id: int,
    embed_fn: EmbedFn,
    model_name: str,
    *,
    dim: int = 4096,
    dsn: str | None = None,
) -> EmbedStats:
    from db.connection import pool_ctx

    async def _run():
        async with pool_ctx(dsn) as pool:
            return await embed_repo_chunks(pool, repo_id, embed_fn, model_name, dim=dim)

    return asyncio.run(_run())


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
