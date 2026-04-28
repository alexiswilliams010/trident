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

Each chunk produces multiple embeddings — one per "view" (see
`core/embed_views.py`). At retrieval time the lateral picks the best-scoring
view per chunk. The embedder skips re-embedding any (chunk, view, model)
triple whose `input_text_hash` is unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import asyncpg

from .chunk_assembler import count_tokens
from .embed_views import VIEW_KINDS, build_views

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
    embedded: int = 0           # number of (chunk, view) rows newly embedded
    skipped_unchanged: int = 0  # rows whose input_text_hash was already current
    skipped_oversize: list[tuple[int, str, int]] = field(default_factory=list)
    # (chunk_id, view_kind, token_count)


# Conservative ceiling — well under typical embedding-model context windows
# (qwen3-embedding-8b: 32k, OpenAI v3: 8191, Voyage: 32k). Per-view rows
# above this are skipped rather than failing the whole run; in practice they
# come from vendored minified bundles or pathological generated code.
DEFAULT_MAX_INPUT_TOKENS = 8000


def _vector_literal(values: list[float]) -> str:
    """pgvector accepts a string of the form '[v1,v2,...]'."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


def _hash_input(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


async def embed_repo_chunks(
    pool: asyncpg.Pool,
    repo_id: int,
    embed_fn: EmbedFn,
    model_name: str,
    *,
    dim: int = 4096,
    batch_size: int = 64,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    view_kinds: tuple[str, ...] = VIEW_KINDS,
) -> EmbedStats:
    """Embed every (chunk, view) pair in `repo_id` whose input_text_hash is
    missing or stale.

    `embed_fn(texts) -> vectors` is the abstraction; production passes
    `OpenAICompatibleEmbedder(...).embed`, tests pass a deterministic stub.

    For each chunk, `build_views` (in `core/embed_views.py`) computes one
    text per view kind. We hash each view's text and compare against the
    existing row for `(chunk_id, view_kind, model_name)`; rows whose hash
    matches the current text are skipped. Rows whose hash differs are
    re-embedded and UPSERTed.
    """
    stats = EmbedStats()
    async with pool.acquire() as conn:
        chunk_rows = await conn.fetch(
            """
            SELECT c.id, c.content, c.anchor_def_id, c.granularity
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE f.repo_id = $1
            ORDER BY c.id
            """,
            repo_id,
        )
        stats.chunks_seen = len(chunk_rows)
        if not chunk_rows:
            return stats

        existing = await conn.fetch(
            """
            SELECT ce.chunk_id, ce.view_kind, ce.input_text_hash
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE f.repo_id = $1 AND ce.model_name = $2
            """,
            repo_id, model_name,
        )
        existing_hash: dict[tuple[int, str], str | None] = {
            (r["chunk_id"], r["view_kind"]): r["input_text_hash"] for r in existing
        }

        # Pending = list of dicts with chunk_id, view_kind, text, text_hash.
        pending: list[dict] = []
        for cr in chunk_rows:
            views = await build_views(
                conn, cr["id"], cr["content"], cr["anchor_def_id"], cr["granularity"],
            )
            for vk in view_kinds:
                text = views.get(vk)
                if text is None:
                    continue
                h = _hash_input(text)
                prev = existing_hash.get((cr["id"], vk))
                if prev == h:
                    stats.skipped_unchanged += 1
                    continue
                tc = count_tokens(text)
                if tc > max_input_tokens:
                    stats.skipped_oversize.append((cr["id"], vk, tc))
                    continue
                pending.append({
                    "chunk_id": cr["id"],
                    "view_kind": vk,
                    "text": text,
                    "hash": h,
                })

        if stats.skipped_oversize:
            print(
                f"[Tier 3 embed] skipping {len(stats.skipped_oversize)} oversize "
                f"view(s) > {max_input_tokens} tokens "
                f"(largest: {max(t for _, _, t in stats.skipped_oversize)})"
            )

        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            texts = [item["text"] for item in batch]
            vectors = await embed_fn(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embed_fn returned {len(vectors)} vectors for batch of {len(batch)}"
                )
            insert_rows = []
            for item, vec in zip(batch, vectors):
                if len(vec) != dim:
                    raise RuntimeError(
                        f"embedding dim mismatch: got {len(vec)}, expected {dim} "
                        f"(chunk {item['chunk_id']}, view {item['view_kind']})"
                    )
                insert_rows.append((
                    item["chunk_id"],
                    item["view_kind"],
                    _vector_literal(vec),
                    model_name,
                    item["text"],
                    item["hash"],
                ))
            await conn.executemany(
                """
                INSERT INTO chunk_embeddings
                    (chunk_id, view_kind, embedding, model_name, input_text, input_text_hash)
                VALUES ($1, $2, $3::vector, $4, $5, $6)
                ON CONFLICT (chunk_id, view_kind, model_name)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    input_text = EXCLUDED.input_text,
                    input_text_hash = EXCLUDED.input_text_hash
                """,
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
