# Future plan: switch `chunk_embeddings` to `halfvec(4000)` + HNSW

## Why this is deferred

The current schema stores `vector(4096)` to match qwen3-embedding-8b's native
output. pgvector's HNSW index caps at 2000 dimensions for `vector` and 4000
for `halfvec`, so the index was dropped — semantic queries fall back to a
sequential scan over `chunk_embeddings`.

Seqscan is fine at small/medium scale: a single repo's worth of chunks
(hundreds to low thousands) finishes in milliseconds. We only need an ANN
index when query latency starts to matter — roughly when chunk count climbs
past ~10K and/or the retriever runs in a hot loop.

## Trigger

Run this migration when **any** of the following holds:

- `SELECT count(*) FROM chunk_embeddings` exceeds ~10K
- `EXPLAIN ANALYZE` on `semantic_query` shows >50ms average
- An agentic workflow makes >10 retrieval calls per task and end-to-end
  latency becomes a complaint

## What it takes

### 1. Confirm pgvector version supports `halfvec`

```
psql -d tsgrep -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

`halfvec` requires pgvector ≥ 0.7.0. Homebrew's `postgresql@18` ships a
recent build, so this is almost certainly fine. If older, upgrade the
extension first.

### 2. Add a forward migration `db/migrations/0002_halfvec_hnsw.up.sql`

```sql
ALTER TABLE chunk_embeddings
    ALTER COLUMN embedding TYPE halfvec(4000)
    USING embedding::halfvec(4000);

CREATE INDEX idx_chunk_embeddings_vector ON chunk_embeddings
    USING hnsw (embedding halfvec_cosine_ops);
```

The `USING ...::halfvec(4000)` cast truncates each existing 4096-dim vector
to its first 4000 dims. Qwen3 is trained with Matryoshka representation
learning, so the leading dims are themselves a usable embedding — quality
loss is in the noise.

### 3. Truncate + renormalize at embed time

In `core/embedder.py`, after `client.embeddings.create(...)` returns,
truncate each vector to 4000 dims and L2-renormalize:

```python
def _truncate_normalize(vec: list[float], target: int) -> list[float]:
    v = vec[:target]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]
```

Renormalization isn't strictly required for cosine (`<=>` divides by
magnitude internally) but keeps the data clean if we ever switch to inner
product (`<#>`) or L2 (`<->`).

### 4. Update casts and config

- `core/embedder.py:154`: `$2::vector` → `$2::halfvec`
- `core/retrieval.py` (two SELECTs): `$1::vector` → `$1::halfvec`
- `.env.template`: `EMBEDDING_DIM=4096` → `4000`
- `core/embedder.py` config defaults / docstring: bump to 4000
- `make_fake_embedder` default dim: bump to 4000 (or pass explicitly from
  tests)

The pgvector text literal `'[v1,v2,...]'` parses identically for `vector`
and `halfvec`, so `_vector_literal` is unchanged.

### 5. Re-embed

Existing 4096-dim rows survive the column ALTER (they get truncated by the
`USING` cast), but they were generated *without* renormalization, so cosine
scores will be subtly off. Cleanest: drop and re-embed.

```
psql -d tsgrep -c "DELETE FROM chunk_embeddings;"
make embed-python    # or whichever fixture / repo
```

## Quality / cost trade-off

- `vector(4096)` fp32, no index → 16 KB/row, O(N) lookup
- `halfvec(4000)` fp16 + HNSW → 8 KB/row, ~O(log N) lookup

Halving precision (fp32 → fp16) is invisible for cosine ranking on
embeddings; truncating 4096 → 4000 (Matryoshka) is ~2.3% of the embedding
discarded, also invisible. Net: same retrieval quality, half the storage,
asymptotically faster queries.

## Alternative if pgvector is too old for halfvec

Use `vector(2000)` with the same Matryoshka truncation. More aggressive
(half the embedding discarded, not 2.3%) but fp32 stays. Only pick this if
you can't get pgvector 0.7+.
