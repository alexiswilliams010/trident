# tsgrep

Tree-sitter semantic code graph + vector embedding pipeline. Indexes a
repository into a relational graph (Tier 1 syntactic CST → Tier 2 semantic
edges → Tier 3 chunks + embeddings) so an LLM can retrieve coherent
multi-file context instead of token-window slop.

---

## Prerequisites

- macOS with Homebrew
- Python 3.12 (managed automatically by `uv`)
- PostgreSQL 18 (Homebrew)
- `pgvector` extension
- `uv` (Python package manager)

Install the system bits:

```sh
brew install postgresql@18 pgvector uv
```

---

## Setup

From the repo root:

```sh
make install     # creates .venv (Python 3.12) and installs project + dev deps via uv
make db-setup    # brew services start postgresql@18 + createdb tsgrep + apply migrations
```

`make db-setup` is idempotent — re-running it is safe.

DSN defaults to `postgresql://$USER@localhost:5432/tsgrep`. Override with
the `DATABASE_URL` env var if needed.

---

## Running tests

```sh
make test            # full pytest suite
make test-extractor  # Phase 1 extractor tests only
make test-resolver   # Phase 2 semantic resolver tests only
make test-imports    # Phase 3 heuristic resolver tests only
make test-chunks     # Phase 4 chunk assembly + retrieval tests
```

Tests that need Postgres connect via the same DSN; they auto-skip with a
clear message if the service is not reachable.

---

## Two workflows: shared DB vs per-repo DB

Repos are addressed by **name** (`--repo-name foo`). The schema has a
`repos` table mapping each name to an internal id, and `files.repo_id`
foreign-keys back to it (so `DELETE FROM repos WHERE name='foo'` cascades
through everything). Pick whichever workflow fits:

| Mode | DB | When to use | Pros |
|---|---|---|---|
| **A — shared** | one `tsgrep` DB, many repos | complex products spanning multiple repos; cross-repo analysis | one connection pool; `SELECT … FROM repos` lists everything; cross-repo queries possible |
| **B — isolated** | one DB per repo (`tsgrep_<name>`) | CI workflows, per-project querying | clean uninstall via `DROP DATABASE`; smaller tables; one DSN = one repo |

Switch modes per-call by setting `DB=tsgrep_other` on any target, or use
the `*-isolated` variants which derive `DB` from `REPO_NAME` automatically
and create + migrate the per-repo DB on first use.

---

## Indexing the bundled fixtures

```sh
make index-python                # repo name `python_fixture` in shared DB
make index-solidity              # repo name `solidity_fixture` in shared DB

make diagnose-python             # resolution stats for python_fixture
make diagnose-solidity           # resolution stats for solidity_fixture

make embed-python-fake           # embed Python fixture chunks with the deterministic stub
make embed-solidity-fake         # ditto for Solidity
```

Retrieval queries (use `--fake` for the stub embedder; otherwise set
`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` or use a
`make query-*` target so pass-cli injects them):

```sh
.venv/bin/python -m cli.query --repo-name solidity_fixture --structural deposit --depth 2
.venv/bin/python -m cli.query --repo-name solidity_fixture --semantic "token transfer balance update" --fake
.venv/bin/python -m cli.query --repo-name solidity_fixture --hybrid "reentrancy guard usage" --fake
```

Inspect the result:

```sh
make db-psql
```

```sql
-- repos in this database
SELECT id, name, root_path, created_at FROM repos ORDER BY id;

-- files indexed per repo
SELECT r.name, f.language, COUNT(*) FROM files f
JOIN repos r ON r.id = f.repo_id
GROUP BY r.name, f.language;

-- definitions for one repo
SELECT d.kind, d.qualified_name FROM definitions d
JOIN files f ON f.id = d.file_id
JOIN repos r ON r.id = f.repo_id
WHERE r.name = 'python_fixture' ORDER BY d.id;

-- call graph: caller → callee with confidence
SELECT caller.qualified_name, callee.qualified_name, ce.confidence
FROM call_edges ce
JOIN definitions caller ON caller.id = ce.caller_def_id
LEFT JOIN definitions callee ON callee.id = ce.callee_def_id
ORDER BY caller.qualified_name;
```

Re-running `make index-python` after no source changes prints
`Skipped N unchanged files` — incremental indexing is keyed off SHA-256 of
each file's contents.

---

## Indexing your own repo

**Mode A — into the shared `tsgrep` DB:**

```sh
make index REPO_PATH=/path/to/your/repo REPO_NAME=myrepo
```

**Mode B — into its own DB `tsgrep_myrepo` (auto-created and migrated):**

```sh
make index-isolated REPO_PATH=/path/to/your/repo REPO_NAME=myrepo
```

Or invoke the CLI directly:

```sh
.venv/bin/python -m cli.index /path/to/your/repo --repo-name myrepo
```

Add `--init-schema` on first use (or just rely on `make db-setup` for the
shared DB / `make index-isolated` which migrates per-repo DBs
automatically). The walker honors per-language dependency directories so
dependency caches (`node_modules/`, `lib/`, `.venv/`, etc.) stay unparsed;
they will be visited only by the Phase 3 targeted resolver pass for files
that are actually imported.

---

## Embeddings (Phase 4b)

Chunks are embedded via an OpenAI-compatible HTTP gateway, configured by
**three required environment variables**. No `.env` file is auto-loaded;
keys live in your shell only.

| Var | Required | Default | Purpose |
|---|---|---|---|
| `EMBEDDING_BASE_URL` | yes | — | Gateway URL, e.g. `https://api.openai.com/v1` |
| `EMBEDDING_API_KEY`  | yes | — | API key for that gateway |
| `EMBEDDING_MODEL`    | yes | — | Model name, e.g. `text-embedding-3-large` |
| `EMBEDDING_DIM`      | no  | `4096` | Must match `chunk_embeddings.embedding vector(4096)` in schema |
| `EMBEDDING_BATCH_SIZE` | no | `64` | Batch size for embedding API calls |

If you don't want to set up a real provider yet, append `--fake` to any
query (or `--embed fake` to `cli.index`) to use the deterministic SHA-256
stub embedder. It exercises the full pipeline locally with no API key.

**Provider configurations:**

| Provider | Vars |
|---|---|
| OpenAI | `EMBEDDING_BASE_URL=https://api.openai.com/v1`<br>`EMBEDDING_API_KEY=sk-...`<br>`EMBEDDING_MODEL=text-embedding-3-large` |
| Voyage AI | `EMBEDDING_BASE_URL=https://api.voyageai.com/v1`<br>`EMBEDDING_API_KEY=pa-...`<br>`EMBEDDING_MODEL=voyage-code-3` |
| LiteLLM proxy | `EMBEDDING_BASE_URL=http://localhost:4000`<br>`EMBEDDING_API_KEY=<anything>`<br>`EMBEDDING_MODEL=<as-registered>` |
| Self-hosted (vLLM, etc.) | `EMBEDDING_BASE_URL=http://localhost:8000/v1`<br>`EMBEDDING_API_KEY=<anything>`<br>`EMBEDDING_MODEL=nomic-embed-code` |

**Setting them:**

```sh
# session-wide (recommended for repeated runs)
export EMBEDDING_BASE_URL=https://api.openai.com/v1
export EMBEDDING_API_KEY=sk-...
export EMBEDDING_MODEL=text-embedding-3-large

# or one-shot, inline:
EMBEDDING_BASE_URL=https://api.openai.com/v1 \
EMBEDDING_API_KEY=sk-... \
EMBEDDING_MODEL=text-embedding-3-large \
.venv/bin/python -m cli.index tests/fixtures/python_fixture --repo-id 1 --embed real

# or stash them in an out-of-tree file and source it:
set -a; source ~/.config/tsgrep/env; set +a
make index-python   # then pass --embed real to the underlying CLI as needed
```

**Schema constraint** — `chunk_embeddings.embedding` is `vector(4096)`,
sized for qwen3-embedding-8b's native output. `EMBEDDING_DIM` must match
both the schema and the vector your model returns; mismatches are caught
at insert time by the validator in `embed_repo_chunks`. For models with a
different native width:

- OpenAI `text-embedding-3-large` — accepts a `dimensions` parameter; the
  embedder forwards it automatically when `EMBEDDING_BASE_URL` contains
  `openai.com` or `EMBEDDING_MODEL` starts with `text-embedding-3`.
- Voyage `voyage-code-3` — natively 1024; needs a schema change to
  `vector(1024)` (see `db/migrations/0001_init.up.sql`).
- Self-hosted / other gateways — either project via a LiteLLM proxy or
  truncate client-side, then bump `EMBEDDING_DIM` and the schema column to
  match.

**No ANN index** — pgvector caps HNSW at 2000 dims for `vector` and 4000
for `halfvec`, so cosine NN runs as a sequential scan. Fine at small/medium
scale; see [docs/halfvec-migration.md](docs/halfvec-migration.md) for the
upgrade path when chunk counts grow.

**`from_env()` raises a clear error** if any required var is missing — so
running `--embed real` without setup fails fast with the missing-var name.

---

## End-to-end: index, embed, query

The full pipeline takes four steps. Steps 1-2 are one-time per repo; step 3
runs whenever the source changes; step 4 is the read path you'll hit
repeatedly.

### 1. Set up secrets via pass-cli (recommended)

Edit `.env.template` at the repo root to point at your secret store:

```
EMBEDDING_BASE_URL=https://ai-gateway.vercel.sh/v1
EMBEDDING_MODEL=alibaba/qwen3-embedding-8b
EMBEDDING_DIM=4096
EMBEDDING_API_KEY={{ pass://Personal/vercel-ai-gateway/secret }}
```

The `make embed-*` and `make query-*` targets stream the API key from
pass-cli into the process env — nothing is written to disk. Verify with:

```sh
pass-cli inject --in-file .env.template
```

Plain values (URL, model name, dim) pass through; only the API key is
fetched. If you'd rather export env vars manually, see the section above —
the make targets and the raw CLI both read the same vars.

### 2. Index the repo (Tiers 1-3a)

Parses sources, builds the call graph, assembles chunks. No API calls.

**Mode A — shared `tsgrep` DB (multi-repo):**

```sh
make index-python                                          # bundled fixture
make index-solidity                                        # bundled fixture
make index REPO_PATH=/path/to/repo REPO_NAME=myrepo        # any other repo
```

**Mode B — per-repo DB `tsgrep_<name>` (CI / isolation):**

```sh
make index-isolated REPO_PATH=/path/to/repo REPO_NAME=myrepo
```

`REPO_NAME` is the human-readable handle; the schema's `repos` table maps
it to the internal id used by foreign keys. List what's in a DB at any
time with `psql -d tsgrep -c "SELECT name FROM repos"`.

### 3. Embed (Tier 3b)

Walks `chunks` rows that don't yet have an embedding, batches them through
the gateway, writes vectors to `chunk_embeddings`. Re-runs are cheap:
unchanged chunks are skipped via `LEFT JOIN ... WHERE ce.id IS NULL`.

```sh
make embed-python                                          # bundled fixture, shared DB
make embed-solidity
make embed REPO_PATH=/path/to/repo REPO_NAME=myrepo        # any repo, shared DB
make embed-isolated REPO_PATH=/path/to/repo REPO_NAME=myrepo  # any repo, per-repo DB

make embed-python-fake                                     # stub embedder, no API key
```

The `embed` / `embed-isolated` targets re-run Tier 1-3a (idempotent /
cheap if unchanged) and then embed — same as the fixture-specific targets.

### 4. Query

Three modes, two of which need the same embedding model that produced the
index:

```sh
# Semantic: pure cosine NN over chunk_embeddings.
make query-semantic QUERY="how does the call graph link cross-module" REPO_NAME=myrepo

# Hybrid: semantic seeds + 1-hop call-graph expansion (best general default).
make query-hybrid QUERY="reentrancy guard usage" REPO_NAME=solidity_fixture

# Per-repo DB variants:
make query-isolated-hybrid QUERY="..." REPO_NAME=myrepo

# Structural: graph walk from a known definition name, no embedding needed.
.venv/bin/python -m cli.query --repo-name solidity_fixture --structural deposit --depth 2
```

Add `--show-content` to print chunk bodies, `--top-k N` to change result
count, `--context-budget N` to also emit a deduped, budget-fitted block
suitable for pasting into an LLM prompt.

### 5. Feed the result to another LLM

`assemble_context(chunks, token_budget)` (`core/retrieval.py:231`) returns
a single string ready to drop into a system or user message. From Python:

```python
from cli._repo import resolve_repo_id
from db.connection import pool_ctx
from core.embedder import EmbedderConfig, OpenAICompatibleEmbedder
from core.retrieval import hybrid_query, assemble_context

async def get_context(repo_name: str, question: str, budget: int = 8000) -> str:
    embed_fn = OpenAICompatibleEmbedder(EmbedderConfig.from_env()).embed
    async with pool_ctx() as pool:
        repo_id = await resolve_repo_id(pool, name=repo_name, create=False)
        chunks = await hybrid_query(pool, repo_id, question, embed_fn, top_k=10)
    return assemble_context(chunks, token_budget=budget)
```

Pass the returned string as context to any model — Claude, GPT, a local
llama, whatever. The query-time embedding model **must** match the one
used in step 3, otherwise cosine distances are meaningless.

---

## Make targets

```
make help            # list every target
make install         # uv venv + install
make lint            # ruff check
make test            # pytest

make db-start        # brew services start postgresql@18
make db-stop         # brew services stop postgresql@18
make db-create       # createdb tsgrep
make db-drop         # dropdb tsgrep
make db-migrate      # apply unapplied migrations from db/migrations/*.up.sql
make db-setup        # db-start + db-create + db-migrate
make db-reset        # db-drop + db-create + db-migrate
make db-psql         # interactive psql shell on tsgrep

# Mode A — into the shared `tsgrep` DB
make index-python                                       # bundled Python fixture
make index-solidity                                     # bundled Solidity fixture
make index REPO_PATH=/path REPO_NAME=name               # any repo
make embed-python                                       # real embedder via pass-cli
make embed-solidity
make embed-python-fake                                  # deterministic stub embedder
make embed-solidity-fake
make embed REPO_PATH=/path REPO_NAME=name               # index + embed any repo
make query-semantic QUERY="..." REPO_NAME=name [DB=...] # cosine NN
make query-hybrid   QUERY="..." REPO_NAME=name [DB=...] # semantic + 1-hop graph expand

# Mode B — into a per-repo DB tsgrep_<name> (auto-created and migrated)
make index-isolated REPO_PATH=/path REPO_NAME=name
make embed-isolated REPO_PATH=/path REPO_NAME=name
make query-isolated-semantic QUERY="..." REPO_NAME=name
make query-isolated-hybrid   QUERY="..." REPO_NAME=name

# Diagnostics
make diagnose-python                                    # stats for python_fixture
make diagnose-solidity                                  # stats for solidity_fixture
```

Override `PG_SERVICE` or `PG_DB` as `make` variables if your local setup
differs (e.g. `make db-setup PG_DB=tsgrep_dev`).
