# tsgrep

Tree-sitter semantic code graph + vector embedding pipeline. Indexes a
repository into a relational graph (syntactic CST → semantic edges →
chunks + embeddings) so an LLM can retrieve coherent multi-file context
instead of token-window slop.

**Supported languages:** Python, Solidity, Go, JavaScript
(`.js`/`.jsx`/`.mjs`/`.cjs`), TypeScript (`.ts`/`.tsx`).

---

## Prerequisites

- macOS with Homebrew
- PostgreSQL 18 + `pgvector`
- `uv`

```sh
brew install postgresql@18 pgvector uv
```

---

## Setup

```sh
make install     # .venv + project deps via uv
make db-setup    # start postgres, create tsgrep DB, apply migrations
```

DSN defaults to `postgresql://$USER@localhost:5432/tsgrep`. Override with
`DATABASE_URL`.

---

## Shared vs per-repo DB

Repos are addressed by name (`--repo-name foo`). Two workflows:

- **Shared** — one `tsgrep` DB holding many repos. Use the plain
  `make index` / `make embed` / `make query-*` targets.
- **Isolated** — one DB per repo (`tsgrep_<name>`). Use the
  `*-isolated` variants; they create and migrate the per-repo DB on
  first use. Clean uninstall via `DROP DATABASE`.

---

## Indexing

```sh
make index REPO_PATH=/path/to/repo REPO_NAME=myrepo
```

Re-running after no source changes prints `Skipped N unchanged files`
— incremental indexing is keyed off SHA-256 of file contents.

---

## Embeddings

Set up secrets via pass-cli. Edit `.env.template` to point at your
secret store:

```
EMBEDDING_BASE_URL=https://ai-gateway.vercel.sh/v1
EMBEDDING_MODEL=alibaba/qwen3-embedding-8b
EMBEDDING_DIM=4096
EMBEDDING_API_KEY={{ pass://Personal/vercel-ai-gateway/secret }}
```

The `make embed` and `make query-*` targets stream the API key from
pass-cli into the process env — nothing is written to disk.

```sh
make embed REPO_PATH=/path/to/repo REPO_NAME=myrepo
```

If you'd rather export `EMBEDDING_*` vars manually, the raw CLI reads
the same vars.

---

## Query

```sh
# semantic: cosine NN over chunk embeddings
make query-semantic QUERY="how does the call graph link cross-module" REPO_NAME=myrepo

# hybrid: semantic seeds + 1-hop call-graph expansion (best default)
make query-hybrid QUERY="reentrancy guard usage" REPO_NAME=myrepo

# lexical: postgres FTS, no embedder needed
make query-lexical QUERY="deposit withdraw" REPO_NAME=myrepo

# structural: graph walk from a known definition name
.venv/bin/python -m cli.query --repo-name myrepo --structural deposit --depth 2
```

Add `--show-content` to print chunk bodies, `--top-k N` to change the
result count, `--context-budget N` to also emit a deduped, budget-fitted
block ready to paste into an LLM prompt.

---

## Feed the result to another LLM

`assemble_context(chunks, token_budget)` (`core/retrieval.py`) returns
a single string ready to drop into a system or user message:

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

The query-time embedding model **must** match the one used at index
time, otherwise cosine distances are meaningless.

---

`make help` lists every target.