# trident

Tree-sitter semantic code graph + vector embedding pipeline. Indexes a
repository into a relational graph (syntactic CST → semantic edges →
chunks + embeddings) so an LLM can retrieve coherent multi-file context
instead of token-window slop.

**Supported languages:** Python, Solidity, Go, JavaScript
(`.js`/`.jsx`/`.mjs`/`.cjs`), TypeScript (`.ts`/`.tsx`).

## Prerequisites

- macOS with Homebrew
- PostgreSQL 18 + `pgvector`
- `uv`

```sh
brew install postgresql@18 pgvector uv
```

## Setup

```sh
make install                       # .venv + project deps via uv
make db-setup DB=trident_myrepo    # start postgres, create the DB, apply migrations
```

There is no default DB — every runtime + admin make target requires
`DB=<name>`. Pick whatever name you want: a per-repo DB
(`trident_<repo>`) for clean uninstall via `DROP DATABASE`, or a shared
DB across many repos to enable cross-repo queries. The make targets
build `DATABASE_URL` from `DB=`; for raw `python -m cli.*` invocations,
export `DATABASE_URL` yourself.

## Indexing

```sh
make index DB=trident_repo REPO_PATH=/path/to/repo REPO_NAME=repo
```

Re-running after no source changes prints `Skipped N unchanged files`
— incremental indexing is keyed off SHA-256 of file contents.

## Embeddings

Copy `.env.template` to `.env` and edit it to point at your secret
store. `.env.template` is the committed scaffold; `.env` is your local,
gitignored copy that the make targets actually read.

```sh
cp .env.template .env
```

```
EMBEDDING_BASE_URL=https://ai-gateway.vercel.sh/v1
EMBEDDING_MODEL=alibaba/qwen3-embedding-8b
EMBEDDING_DIM=4096
EMBEDDING_API_KEY=<key>
```

The `make embed` and `make query-*` targets stream the API key from
pass-cli into the process env — nothing extra is written to disk.

```sh
make embed DB=trident_repo REPO_PATH=/path/to/repo REPO_NAME=repo
```

If you'd rather export `EMBEDDING_*` vars manually, the raw CLI reads
the same vars.

## Query

The query targets take a `REPO_LIST=a[:branch][,b[:branch]...]`. A single entry queries one repo (branch optional after `:`); comma-separated entries fan out to a cross-repo query — which only works when those repos share a DB.

```sh
# semantic: cosine NN over chunk embeddings
make query-semantic DB=trident_repo REPO_LIST=repo QUERY="how does the call graph link cross-module"

# hybrid: semantic seeds + 1-hop call-graph expansion (best default)
make query-hybrid DB=trident_repo REPO_LIST=repo QUERY="reentrancy guard usage"

# lexical: postgres FTS, no embedder needed
make query-lexical DB=trident_repo REPO_LIST=repo QUERY="deposit withdraw"

# cross-repo hybrid query (both repos must be indexed in the same DB)
make query-hybrid DB=trident_shared REPO_LIST=api,worker QUERY="user data flow"

# structural: graph walk from a known definition name
DATABASE_URL=postgresql://$USER@localhost:5432/trident_repo \
  .venv/bin/python -m cli.query --repo-name myrepo --structural deposit --depth 2
```

Add `--show-content` to print chunk bodies, `--top-k N` to change the
result count, `--context-budget N` to also emit a deduped, budget-fitted
block ready to paste into an LLM prompt.

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

`make help` lists every target.
