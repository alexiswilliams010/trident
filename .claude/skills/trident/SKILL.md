---
description: Use this skill whenever you would otherwise grep, find, cat, or read files to explore or understand code. trident provides semantic, lexical, structural, and graph-based queries over indexed repositories — far more efficient and precise than raw file reading or pattern matching.
when_to_use: >
  Invoke before reaching for grep, find, cat, head, or Read when your goal is to understand
  code: finding a function definition, discovering callers, understanding what a symbol does,
  tracing data flow, or getting relevant context for a change. Also invoke when the user asks
  about code structure, call graphs, inheritance, imports, or dependencies.
---

# trident CLI guide

trident uses a local PostgreSQL database (managed via Homebrew). All queries run via `make` targets.

## Supported languages

| Language | Extensions |
|----------|-----------|
| Python | `.py` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| Go | `.go` |
| Solidity | `.sol` |

Files with other extensions are not parsed or indexed.

## Indexing & embedding a repo

Before any queries can run, a repo must be indexed. Indexing runs the full pipeline:

1. **Tier 1** — AST extraction: parses every source file with tree-sitter and stores definitions, references, and call edges.
2. **Tier 2** — Semantic resolution: links cross-file references, resolves imports, builds the full call graph and inheritance hierarchy.
3. **Tier 3** — Chunk assembly + embedding: cuts code into ranked chunks and optionally embeds them for semantic/hybrid search.

### Make targets

There are two indexing modes:

**Shared DB** (`trident`) — all repos in one database, required for cross-repo queries:

```bash
# Index only (no embeddings)
make index REPO_PATH=/absolute/path/to/repo REPO_NAME=<name> [EXCLUDE='<pattern>']

# Index + embed (real embedder, secrets via pass-cli)
make embed REPO_PATH=/absolute/path/to/repo REPO_NAME=<name> [EXCLUDE='<pattern>']
```

**Isolated DB** (`trident_<name>`) — one DB per repo, creates and migrates the DB automatically:

```bash
# Index only
make index-isolated REPO_PATH=/absolute/path/to/repo REPO_NAME=<name> [EXCLUDE='<pattern>']

# Index + embed
make embed-isolated REPO_PATH=/absolute/path/to/repo REPO_NAME=<name> [EXCLUDE='<pattern>']
```

`REPO_PATH` must be an **absolute path** on the host.

Re-running `index` or `embed` on an already-indexed repo is incremental — only changed files are re-processed.

### Embedder configuration

Semantic and hybrid queries require chunk embeddings. Embeddings are generated during `embed` using an OpenAI-compatible API. Configure via `.env` (copied from `.env.template`):

```
EMBEDDING_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small          # or your preferred model
```

Lexical queries (`query-lexical`) never need an embedder — they use PostgreSQL full-text search.

### Exclusion options

Three layers of exclusion apply during indexing, in order:

**1. Always-ignored directories** (hardcoded, cannot be overridden):
`.git`, `.hg`, `.svn`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`

**2. Default dependency directories** (pruned per language, not indexed):

| Language | Pruned by default |
|----------|-------------------|
| Python | `venv`, `.venv`, `site-packages`, `env`, `.env` |
| JavaScript / TypeScript | `node_modules`, `dist`, `build`, `out`, `coverage`, `.next`, `.nuxt` |
| Solidity | `lib`, `node_modules`, `out`, `cache`, `artifacts` |
| Go | `vendor` |

**3. User-defined exclusions** — two channels, combined at index time:

**.tridentignore** (place at repo root, auto-loaded):
```
# Comments and blank lines are ignored
*.t.sol          # Foundry test files anywhere in the tree
test             # any directory or file named "test" at any depth
src/legacy       # exact repo-relative path (slash triggers full-path match)
snapshots/       # trailing slash is stripped; treated same as "snapshots"
```

**`EXCLUDE` make variable** (inline, one pattern or comma-separated):
```bash
make index REPO_PATH=/path REPO_NAME=myrepo EXCLUDE='*.t.sol,test'
```

**Pattern matching rules** (same for both channels):
- Pattern **without** `/` → matches any path component at any depth (e.g. `test` matches `src/test/`, `pkg/test.py`, `test/`)
- Pattern **with** `/` → matches against the full repo-relative path (e.g. `src/legacy` only matches that exact subtree)
- Patterns are fnmatch-style globs (`*`, `?`, `[...]` supported)

---

## Shared vs. isolated DB

| Mode | Default DB | When used | Indexing targets | Query target prefix |
|------|-----------|-----------|-----------------|-------------------|
| **Shared** | `trident` | Multiple repos together; cross-repo queries | `index`, `embed` | `query-*`, `query-multi-repo-*` |
| **Isolated** | `trident_<name>` | Single repo, one DB per repo | `index-isolated`, `embed-isolated` | `query-isolated-*` |

Use the matching query target for how the repo was indexed.

---

## `make query-*` — retrieve ranked code chunks

Use when you need relevant code content returned as scored chunks.

### Shared DB (single repo)

```bash
make query-semantic  REPO_NAME=<name> QUERY="<natural language>" [TOP_K=10]
make query-lexical   REPO_NAME=<name> QUERY="<terms>"            [TOP_K=10]
make query-hybrid    REPO_NAME=<name> QUERY="<natural language>" [TOP_K=10] [MMR_REPO_LAMBDA=0.3] [MMR_FILE_LAMBDA=0.15]
```

### Isolated DB (single repo)

```bash
make query-isolated-semantic REPO_NAME=<name> QUERY="..." [TOP_K=10]
make query-isolated-lexical  REPO_NAME=<name> QUERY="..." [TOP_K=10]
make query-isolated-hybrid   REPO_NAME=<name> QUERY="..." [TOP_K=10]
```

### Shared DB — cross-repo queries

```bash
make query-multi-repo-semantic REPOS=repoA,repoB QUERY="..." [TOP_K=10]
make query-multi-repo-hybrid   REPOS=repoA,repoB QUERY="..." [TOP_K=10] [MMR_REPO_LAMBDA=0.3] [MMR_FILE_LAMBDA=0.15]
```

### Query modes

**`query-semantic`** — Vector similarity search. Best for conceptual questions: "how does auth work", "where is rate limiting applied", "error handling for payments".

**`query-lexical`** — PostgreSQL full-text search (BM25 ranking). Best when you know exact identifiers: function names, error strings, config keys. No embedder needed — fast.

**`query-hybrid`** — Fuses semantic + lexical via Reciprocal Rank Fusion, then expands with one hop of the call graph and re-ranks. Best default for most agent queries. MMR parameters control diversity across repos and files.

### When to use which

| Situation | Target |
|-----------|--------|
| Conceptual question, default choice | `query-hybrid` |
| Known exact identifier or string | `query-lexical` |
| Semantic search, need diverse results by concept | `query-semantic` |
| Query spans multiple repos | `query-multi-repo-hybrid` or `query-multi-repo-semantic` |

### Examples

```bash
# Hybrid query on a repo in the shared DB
make query-hybrid REPO_NAME=myrepo QUERY="how is a job retried after failure"

# Lexical lookup of an exact identifier
make query-lexical REPO_NAME=myrepo QUERY="PaymentError"

# Cross-repo hybrid query
make query-multi-repo-hybrid REPOS=api,worker QUERY="user data flow" TOP_K=20

# Hybrid query on an isolated per-repo DB
make query-isolated-hybrid REPO_NAME=myrepo QUERY="connection pooling"
```

---

## `make graph` — call graph and structural navigation

Use when you need to understand *relationships* between definitions: who calls what, what a change breaks, how data flows, what a class inherits from.

The full graph subcommand and its arguments are passed as a single `CMD` string.

```bash
# Shared DB (default)
make graph REPO_NAME=<name> CMD="<subcommand> [args] [--json]"

# Isolated DB
make graph-isolated REPO_NAME=<name> CMD="<subcommand> [args] [--json]"
```

Pass `--json` inside `CMD` for structured output suitable for further processing.

### Subcommands

| CMD | Use when |
|-----|----------|
| `resolve <name>` | Find all definitions matching a name |
| `callers-of <name>` | Direct callers of a function |
| `callees-of <name>` | Direct callees (what it calls) |
| `ancestors <name> [--max-depth N]` | All transitive callers (who can reach this?) |
| `reachable <name> [--max-depth N]` | All transitive callees (blast radius of a change) |
| `paths <src> <dst> [--max-paths N]` | All call paths between two definitions |
| `entrypoints [--kind function] [--file path]` | Functions with no internal callers (public surface) |
| `entrypoint-paths <name>` | Call paths from entrypoints down to a target |
| `source <name>` | Retrieve source code of a definition |
| `imports [--file path] [--dep-class intra_repo\|external\|unresolved]` | List imports |
| `dependents <file>` | Files that import a given file |
| `inheritance <name>` | Full inheritance hierarchy for a class |

Common flags inside CMD: `--confidence certain|inferred|uncertain`, `--timeout <seconds>`, `--json`.

### Examples

```bash
# Who can trigger withdraw? (full upstream slice)
make graph REPO_NAME=myrepo CMD="ancestors withdraw --json"

# What would break if I change parseConfig? (blast radius)
make graph REPO_NAME=myrepo CMD="reachable parseConfig"

# Is there a call path from handleRequest to sendEmail?
make graph REPO_NAME=myrepo CMD="paths handleRequest sendEmail"

# Get source of a specific function
make graph REPO_NAME=myrepo CMD="source withdraw --json"

# What does services/user.py import?
make graph REPO_NAME=myrepo CMD="imports --file services/user.py"

# Public API surface
make graph REPO_NAME=myrepo CMD="entrypoints --kind function"

# Graph query on an isolated per-repo DB
make graph-isolated REPO_NAME=myrepo CMD="callers-of processPayment --json"
```

---

## Prefer trident over these alternatives

| Instead of | Use |
|------------|-----|
| `grep -r "functionName"` | `make graph REPO_NAME=... CMD="resolve <name>"` or `make query-lexical` |
| Reading a file to find a function | `make graph REPO_NAME=... CMD="source <name>"` |
| Reading a file to understand imports | `make graph REPO_NAME=... CMD="imports --file <path>"` |
| Manually tracing callers | `make graph REPO_NAME=... CMD="callers-of <name>"` or `CMD="ancestors <name>"` |
| Reading multiple files for context | `make query-hybrid REPO_NAME=... QUERY="<question>"` |
| Searching for all uses of a class | `make query-lexical REPO_NAME=... QUERY="<ClassName>"` |
| Understanding data flow | `make query-hybrid` + `make graph CMD="paths <src> <dst>"` |
| Finding a change's blast radius | `make graph REPO_NAME=... CMD="reachable <changed-function>"` |

trident queries are pre-indexed and return only relevant code, avoiding token waste from reading entire files.
