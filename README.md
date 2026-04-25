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
```

Tests that need Postgres connect via the same DSN; they auto-skip with a
clear message if the service is not reachable.

---

## Indexing the bundled fixtures

```sh
make index-python    # indexes tests/fixtures/python_fixture under repo_id 1
make index-solidity  # indexes tests/fixtures/solidity_foundry_fixture under repo_id 2

make diagnose-python    # resolution stats for repo_id 1 (incl. unresolved imports)
make diagnose-solidity  # resolution stats for repo_id 2
```

Inspect the result:

```sh
make db-psql
```

```sql
-- files indexed per repo
SELECT repo_id, language, COUNT(*) FROM files GROUP BY repo_id, language;

-- total nodes per repo
SELECT f.repo_id, COUNT(n.id) FROM nodes n
JOIN files f ON f.id = n.file_id GROUP BY f.repo_id;

-- definitions with qualified names (Phase 2)
SELECT d.kind, d.qualified_name FROM definitions d
JOIN files f ON f.id = d.file_id
WHERE f.repo_id = 1 ORDER BY d.id;

-- call graph (Phase 2): caller → callee with confidence
SELECT caller.qualified_name, callee.qualified_name, ce.confidence
FROM call_edges ce
JOIN definitions caller ON caller.id = ce.caller_def_id
LEFT JOIN definitions callee ON callee.id = ce.callee_def_id
ORDER BY caller.qualified_name;

-- data access (Phase 2): function reads/writes of state vars
SELECT accessor.qualified_name AS by, target.qualified_name AS field, da.access_type
FROM data_access da
JOIN definitions accessor ON accessor.id = da.accessor_def_id
JOIN definitions target   ON target.id   = da.target_def_id
ORDER BY by, field;
```

Re-running `make index-python` after no source changes prints
`Skipped N unchanged files` — incremental indexing is keyed off SHA-256 of
each file's contents.

---

## Indexing your own repo

```sh
.venv/bin/python -m cli.index /path/to/your/repo --repo-id 42
```

Add `--init-schema` on first use (or just rely on `make db-setup`). The
walker honors per-language dependency directories so dependency caches
(`node_modules/`, `lib/`, `.venv/`, etc.) stay unparsed; they will be
visited only by the Phase 3 targeted resolver pass for files that are
actually imported.

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

make index-python    # index the Python fixture
make index-solidity  # index the Solidity fixture
```

Override `PG_SERVICE` or `PG_DB` as `make` variables if your local setup
differs (e.g. `make db-setup PG_DB=tsgrep_dev`).

---

## Project layout

```
tsgrep/
├── pyproject.toml           # uv-managed project, Python 3.12+
├── .python-version          # 3.12
├── Makefile                 # uv + Postgres + index helpers
├── Architecture.md          # full design doc (8 phases)
│
├── configs/
│   ├── _schema.json         # JSON Schema validating language YAMLs
│   ├── python.yaml          # Phase 2 rules for Python
│   └── solidity.yaml        # Phase 2 rules for Solidity
│
├── core/
│   ├── extractor.py          # Phase 1: Tree-sitter -> nodes table
│   ├── file_walker.py        # Phase 1: dep-aware repo walker
│   ├── grammar_meta.py       # Language registry (Python, Solidity)
│   ├── config_loader.py      # Phase 2: YAML loader + validation
│   ├── semantic_resolver.py  # Phase 2: defs / refs / calls / data_access
│   └── heuristic_resolver.py # Phase 3: imports + cross-file linking
│
├── db/
│   ├── connection.py        # asyncpg pool + migration runner
│   └── migrations/
│       └── 0001_init.up.sql # Tier 1 + 2 + 3 schema, pgvector, HNSW
│
├── cli/
│   ├── index.py             # python -m cli.index <repo> --repo-id N
│   └── diagnose.py          # python -m cli.diagnose --repo-id N
│
└── tests/
    ├── conftest.py
    ├── test_extractor.py
    ├── test_semantic_resolver.py
    └── fixtures/
        ├── python_fixture/             # multi-file package + .venv decoy
        └── solidity_foundry_fixture/   # foundry layout + lib/ decoy
```

---

## Implementation phase status

| Phase | Description | Status |
|---|---|---|
| 1 | Tier 1 extractor + DB schema | done |
| 2 | YAML-driven semantic resolver (definitions, references, scopes, calls, data access) | done |
| 3 | Heuristic cross-file import resolution + cross-file edge linking | done |
| 4 | Graph-informed chunk assembly + embeddings + retrieval | next |

Phase 5+ (eval harness, Deno resolver sandbox, additional languages) are
deferred until the MVP shows the approach works.
