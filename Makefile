.PHONY: help install lint test test-extractor test-resolver test-imports test-chunks \
        db-start db-stop db-create db-drop db-migrate db-setup db-teardown db-reset db-psql \
        db-test-setup \
        diagnose graph \
        query-semantic query-lexical query-hybrid query-fake-hybrid \
        index embed \
        branches branch-set-default branch-drop gc \
        _require-db

UV ?= uv
PYTHON := .venv/bin/python

PG_SERVICE ?= postgresql@18
MIGRATIONS_DIR := db/migrations

PASS_CLI ?= pass-cli
ENV_FILE ?= .env
ENV_TEMPLATE ?= .env.template

# DB to operate on. Required for every runtime target and every db-* admin
# target. No default — pick `trident_<repo>` for per-repo isolation, or share
# one DB across many repos for cross-repo queries.
DB ?=
DB_USER ?= $(USER)
DB_DSN := postgresql://$(DB_USER)@localhost:5432/$(DB)

# Test DB. The `test` target points DATABASE_URL here unconditionally, so
# tests never touch a real DB regardless of DB=.
TEST_DB ?= trident_test
TEST_DB_DSN := postgresql://$(DB_USER)@localhost:5432/$(TEST_DB)

# Run $(1) with:
#   - secrets streamed from pass-cli into the env (no file on disk),
#   - DATABASE_URL pointing at $(DB).
# Reads $(ENV_FILE) (default `.env`); copy $(ENV_TEMPLATE) to $(ENV_FILE)
# on first use and edit values to taste.
define inject_and_run
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(ENV_FILE) not found — copy $(ENV_TEMPLATE) to $(ENV_FILE) and edit it"; \
		exit 1; \
	fi; \
	OUTPUT=$$($(PASS_CLI) inject --in-file $(ENV_FILE)) || { echo "pass-cli inject failed"; exit 1; }; \
	set -a; \
	eval "$$OUTPUT"; \
	set +a; \
	unset OUTPUT; \
	exec env DATABASE_URL=$(DB_DSN) $(1)
endef

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

_require-db:
	@if [ -z "$(DB)" ]; then \
		echo 'DB is required — e.g. DB=trident_myrepo. Pick any name; share across repos for cross-repo queries.'; \
		exit 2; \
	fi

# ------------------------------------------------------------------------------
# Python / uv
# ------------------------------------------------------------------------------
install: ## Create venv (Python 3.12) and install project + dev deps with uv.
	@$(UV) venv --python 3.12
	@$(UV) pip install -e ".[dev]"

lint: ## Run ruff over the source tree.
	@$(PYTHON) -m ruff check core cli db tests

test: ## Run all pytest tests against $(TEST_DB).
	@env DATABASE_URL=$(TEST_DB_DSN) $(PYTHON) -m pytest -v

test-extractor: ## Run only Phase 1 extractor tests.
	@env DATABASE_URL=$(TEST_DB_DSN) $(PYTHON) -m pytest -v tests/test_extractor.py

test-resolver: ## Run only Phase 2 semantic resolver tests.
	@env DATABASE_URL=$(TEST_DB_DSN) $(PYTHON) -m pytest -v tests/test_semantic_resolver.py

test-imports: ## Run only Phase 3 heuristic resolver tests.
	@env DATABASE_URL=$(TEST_DB_DSN) $(PYTHON) -m pytest -v tests/test_heuristic_resolver.py

test-chunks: ## Run only Phase 4 chunk + retrieval tests.
	@env DATABASE_URL=$(TEST_DB_DSN) $(PYTHON) -m pytest -v tests/test_chunk_assembler.py tests/test_retrieval.py

# ------------------------------------------------------------------------------
# Local PostgreSQL (Homebrew). Override PG_SERVICE as needed.
# ------------------------------------------------------------------------------
db-start: ## Start the local PostgreSQL service.
	@brew services start $(PG_SERVICE)

db-stop: ## Stop the local PostgreSQL service.
	@brew services stop $(PG_SERVICE)

db-create: _require-db ## Create $(DB) (idempotent).
	@createdb $(DB) 2>/dev/null || echo "database $(DB) already exists"

db-drop: _require-db ## Drop $(DB) if it exists.
	@dropdb --if-exists $(DB)

db-migrate: _require-db ## Apply all unapplied forward migrations against $(DB).
	@psql -d $(DB) -c \
		"CREATE TABLE IF NOT EXISTS schema_migrations (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());" \
		> /dev/null
	@for f in $(MIGRATIONS_DIR)/*.up.sql; do \
		fname=$$(basename $$f); \
		applied=$$(psql -d $(DB) -tAc "SELECT 1 FROM schema_migrations WHERE filename = '$$fname'"); \
		if [ "$$applied" = "1" ]; then \
			echo "skipping $$fname (already applied)"; \
		else \
			echo "applying $$fname"; \
			psql -d $(DB) -v ON_ERROR_STOP=1 -f $$f || exit 1; \
			psql -d $(DB) -c "INSERT INTO schema_migrations (filename) VALUES ('$$fname');" > /dev/null; \
		fi; \
	done

db-setup: db-start db-create db-migrate ## Start the service, create $(DB), and apply migrations.

db-teardown: db-drop ## Drop $(DB) (service keeps running).

db-reset: db-drop db-create db-migrate ## Drop and recreate $(DB) from scratch.

db-psql: _require-db ## Open a psql shell on $(DB).
	@psql -d $(DB)

db-test-setup: ## Create + migrate $(TEST_DB) so `make test` can run.
	@brew services start $(PG_SERVICE) > /dev/null
	@createdb $(TEST_DB) 2>/dev/null || echo "database $(TEST_DB) already exists"
	@$(MAKE) --no-print-directory db-migrate DB=$(TEST_DB)

# ------------------------------------------------------------------------------
# trident CLI helpers
# ------------------------------------------------------------------------------
REPO_PATH ?=
REPO_NAME ?=
REPO_LIST ?=
QUERY     ?=
EXCLUDE   ?=
BRANCH    ?=

# Knobs exposed by the retrieval features. All optional — only threaded into
# the CLI invocation when set, so existing usage is unchanged.
TOP_K            ?= 10
MMR_REPO_LAMBDA  ?=    # hybrid only; default 0.3 inside the CLI
MMR_FILE_LAMBDA  ?=    # hybrid only; default 0.15 inside the CLI

# When EXCLUDE is set, expand to a single --exclude flag carrying the
# comma-separated value (argparse splits on comma).
EXCLUDE_FLAG := $(if $(EXCLUDE),--exclude '$(EXCLUDE)',)

# When BRANCH is set, pass it through to the CLI. Omitted = the repo's
# designated default branch (auto-created as `main` on first index).
BRANCH_FLAG := $(if $(BRANCH),--branch $(BRANCH),)

# Force semantic re-resolution even when Tier-1 says nothing changed. Use
# this after editing core/semantic_resolver.py / configs/*.yaml etc., where
# the source files are unchanged but the analysis logic is.
FORCE_RESOLVE_FLAG := $(if $(FORCE_RESOLVE),--force-resolve,)

# Optional flag expansions — empty when the variable is unset, so the CLI
# falls back to its built-in defaults.
MMR_REPO_LAMBDA_FLAG  := $(if $(MMR_REPO_LAMBDA),--mmr-repo-lambda $(MMR_REPO_LAMBDA),)
MMR_FILE_LAMBDA_FLAG  := $(if $(MMR_FILE_LAMBDA),--mmr-file-lambda $(MMR_FILE_LAMBDA),)

index: _require-db ## Index a repo into $(DB). REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE='pat1,pat2']
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make index DB=name REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE=...]'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) $(BRANCH_FLAG) $(EXCLUDE_FLAG) $(FORCE_RESOLVE_FLAG)

embed: _require-db ## Index + embed a repo into $(DB) (real embedder, secrets via pass-cli). REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE='pat1,pat2'] [FORCE_RESOLVE=1]
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make embed DB=name REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE=...] [FORCE_RESOLVE=1]'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) --embed real $(BRANCH_FLAG) $(EXCLUDE_FLAG) $(FORCE_RESOLVE_FLAG))

# Query targets accept REPO_LIST=a[:branch][,b[:branch]...]. A single entry
# without a colon queries the repo's default branch; with a colon, the named
# branch. Comma-separated entries fan out to a cross-repo query.
query-semantic: _require-db ## Semantic query. QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=10]
	@if [ -z "$(QUERY)" ] || [ -z "$(REPO_LIST)" ]; then \
		echo 'usage: make query-semantic DB=name QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=...]'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.query --repos $(REPO_LIST) --semantic "$(QUERY)" --top-k $(TOP_K))

query-lexical: _require-db ## Lexical (FTS) query. QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=10]. No embedder needed.
	@if [ -z "$(QUERY)" ] || [ -z "$(REPO_LIST)" ]; then \
		echo 'usage: make query-lexical DB=name QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=...]'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.query --repos $(REPO_LIST) --lexical "$(QUERY)" --top-k $(TOP_K)

query-hybrid: _require-db ## Hybrid query. QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=10] [MMR_REPO_LAMBDA=0.3] [MMR_FILE_LAMBDA=0.15]
	@if [ -z "$(QUERY)" ] || [ -z "$(REPO_LIST)" ]; then \
		echo 'usage: make query-hybrid DB=name QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=...] [MMR_*=...]'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.query --repos $(REPO_LIST) --hybrid "$(QUERY)" \
		--top-k $(TOP_K) $(MMR_REPO_LAMBDA_FLAG) $(MMR_FILE_LAMBDA_FLAG))

query-fake-hybrid: _require-db ## Hybrid query with the deterministic fake embedder (no API key). QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=...]
	@if [ -z "$(QUERY)" ] || [ -z "$(REPO_LIST)" ]; then \
		echo 'usage: make query-fake-hybrid DB=name QUERY="..." REPO_LIST=a[:branch][,b[:branch]] [TOP_K=...]'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.query --repos $(REPO_LIST) --hybrid "$(QUERY)" --fake \
		--top-k $(TOP_K) $(MMR_REPO_LAMBDA_FLAG) $(MMR_FILE_LAMBDA_FLAG)

diagnose: _require-db ## Print resolution stats for a repo branch in $(DB). REPO_NAME=name [BRANCH=name]
	@if [ -z "$(REPO_NAME)" ]; then echo 'usage: make diagnose DB=name REPO_NAME=name [BRANCH=name]'; exit 2; fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.diagnose --repo-name $(REPO_NAME) $(BRANCH_FLAG) --unresolved

graph: _require-db ## Graph exploration. REPO_NAME=name CMD="callers-of foo" [BRANCH=name]
	@if [ -z "$(REPO_NAME)" ] || [ -z "$(CMD)" ]; then \
		echo 'usage: make graph DB=name REPO_NAME=name CMD="callers-of foo" [BRANCH=name]'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.graph --repo-name $(REPO_NAME) $(BRANCH_FLAG) $(CMD)

# ------------------------------------------------------------------------------
# Branch management
# ------------------------------------------------------------------------------
branches: _require-db ## List branches per repo in $(DB) with file counts.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.branches list

branch-set-default: _require-db ## Re-flag a repo's default branch. REPO_NAME=name BRANCH=name
	@if [ -z "$(REPO_NAME)" ] || [ -z "$(BRANCH)" ]; then \
		echo 'usage: make branch-set-default DB=name REPO_NAME=name BRANCH=name'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.branches set-default --repo-name $(REPO_NAME) --branch $(BRANCH)

branch-drop: _require-db ## Delete a non-default branch (cascade). REPO_NAME=name BRANCH=name
	@if [ -z "$(REPO_NAME)" ] || [ -z "$(BRANCH)" ]; then \
		echo 'usage: make branch-drop DB=name REPO_NAME=name BRANCH=name'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.branches drop --repo-name $(REPO_NAME) --branch $(BRANCH)

gc: _require-db ## Reclaim orphan file_versions and chunk_embeddings in $(DB).
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.branches gc
