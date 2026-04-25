.PHONY: help install lint test test-extractor \
        db-start db-stop db-create db-drop db-migrate db-setup db-teardown db-reset db-psql \
        index-python index-solidity diagnose diagnose-isolated \
        embed-python embed-solidity query-semantic query-hybrid \
        index embed \
        index-isolated embed-isolated query-isolated-semantic query-isolated-hybrid \
        db-ensure-isolated

UV ?= uv
PYTHON := .venv/bin/python

PG_SERVICE ?= postgresql@18
PG_DB ?= tsgrep
MIGRATIONS_DIR := db/migrations

PYTHON_FIXTURE := tests/fixtures/python_fixture
SOLIDITY_FIXTURE := tests/fixtures/solidity_foundry_fixture

PASS_CLI ?= pass-cli
ENV_TEMPLATE ?= .env.template

# DB selection. Default `tsgrep` is the shared multi-repo DB. Override on
# any target with DB=tsgrep_myrepo, or use the *-isolated variants which do
# this automatically.
DB ?= $(PG_DB)
DB_USER ?= $(USER)
DB_DSN := postgresql://$(DB_USER)@localhost:5432/$(DB)

# Run $(1) with:
#   - secrets streamed from pass-cli into the env (no file on disk),
#   - DATABASE_URL pointing at $(DB).
define inject_and_run
	@OUTPUT=$$($(PASS_CLI) inject --in-file $(ENV_TEMPLATE)) || { echo "pass-cli inject failed"; exit 1; }; \
	set -a; \
	eval "$$OUTPUT"; \
	set +a; \
	unset OUTPUT; \
	exec env DATABASE_URL=$(DB_DSN) $(1)
endef

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Python / uv
# ------------------------------------------------------------------------------
install: ## Create venv (Python 3.12) and install project + dev deps with uv.
	@$(UV) venv --python 3.12
	@$(UV) pip install -e ".[dev]"

lint: ## Run ruff over the source tree.
	@$(PYTHON) -m ruff check core cli db tests

test: ## Run all pytest tests.
	@$(PYTHON) -m pytest -v

test-extractor: ## Run only Phase 1 extractor tests.
	@$(PYTHON) -m pytest -v tests/test_extractor.py

test-resolver: ## Run only Phase 2 semantic resolver tests.
	@$(PYTHON) -m pytest -v tests/test_semantic_resolver.py

test-imports: ## Run only Phase 3 heuristic resolver tests.
	@$(PYTHON) -m pytest -v tests/test_heuristic_resolver.py

test-chunks: ## Run only Phase 4 chunk + retrieval tests.
	@$(PYTHON) -m pytest -v tests/test_chunk_assembler.py tests/test_retrieval.py

# ------------------------------------------------------------------------------
# Local PostgreSQL (Homebrew). Override PG_SERVICE / PG_DB as needed.
# ------------------------------------------------------------------------------
db-start: ## Start the local PostgreSQL service.
	@brew services start $(PG_SERVICE)

db-stop: ## Stop the local PostgreSQL service.
	@brew services stop $(PG_SERVICE)

db-create: ## Create the $(PG_DB) database (idempotent).
	@createdb $(PG_DB) 2>/dev/null || echo "database $(PG_DB) already exists"

db-drop: ## Drop the $(PG_DB) database if it exists.
	@dropdb --if-exists $(PG_DB)

db-migrate: ## Apply all unapplied forward migrations against $(PG_DB).
	@psql -d $(PG_DB) -c \
		"CREATE TABLE IF NOT EXISTS schema_migrations (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());" \
		> /dev/null
	@for f in $(MIGRATIONS_DIR)/*.up.sql; do \
		fname=$$(basename $$f); \
		applied=$$(psql -d $(PG_DB) -tAc "SELECT 1 FROM schema_migrations WHERE filename = '$$fname'"); \
		if [ "$$applied" = "1" ]; then \
			echo "skipping $$fname (already applied)"; \
		else \
			echo "applying $$fname"; \
			psql -d $(PG_DB) -v ON_ERROR_STOP=1 -f $$f || exit 1; \
			psql -d $(PG_DB) -c "INSERT INTO schema_migrations (filename) VALUES ('$$fname');" > /dev/null; \
		fi; \
	done

db-setup: db-start db-create db-migrate ## Start the service, create the DB, and apply migrations.

db-teardown: db-drop ## Drop the $(PG_DB) database (service keeps running).

db-reset: db-drop db-create db-migrate ## Drop and recreate $(PG_DB) from scratch.

db-psql: ## Open a psql shell on $(PG_DB).
	@psql -d $(PG_DB)

# ------------------------------------------------------------------------------
# tsgrep CLI helpers
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Mode A — multi-repo into one shared DB (default `tsgrep`).
# Repos are addressed by name; cross-repo queries are possible.
# ------------------------------------------------------------------------------
PYTHON_REPO_NAME   ?= python_fixture
SOLIDITY_REPO_NAME ?= solidity_fixture

REPO_PATH ?=
REPO_NAME ?=
QUERY     ?=
EXCLUDE   ?=

# When EXCLUDE is set, expand to a single --exclude flag carrying the
# comma-separated value (argparse splits on comma).
EXCLUDE_FLAG := $(if $(EXCLUDE),--exclude '$(EXCLUDE)',)

index-python: ## Index the Python test fixture into the shared DB.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-name $(PYTHON_REPO_NAME)

index-solidity: ## Index the Solidity test fixture into the shared DB.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-name $(SOLIDITY_REPO_NAME)

embed-python-fake: ## Embed Python fixture chunks with the deterministic stub.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-name $(PYTHON_REPO_NAME) --embed fake

embed-solidity-fake: ## Embed Solidity fixture chunks with the deterministic stub.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-name $(SOLIDITY_REPO_NAME) --embed fake

embed-python: ## Embed Python fixture chunks (real embedder, secrets via pass-cli).
	$(call inject_and_run,$(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-name $(PYTHON_REPO_NAME) --embed real)

embed-solidity: ## Embed Solidity fixture chunks (real embedder, secrets via pass-cli).
	$(call inject_and_run,$(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-name $(SOLIDITY_REPO_NAME) --embed real)

# Generic targets — any repo into $(DB) (default `tsgrep`).
#   make index REPO_PATH=/path REPO_NAME=name [DB=tsgrep_other]
index: ## Index any repo into $(DB). REPO_PATH=/path REPO_NAME=name [EXCLUDE='pat1,pat2']
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make index REPO_PATH=/path REPO_NAME=name [EXCLUDE=...]'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) $(EXCLUDE_FLAG)

embed: ## Index + embed any repo into $(DB) (real embedder, secrets via pass-cli). REPO_PATH=/path REPO_NAME=name [EXCLUDE='pat1,pat2']
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make embed REPO_PATH=/path REPO_NAME=name [EXCLUDE=...]'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) --embed real $(EXCLUDE_FLAG))

query-semantic: ## Semantic query. QUERY="..." REPO_NAME=name [DB=...]
	@if [ -z "$(QUERY)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make query-semantic QUERY="..." REPO_NAME=name'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.query --repo-name $(REPO_NAME) --semantic "$(QUERY)")

query-hybrid: ## Hybrid query. QUERY="..." REPO_NAME=name [DB=...]
	@if [ -z "$(QUERY)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make query-hybrid QUERY="..." REPO_NAME=name'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.query --repo-name $(REPO_NAME) --hybrid "$(QUERY)")

diagnose-python: ## Print resolution stats for the Python fixture.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.diagnose --repo-name $(PYTHON_REPO_NAME) --unresolved

diagnose-solidity: ## Print resolution stats for the Solidity fixture.
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.diagnose --repo-name $(SOLIDITY_REPO_NAME) --unresolved

diagnose: ## Print resolution stats for any repo in $(DB). REPO_NAME=name [DB=...]
	@if [ -z "$(REPO_NAME)" ]; then echo 'usage: make diagnose REPO_NAME=name [DB=...]'; exit 2; fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.diagnose --repo-name $(REPO_NAME) --unresolved

# ------------------------------------------------------------------------------
# Mode B — one DB per repo (CI-friendly, per-repo isolation).
# Each *-isolated target derives DB=tsgrep_$(REPO_NAME), creates and migrates
# that DB if needed, then delegates to the generic target above.
# ------------------------------------------------------------------------------
db-ensure-isolated: ## Create (idempotent) and migrate tsgrep_$(REPO_NAME).
	@if [ -z "$(REPO_NAME)" ]; then echo 'usage: requires REPO_NAME=name'; exit 2; fi
	@createdb tsgrep_$(REPO_NAME) 2>/dev/null || true
	@$(MAKE) --no-print-directory db-migrate PG_DB=tsgrep_$(REPO_NAME)

index-isolated: ## Index any repo into its own DB tsgrep_$(REPO_NAME). REPO_PATH=/path REPO_NAME=name [EXCLUDE=...]
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make index-isolated REPO_PATH=/path REPO_NAME=name [EXCLUDE=...]'; exit 2; \
	fi
	@$(MAKE) --no-print-directory db-ensure-isolated REPO_NAME=$(REPO_NAME)
	@$(MAKE) --no-print-directory index REPO_PATH=$(REPO_PATH) REPO_NAME=$(REPO_NAME) DB=tsgrep_$(REPO_NAME) EXCLUDE='$(EXCLUDE)'

embed-isolated: ## Index + embed any repo into its own DB. REPO_PATH=/path REPO_NAME=name [EXCLUDE=...]
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make embed-isolated REPO_PATH=/path REPO_NAME=name [EXCLUDE=...]'; exit 2; \
	fi
	@$(MAKE) --no-print-directory db-ensure-isolated REPO_NAME=$(REPO_NAME)
	@$(MAKE) --no-print-directory embed REPO_PATH=$(REPO_PATH) REPO_NAME=$(REPO_NAME) DB=tsgrep_$(REPO_NAME) EXCLUDE='$(EXCLUDE)'

query-isolated-semantic: ## Semantic query against per-repo DB. QUERY="..." REPO_NAME=name
	@$(MAKE) --no-print-directory query-semantic QUERY="$(QUERY)" REPO_NAME=$(REPO_NAME) DB=tsgrep_$(REPO_NAME)

query-isolated-hybrid: ## Hybrid query against per-repo DB. QUERY="..." REPO_NAME=name
	@$(MAKE) --no-print-directory query-hybrid QUERY="$(QUERY)" REPO_NAME=$(REPO_NAME) DB=tsgrep_$(REPO_NAME)

diagnose-isolated: ## Print resolution stats from per-repo DB tsgrep_$(REPO_NAME). REPO_NAME=name
	@if [ -z "$(REPO_NAME)" ]; then echo 'usage: make diagnose-isolated REPO_NAME=name'; exit 2; fi
	@$(MAKE) --no-print-directory diagnose REPO_NAME=$(REPO_NAME) DB=tsgrep_$(REPO_NAME)
