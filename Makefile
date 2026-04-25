.PHONY: help install lint test test-extractor \
        db-start db-stop db-create db-drop db-migrate db-setup db-teardown db-reset db-psql \
        index-python index-solidity diagnose

UV ?= uv
PYTHON := .venv/bin/python

PG_SERVICE ?= postgresql@18
PG_DB ?= tsgrep
MIGRATIONS_DIR := db/migrations

PYTHON_FIXTURE := tests/fixtures/python_fixture
SOLIDITY_FIXTURE := tests/fixtures/solidity_foundry_fixture

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
index-python: ## Index the Python test fixture (repo-id 1).
	@$(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-id 1

index-solidity: ## Index the Solidity test fixture (repo-id 2).
	@$(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-id 2

embed-python-fake: ## Embed Python fixture chunks with the deterministic stub.
	@$(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-id 1 --embed fake

embed-solidity-fake: ## Embed Solidity fixture chunks with the deterministic stub.
	@$(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-id 2 --embed fake

diagnose-python: ## Print resolution stats for the Python fixture (repo-id 1).
	@$(PYTHON) -m cli.diagnose --repo-id 1 --unresolved

diagnose-solidity: ## Print resolution stats for the Solidity fixture (repo-id 2).
	@$(PYTHON) -m cli.diagnose --repo-id 2 --unresolved
