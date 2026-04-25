.PHONY: help install lint test test-extractor \
        db-start db-stop db-create db-drop db-migrate db-setup db-teardown db-reset db-psql \
        index-python index-solidity diagnose \
        embed-python embed-solidity query-semantic query-hybrid \
        index embed

UV ?= uv
PYTHON := .venv/bin/python

PG_SERVICE ?= postgresql@18
PG_DB ?= tsgrep
MIGRATIONS_DIR := db/migrations

PYTHON_FIXTURE := tests/fixtures/python_fixture
SOLIDITY_FIXTURE := tests/fixtures/solidity_foundry_fixture

PASS_CLI ?= pass-cli
ENV_TEMPLATE ?= .env.template

# Run $(1) with secrets streamed from pass-cli into the process env.
# pass-cli inject's stdout is eval'd then unset; no file is written to disk.
define inject_and_run
	@OUTPUT=$$($(PASS_CLI) inject --in-file $(ENV_TEMPLATE)) || { echo "pass-cli inject failed"; exit 1; }; \
	set -a; \
	eval "$$OUTPUT"; \
	set +a; \
	unset OUTPUT; \
	exec $(1)
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
index-python: ## Index the Python test fixture (repo-id 1).
	@$(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-id 1

index-solidity: ## Index the Solidity test fixture (repo-id 2).
	@$(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-id 2

embed-python-fake: ## Embed Python fixture chunks with the deterministic stub.
	@$(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-id 1 --embed fake

embed-solidity-fake: ## Embed Solidity fixture chunks with the deterministic stub.
	@$(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-id 2 --embed fake

embed-python: ## Embed Python fixture chunks (real embedder, secrets via pass-cli).
	$(call inject_and_run,$(PYTHON) -m cli.index $(PYTHON_FIXTURE) --repo-id 1 --embed real)

embed-solidity: ## Embed Solidity fixture chunks (real embedder, secrets via pass-cli).
	$(call inject_and_run,$(PYTHON) -m cli.index $(SOLIDITY_FIXTURE) --repo-id 2 --embed real)

# Generic targets for any repo. Pass REPO_PATH and REPO_ID on the command line:
#   make index REPO_PATH=/path/to/repo REPO_ID=42
#   make embed REPO_PATH=/path/to/repo REPO_ID=42
REPO_PATH ?=
REPO_ID ?=

index: ## Index any repo. REPO_PATH=/path REPO_ID=N
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_ID)" ]; then \
		echo 'usage: make index REPO_PATH=/path/to/repo REPO_ID=N'; exit 2; \
	fi
	@$(PYTHON) -m cli.index $(REPO_PATH) --repo-id $(REPO_ID)

embed: ## Index + embed any repo (real embedder, secrets via pass-cli). REPO_PATH=/path REPO_ID=N
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_ID)" ]; then \
		echo 'usage: make embed REPO_PATH=/path/to/repo REPO_ID=N'; exit 2; \
	fi
	@OUTPUT=$$($(PASS_CLI) inject --in-file $(ENV_TEMPLATE)) || { echo "pass-cli inject failed"; exit 1; }; \
	set -a; eval "$$OUTPUT"; set +a; unset OUTPUT; \
	exec $(PYTHON) -m cli.index $(REPO_PATH) --repo-id $(REPO_ID) --embed real

# Usage: make query-semantic QUERY="how does helper resolve?" [REPO=1]
REPO ?= 1
query-semantic: ## Run a semantic query. Pass QUERY="..." [REPO=N].
	@if [ -z "$(QUERY)" ]; then echo 'usage: make query-semantic QUERY="..." [REPO=1]'; exit 2; fi
	@OUTPUT=$$($(PASS_CLI) inject --in-file $(ENV_TEMPLATE)) || { echo "pass-cli inject failed"; exit 1; }; \
	set -a; eval "$$OUTPUT"; set +a; unset OUTPUT; \
	exec $(PYTHON) -m cli.query --repo-id $(REPO) --semantic "$(QUERY)"

query-hybrid: ## Run a hybrid query. Pass QUERY="..." [REPO=N].
	@if [ -z "$(QUERY)" ]; then echo 'usage: make query-hybrid QUERY="..." [REPO=1]'; exit 2; fi
	@OUTPUT=$$($(PASS_CLI) inject --in-file $(ENV_TEMPLATE)) || { echo "pass-cli inject failed"; exit 1; }; \
	set -a; eval "$$OUTPUT"; set +a; unset OUTPUT; \
	exec $(PYTHON) -m cli.query --repo-id $(REPO) --hybrid "$(QUERY)"

diagnose-python: ## Print resolution stats for the Python fixture (repo-id 1).
	@$(PYTHON) -m cli.diagnose --repo-id 1 --unresolved

diagnose-solidity: ## Print resolution stats for the Solidity fixture (repo-id 2).
	@$(PYTHON) -m cli.diagnose --repo-id 2 --unresolved
