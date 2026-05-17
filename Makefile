.PHONY: help install lint test test-extractor test-resolver test-imports test-chunks \
        db-start db-stop db-create db-drop db-migrate db-setup db-teardown db-reset db-psql \
        db-test-setup \
        diagnose graph \
        query-semantic query-lexical query-hybrid query-fake-hybrid \
        index embed profile-index profile-view \
        branches branch-set-default branch-drop gc \
        benchmark \
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

# FORCE=1 bypasses every "looks unchanged" shortcut: ignore the size cache
# (re-read + re-hash every file) and re-run semantic resolution on the full
# repo. Use after editing core/semantic_resolver.py / configs/*.yaml, or
# when a content edit preserved file size and slipped past the size cache.
FORCE_FLAGS := $(if $(FORCE),--rehash --force-resolve,)

# Optional flag expansions — empty when the variable is unset, so the CLI
# falls back to its built-in defaults.
MMR_REPO_LAMBDA_FLAG  := $(if $(MMR_REPO_LAMBDA),--mmr-repo-lambda $(MMR_REPO_LAMBDA),)
MMR_FILE_LAMBDA_FLAG  := $(if $(MMR_FILE_LAMBDA),--mmr-file-lambda $(MMR_FILE_LAMBDA),)

index: _require-db ## Index a repo into $(DB). REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE='pat1,pat2']
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make index DB=name REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE=...]'; exit 2; \
	fi
	@env DATABASE_URL=$(DB_DSN) $(PYTHON) -m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) $(BRANCH_FLAG) $(EXCLUDE_FLAG) $(FORCE_FLAGS)

embed: _require-db ## Index + embed a repo into $(DB) (real embedder, secrets via pass-cli). REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE='pat1,pat2'] [FORCE=1]
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make embed DB=name REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE=...] [FORCE=1]'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) --embed real $(BRANCH_FLAG) $(EXCLUDE_FLAG) $(FORCE_FLAGS))

# Function-call tracer over the index run. Emits a JSON trace at
# profile/index-<timestamp>.json. Open it with `make profile-view
# TRACE=profile/index-<timestamp>.json` for the interactive flame
# graph + timeline (await spans are visible, so DB-wait shows up as
# real time blocks, not CPU). No sudo required — viztracer is an
# in-process tracer. Wipe + recreate the DB beforehand (e.g.
# `make db-reset DB=...`) to profile the cold path.
VIZTRACER ?= $(CURDIR)/.venv/bin/viztracer
VIZVIEWER ?= $(CURDIR)/.venv/bin/vizviewer
# Tunables: PROFILE_MIN_US drops calls shorter than N microseconds (skips
# per-node tree-sitter walk noise). PROFILE_ENTRIES grows the circular
# buffer if 1M still overflows after filtering.
PROFILE_MIN_US ?= 100
PROFILE_ENTRIES ?= 1000000
profile-index: _require-db ## Trace an index run. REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE='pat1,pat2']
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo 'usage: make profile-index DB=name REPO_PATH=/path REPO_NAME=name [BRANCH=name] [EXCLUDE=...]'; exit 2; \
	fi
	@if [ ! -x "$(VIZTRACER)" ]; then \
		echo "viztracer not found at $(VIZTRACER) — run: make install (or uv pip install viztracer)"; exit 2; \
	fi
	@mkdir -p profile
	@OUT=profile/index-$$(date +%Y%m%d-%H%M%S).json; \
		echo "tracing -> $$OUT (min_duration=$(PROFILE_MIN_US)us, entries=$(PROFILE_ENTRIES))"; \
		env DATABASE_URL=$(DB_DSN) $(VIZTRACER) --output_file $$OUT \
			--min_duration $(PROFILE_MIN_US)us --ignore_c_function --log_async \
			--tracer_entries $(PROFILE_ENTRIES) \
			-m cli.index $(REPO_PATH) --repo-name $(REPO_NAME) $(BRANCH_FLAG) $(EXCLUDE_FLAG) $(FORCE_FLAGS) && \
		echo "wrote $$OUT — view with: make profile-view TRACE=$$OUT"

profile-view: ## Open a viztracer JSON trace in the interactive viewer. TRACE=profile/index-<ts>.json
	@if [ -z "$(TRACE)" ]; then \
		echo 'usage: make profile-view TRACE=profile/index-<timestamp>.json'; exit 2; \
	fi
	@$(VIZVIEWER) $(TRACE)

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

# ------------------------------------------------------------------------------
# Benchmark: trident-vs-baseline agent comparison.
# ------------------------------------------------------------------------------
QUESTION ?=
MODEL    ?=
OUTPUT_DIR ?=
VERBOSE  ?=

MODEL_FLAG := $(if $(MODEL),--model $(MODEL),)
BRANCH_BENCH_FLAG := $(if $(BRANCH),--branch $(BRANCH),)
OUTPUT_DIR_FLAG := $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)
VERBOSE_FLAG := $(if $(VERBOSE),--verbose,)

benchmark: _require-db ## Run the trident-vs-baseline benchmark. REPO_PATH=/path REPO_NAME=name QUESTION="..." [BRANCH=name] [MODEL=claude-sonnet-4-6] [OUTPUT_DIR=path] [VERBOSE=1]
	@if [ -z "$(REPO_PATH)" ] || [ -z "$(REPO_NAME)" ] || [ -z "$(QUESTION)" ]; then \
		echo 'usage: make benchmark DB=name REPO_PATH=/path REPO_NAME=name QUESTION="..." [BRANCH=name] [MODEL=...] [OUTPUT_DIR=path] [VERBOSE=1]'; exit 2; \
	fi
	$(call inject_and_run,$(PYTHON) -m benchmark $(REPO_PATH) --db $(DB) --repo-name $(REPO_NAME) --question "$(QUESTION)" $(BRANCH_BENCH_FLAG) $(MODEL_FLAG) $(OUTPUT_DIR_FLAG) $(VERBOSE_FLAG))
