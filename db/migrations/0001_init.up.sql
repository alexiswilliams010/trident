-- ═══════════════════════════════════════════════════════════
-- tsgrep schema (Architecture.md §3.2)
-- Initial migration: Tier 1 (syntactic) + Tier 2 (semantic) + Tier 3 (chunks/embeddings).
-- Idempotent: safe to re-run.
-- Requires: pgvector extension (`brew install pgvector`).
-- ═══════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;

-- ═══════════════════════════════════════════════════════════
-- TIER 0: Repo + branch directory
--
-- A repo can host multiple branches. Each branch is a logical view of a
-- working tree at a point in time; many branches typically share most file
-- content (file_versions are content-keyed and shared per repo).
--
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS repos (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    root_path   TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS branches (
    id          BIGSERIAL PRIMARY KEY,
    repo_id     BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    is_default  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo_id, name)
);

-- One default branch per repo. Partial unique index makes the constraint
-- cheap (only the small set of is_default=TRUE rows are indexed).
CREATE UNIQUE INDEX IF NOT EXISTS idx_branches_one_default
    ON branches(repo_id) WHERE is_default;

-- ═══════════════════════════════════════════════════════════
-- TIER 1: Syntactic tables
--
-- file_versions: content-keyed source. Two branches with identical content
--                for some path share one file_versions row, so the parse
--                output (nodes, definitions, chunks) is shared too. No
--                tree-sitter re-parse and no chunk re-embed across branches
--                with shared content.
--
-- branch_files:  per-branch (path → file_version) mapping. The "view" of a
--                branch is its set of branch_files rows.
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS file_versions (
    id            BIGSERIAL PRIMARY KEY,
    repo_id       BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    content_hash  TEXT NOT NULL,
    language      TEXT NOT NULL,
    raw_content   TEXT,
    indexed_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (repo_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_file_versions_repo ON file_versions(repo_id);

CREATE TABLE IF NOT EXISTS branch_files (
    branch_id        BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    path             TEXT NOT NULL,
    file_version_id  BIGINT NOT NULL REFERENCES file_versions(id) ON DELETE RESTRICT,
    from_dependency  BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (branch_id, path)
);

CREATE INDEX IF NOT EXISTS idx_branch_files_fv ON branch_files(file_version_id);

CREATE TABLE IF NOT EXISTS nodes (
    id              BIGINT PRIMARY KEY,
    file_version_id BIGINT NOT NULL REFERENCES file_versions(id) ON DELETE CASCADE,
    node_type       TEXT NOT NULL,
    is_named        BOOLEAN NOT NULL,
    start_byte      INT NOT NULL,
    end_byte        INT NOT NULL,
    start_row       INT NOT NULL,
    start_col       INT NOT NULL,
    end_row         INT NOT NULL,
    end_col         INT NOT NULL,
    text            TEXT,
    parent_id       BIGINT REFERENCES nodes(id) DEFERRABLE INITIALLY DEFERRED,
    child_index     INT
);

CREATE INDEX IF NOT EXISTS idx_nodes_file_version ON nodes(file_version_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent       ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type         ON nodes(node_type);

CREATE SEQUENCE IF NOT EXISTS nodes_id_seq;

-- ═══════════════════════════════════════════════════════════
-- TIER 2: Semantic tables
--
-- Definitions are content-derived (deterministic from the parsed AST), so
-- they hang off file_version_id and are SHARED across branches that include
-- the same file_version.
--
-- Cross-file resolution outputs (refs, calls, data_access, inherits, overrides,
-- imports, external_dependencies) depend on the *set of files visible in the
-- branch*, so every row is tagged with branch_id.
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS definitions (
    id              BIGSERIAL PRIMARY KEY,
    node_id         BIGINT REFERENCES nodes(id) ON DELETE CASCADE UNIQUE,
    file_version_id BIGINT REFERENCES file_versions(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL,
    qualified_name  TEXT,
    scope_id        BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    visibility      TEXT
);

CREATE INDEX IF NOT EXISTS idx_definitions_name         ON definitions(name);
CREATE INDEX IF NOT EXISTS idx_definitions_kind         ON definitions(kind);
CREATE INDEX IF NOT EXISTS idx_definitions_scope        ON definitions(scope_id);
CREATE INDEX IF NOT EXISTS idx_definitions_file_version ON definitions(file_version_id);

CREATE TABLE IF NOT EXISTS "references" (
    id                    BIGSERIAL PRIMARY KEY,
    branch_id             BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    node_id               BIGINT REFERENCES nodes(id) ON DELETE CASCADE,
    file_version_id       BIGINT REFERENCES file_versions(id) ON DELETE CASCADE,
    target_def_id         BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    name                  TEXT NOT NULL,
    resolution_confidence FLOAT DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_references_target          ON "references"(target_def_id);
CREATE INDEX IF NOT EXISTS idx_references_name            ON "references"(name);
CREATE INDEX IF NOT EXISTS idx_references_branch_fv       ON "references"(branch_id, file_version_id);

CREATE TABLE IF NOT EXISTS external_dependencies (
    id           BIGSERIAL PRIMARY KEY,
    branch_id    BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    package_name TEXT NOT NULL,
    version      TEXT,
    language     TEXT NOT NULL,
    source       TEXT,
    UNIQUE (branch_id, package_name, language)
);

CREATE INDEX IF NOT EXISTS idx_external_dependencies_branch ON external_dependencies(branch_id);

CREATE TABLE IF NOT EXISTS imports (
    id                       BIGSERIAL PRIMARY KEY,
    branch_id                BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    file_version_id          BIGINT REFERENCES file_versions(id) ON DELETE CASCADE,
    node_id                  BIGINT REFERENCES nodes(id) ON DELETE CASCADE,
    import_path              TEXT NOT NULL,
    resolved_file_version_id BIGINT REFERENCES file_versions(id) ON DELETE SET NULL,
    imported_names           TEXT[],
    dep_class                TEXT NOT NULL DEFAULT 'unknown'
                             CHECK (dep_class IN ('intra_repo', 'external', 'unresolved')),
    external_dep_id          BIGINT REFERENCES external_dependencies(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_imports_file_version ON imports(file_version_id);
CREATE INDEX IF NOT EXISTS idx_imports_resolved     ON imports(resolved_file_version_id);
CREATE INDEX IF NOT EXISTS idx_imports_dep_class    ON imports(dep_class);
CREATE INDEX IF NOT EXISTS idx_imports_branch       ON imports(branch_id);

CREATE TABLE IF NOT EXISTS call_edges (
    id               BIGSERIAL PRIMARY KEY,
    branch_id        BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    callsite_node_id BIGINT REFERENCES nodes(id) ON DELETE CASCADE,
    caller_def_id    BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    callee_def_id    BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    -- Callee name as written at the call site (last identifier of the function expression).
    -- Stored to enable Phase 3 cross-file re-linking without re-parsing the CST.
    callee_name      TEXT,
    confidence       TEXT NOT NULL DEFAULT 'certain'
                     CHECK (confidence IN ('certain', 'inferred', 'uncertain'))
);

CREATE INDEX IF NOT EXISTS idx_call_edges_callee_name ON call_edges(callee_name);

CREATE INDEX IF NOT EXISTS idx_call_edges_caller     ON call_edges(caller_def_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee     ON call_edges(callee_def_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_confidence ON call_edges(confidence);
CREATE INDEX IF NOT EXISTS idx_call_edges_branch     ON call_edges(branch_id);

CREATE TABLE IF NOT EXISTS data_access (
    id              BIGSERIAL PRIMARY KEY,
    branch_id       BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    accessor_def_id BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    target_def_id   BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    access_type     TEXT NOT NULL CHECK (access_type IN ('read', 'write', 'readwrite')),
    node_id         BIGINT REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_data_access_accessor ON data_access(accessor_def_id);
CREATE INDEX IF NOT EXISTS idx_data_access_target   ON data_access(target_def_id);
CREATE INDEX IF NOT EXISTS idx_data_access_branch   ON data_access(branch_id);

-- Inheritance graph between contract/interface/class definitions.
-- Solidity:  contract Foo is Bar { ... }      → child=Foo, base_name='Bar'
-- Python:    class Foo(Bar): ...              → child=Foo, base_name='Bar'
-- `ord` preserves declaration order (matters for C3 linearization / Solidity MRO).
-- `base_def_id` is filled in by the resolver: intra-file in semantic_resolver,
-- cross-file via the heuristic resolver's import-aware linking pass.
--
-- Per-branch row: the same intra-file inheritance is regenerated for each
-- branch that includes the file_version. The table is small (typically O(N)
-- in the number of class/interface declarations) so the duplication cost is
-- acceptable in exchange for a clean per-branch resolution model.
CREATE TABLE IF NOT EXISTS inherits_edges (
    id              BIGSERIAL PRIMARY KEY,
    branch_id       BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    child_def_id    BIGINT NOT NULL REFERENCES definitions(id) ON DELETE CASCADE,
    base_name       TEXT NOT NULL,
    base_def_id     BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    confidence      TEXT NOT NULL DEFAULT 'certain'
                    CHECK (confidence IN ('certain', 'inferred', 'uncertain')),
    ord             INT NOT NULL,
    UNIQUE (branch_id, child_def_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_inherits_child  ON inherits_edges(child_def_id);
CREATE INDEX IF NOT EXISTS idx_inherits_base   ON inherits_edges(base_def_id);
CREATE INDEX IF NOT EXISTS idx_inherits_name   ON inherits_edges(base_name);
CREATE INDEX IF NOT EXISTS idx_inherits_branch ON inherits_edges(branch_id);

-- Method-override edges generated after inherits_edges is resolved.
-- Pair (child_def_id, base_def_id) means: child function/method/modifier
-- shadows / overrides the base one with the same name. Generated for the
-- *nearest* matching ancestor; transitive shadowing collapses to the closest.
CREATE TABLE IF NOT EXISTS overrides_edges (
    id              BIGSERIAL PRIMARY KEY,
    branch_id       BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    child_def_id    BIGINT NOT NULL REFERENCES definitions(id) ON DELETE CASCADE,
    base_def_id     BIGINT NOT NULL REFERENCES definitions(id) ON DELETE CASCADE,
    UNIQUE (branch_id, child_def_id, base_def_id)
);

CREATE INDEX IF NOT EXISTS idx_overrides_child  ON overrides_edges(child_def_id);
CREATE INDEX IF NOT EXISTS idx_overrides_base   ON overrides_edges(base_def_id);
CREATE INDEX IF NOT EXISTS idx_overrides_branch ON overrides_edges(branch_id);

-- ═══════════════════════════════════════════════════════════
-- TIER 3: Chunk & embedding tables
--
-- Chunks are per-branch because the inlined skeleton (cross-file refs to
-- callees, accessed data, base classes, etc.) reflects branch-resolved
-- targets, so chunk content can differ per branch even for the same anchor.
--
-- Embeddings are content-keyed: identical chunk text across branches reuses
-- one embedding row. This is the API-cost saver — re-indexing a branch with
-- mostly-shared content does ~0 new embedding API calls.
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    branch_id       BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    file_version_id BIGINT REFERENCES file_versions(id) ON DELETE CASCADE,
    anchor_def_id   BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    granularity     TEXT NOT NULL,
    content         TEXT NOT NULL,
    token_count     INT,
    metadata        JSONB,
    content_hash    TEXT NOT NULL,
    UNIQUE (branch_id, anchor_def_id, granularity)
);

CREATE INDEX IF NOT EXISTS idx_chunks_branch        ON chunks(branch_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash  ON chunks(content_hash);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id            BIGSERIAL PRIMARY KEY,
    content_hash  TEXT NOT NULL UNIQUE,
    embedding     vector(4096),
    model_name    TEXT NOT NULL
);

-- No ANN index: pgvector caps HNSW at 2000 dims for `vector` (4000 for
-- `halfvec`), and qwen3-embedding-8b is natively 4096. Cosine NN runs as a
-- sequential scan, which is fine at small/medium chunk counts. See
-- docs/halfvec-migration.md for the path to halfvec(4000) + HNSW when scale
-- demands it.
