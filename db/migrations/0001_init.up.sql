-- ═══════════════════════════════════════════════════════════
-- tsgrep schema (Architecture.md §3.2)
-- Initial migration: Tier 1 (syntactic) + Tier 2 (semantic) + Tier 3 (chunks/embeddings).
-- Idempotent: safe to re-run.
-- Requires: pgvector extension (`brew install pgvector`).
-- ═══════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;

-- ═══════════════════════════════════════════════════════════
-- TIER 1: Syntactic tables
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS files (
    id              BIGSERIAL PRIMARY KEY,
    repo_id         BIGINT NOT NULL,
    path            TEXT NOT NULL,
    language        TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    raw_content     TEXT,
    from_dependency BOOLEAN NOT NULL DEFAULT FALSE,
    indexed_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (repo_id, path)
);

CREATE TABLE IF NOT EXISTS nodes (
    id           BIGINT PRIMARY KEY,
    file_id      BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    node_type    TEXT NOT NULL,
    is_named     BOOLEAN NOT NULL,
    start_byte   INT NOT NULL,
    end_byte     INT NOT NULL,
    start_row    INT NOT NULL,
    start_col    INT NOT NULL,
    end_row      INT NOT NULL,
    end_col      INT NOT NULL,
    text         TEXT,
    parent_id    BIGINT REFERENCES nodes(id) DEFERRABLE INITIALLY DEFERRED,
    child_index  INT
);

CREATE INDEX IF NOT EXISTS idx_nodes_file   ON nodes(file_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type   ON nodes(node_type);

CREATE SEQUENCE IF NOT EXISTS nodes_id_seq;

-- ═══════════════════════════════════════════════════════════
-- TIER 2: Semantic tables
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS definitions (
    id              BIGSERIAL PRIMARY KEY,
    node_id         BIGINT REFERENCES nodes(id) ON DELETE CASCADE UNIQUE,
    file_id         BIGINT REFERENCES files(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL,
    qualified_name  TEXT,
    scope_id        BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    visibility      TEXT
);

CREATE INDEX IF NOT EXISTS idx_definitions_name  ON definitions(name);
CREATE INDEX IF NOT EXISTS idx_definitions_kind  ON definitions(kind);
CREATE INDEX IF NOT EXISTS idx_definitions_scope ON definitions(scope_id);
CREATE INDEX IF NOT EXISTS idx_definitions_file  ON definitions(file_id);

CREATE TABLE IF NOT EXISTS "references" (
    id                    BIGSERIAL PRIMARY KEY,
    node_id               BIGINT REFERENCES nodes(id) ON DELETE CASCADE,
    file_id               BIGINT REFERENCES files(id) ON DELETE CASCADE,
    target_def_id         BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    name                  TEXT NOT NULL,
    resolution_confidence FLOAT DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_references_target ON "references"(target_def_id);
CREATE INDEX IF NOT EXISTS idx_references_name   ON "references"(name);

CREATE TABLE IF NOT EXISTS external_dependencies (
    id           BIGSERIAL PRIMARY KEY,
    repo_id      BIGINT NOT NULL,
    package_name TEXT NOT NULL,
    version      TEXT,
    language     TEXT NOT NULL,
    source       TEXT,
    UNIQUE (repo_id, package_name, language)
);

CREATE TABLE IF NOT EXISTS imports (
    id               BIGSERIAL PRIMARY KEY,
    file_id          BIGINT REFERENCES files(id) ON DELETE CASCADE,
    node_id          BIGINT REFERENCES nodes(id) ON DELETE CASCADE,
    import_path      TEXT NOT NULL,
    resolved_file_id BIGINT REFERENCES files(id) ON DELETE SET NULL,
    imported_names   TEXT[],
    dep_class        TEXT NOT NULL DEFAULT 'unknown'
                     CHECK (dep_class IN ('intra_repo', 'external', 'unresolved')),
    external_dep_id  BIGINT REFERENCES external_dependencies(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_imports_file       ON imports(file_id);
CREATE INDEX IF NOT EXISTS idx_imports_resolved   ON imports(resolved_file_id);
CREATE INDEX IF NOT EXISTS idx_imports_dep_class  ON imports(dep_class);

CREATE TABLE IF NOT EXISTS call_edges (
    id               BIGSERIAL PRIMARY KEY,
    callsite_node_id BIGINT REFERENCES nodes(id) ON DELETE CASCADE,
    caller_def_id    BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    callee_def_id    BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    confidence       TEXT NOT NULL DEFAULT 'certain'
                     CHECK (confidence IN ('certain', 'inferred', 'uncertain'))
);

CREATE INDEX IF NOT EXISTS idx_call_edges_caller     ON call_edges(caller_def_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee     ON call_edges(callee_def_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_confidence ON call_edges(confidence);

CREATE TABLE IF NOT EXISTS data_access (
    id              BIGSERIAL PRIMARY KEY,
    accessor_def_id BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    target_def_id   BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    access_type     TEXT NOT NULL CHECK (access_type IN ('read', 'write', 'readwrite')),
    node_id         BIGINT REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_data_access_accessor ON data_access(accessor_def_id);
CREATE INDEX IF NOT EXISTS idx_data_access_target   ON data_access(target_def_id);

-- ═══════════════════════════════════════════════════════════
-- TIER 3: Chunk & embedding tables
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    file_id       BIGINT REFERENCES files(id) ON DELETE CASCADE,
    anchor_def_id BIGINT REFERENCES definitions(id) ON DELETE CASCADE,
    granularity   TEXT NOT NULL,
    content       TEXT NOT NULL,
    token_count   INT,
    metadata      JSONB,
    content_hash  TEXT NOT NULL,
    UNIQUE (anchor_def_id, granularity)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id         BIGSERIAL PRIMARY KEY,
    chunk_id   BIGINT REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding  vector(1024),
    model_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_vector ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops);
