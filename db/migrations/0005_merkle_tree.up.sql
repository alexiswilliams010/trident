-- Size-only Merkle tree for skipping unchanged files on re-index, plus the
-- FK indexes needed to keep `make gc` from going quadratic on the cascade.

-- ── Merkle tree ────────────────────────────────────────────────────
-- branch_files.size caches each file's byte length so the walker can
-- compare against os.stat().st_size and skip read+SHA-256 when it matches.
-- branch_dirs stores a SHA-256 over each directory's sorted manifest of
-- (kind, name, child_hash) entries, propagating bottom-up so the root row
-- answers "did anything in this branch change" in one lookup.

ALTER TABLE branch_files ADD COLUMN IF NOT EXISTS size BIGINT;

CREATE TABLE IF NOT EXISTS branch_dirs (
    branch_id  BIGINT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    tree_hash  TEXT NOT NULL,
    PRIMARY KEY (branch_id, path)
);

-- ── FK indexes for cascade-delete performance ──────────────────────
-- Every FK referencing file_versions(id) or nodes(id) with ON DELETE CASCADE
-- needs an index on its referencing column, or `gc` seq-scans the child
-- table once per parent row deleted. references.node_id was the worst
-- offender: EXPLAIN ANALYZE on a 292-node cascade showed its trigger
-- alone taking 1.77s; with the index it drops to ~0.4ms (a similar
-- cascade on 2876 nodes ran in 4ms total).

CREATE INDEX IF NOT EXISTS idx_chunks_file_version  ON chunks(file_version_id);
CREATE INDEX IF NOT EXISTS idx_imports_node         ON imports(node_id);
CREATE INDEX IF NOT EXISTS idx_data_access_node     ON data_access(node_id);
CREATE INDEX IF NOT EXISTS idx_references_node      ON "references"(node_id);
