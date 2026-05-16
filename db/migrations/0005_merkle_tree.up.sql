-- Size-only Merkle tree for skipping unchanged files on re-index.
--
-- branch_files.size caches each file's byte length so the walker can compare
-- against os.stat().st_size and skip read+SHA-256 when size matches.
--
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
