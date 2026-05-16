-- Partial covering index for the Tier-B fuzzy linker's unresolved-reference
-- lookups in core/heuristic_resolver._link_cross_file.
--
-- The linker repeatedly issues queries shaped like:
--
--   SELECT DISTINCT name FROM "references"
--   WHERE branch_id=$1 AND file_version_id=$2 AND target_def_id IS NULL
--
-- idx_references_branch_fv (in 0001) covers (branch_id, file_version_id) but
-- Postgres still needs heap lookups to filter on target_def_id IS NULL and to
-- project `name`. This partial index includes `name` and restricts to the
-- unresolved rows the linker actually scans, enabling an index-only scan.

CREATE INDEX IF NOT EXISTS idx_references_unresolved
    ON "references" (branch_id, file_version_id, name)
    WHERE target_def_id IS NULL;
