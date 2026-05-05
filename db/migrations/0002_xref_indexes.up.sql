-- Indexes that the Phase 3 cross-file linker (heuristic_resolver) depends on.
-- Without these, every UPDATE in the linker did a full scan of either
-- references (~hundreds of thousands of rows on a real repo) or call_edges
-- (~tens of thousands of rows). With ~one importer file × multiple UPDATEs
-- per file, the cost compounded into multi-minute hangs on indexing runs.
--
-- - references(file_version_id): drives the per-file scans in Tier-A direct
--   linking and Tier-B fuzzy linking, plus the per-file cleanup in
--   semantic_resolver._clear_branch_semantic_for_file_version. Note that
--   idx_references_branch_fv (in 0001) already covers this when the linker
--   filters by branch_id; this single-column index is kept for any code path
--   that filters only on file_version_id.
--
-- - call_edges(callsite_node_id): drives `WHERE ce.callsite_node_id IN
--   (SELECT id FROM nodes WHERE file_version_id = $1)` in both Tier-A and
--   Tier-B linkers, plus the per-file cleanup. The matching
--   nodes(file_version_id) index already exists, but call_edges itself was
--   un-indexed on this column.

CREATE INDEX IF NOT EXISTS idx_references_file_version ON "references"(file_version_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callsite     ON call_edges(callsite_node_id);
