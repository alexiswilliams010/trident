-- Indexes that the Phase 3 cross-file linker (heuristic_resolver) depends on.
-- Without these, every UPDATE in the linker did a full scan of either
-- references (~hundreds of thousands of rows on a real repo) or call_edges
-- (~tens of thousands of rows). With ~one importer file × multiple UPDATEs
-- per file, the cost compounded into multi-minute hangs on indexing runs.
--
-- - references(file_id): drives `WHERE r.file_id = $1` in Tier-A direct
--   linking and Tier-B fuzzy linking, plus the per-file cleanup in
--   semantic_resolver._clear_semantic_for_file.
--
-- - call_edges(callsite_node_id): drives `WHERE ce.callsite_node_id IN
--   (SELECT id FROM nodes WHERE file_id = $1)` in both Tier-A and Tier-B
--   linkers, plus the per-file cleanup. The matching nodes(file_id) index
--   already exists, but call_edges itself was un-indexed on this column.

CREATE INDEX IF NOT EXISTS idx_references_file       ON "references"(file_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callsite   ON call_edges(callsite_node_id);
