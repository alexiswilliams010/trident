-- BM25-style lexical search layer for chunks. Pre-computed at chunk-upsert
-- time (see core/chunk_assembler._upsert_chunk) so reads don't pay the
-- tokenization cost. The text fed into to_tsvector is:
--
--   <qualified_name>  <camelCase-split qualified_name>  <content>
--
-- Including the qualified_name twice — once raw, once with CamelCase
-- boundaries split — lets a literal-identifier query like `fooBar` match
-- exactly while a token query like `foo` also matches compound identifiers
-- such as `fooBar`, `MyFooThing`, etc. The 'english' config handles
-- natural-language stemming in code comments.

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS fts_doc tsvector;

-- Backfill any rows that were inserted before this migration. We can't do
-- the CamelCase split in pure SQL without a regex helper, so we just stuff
-- qualified_name + content. Subsequent re-assemblies (which always rewrite
-- fts_doc when content changes) will replace this with the better
-- Python-built document.
UPDATE chunks AS c
SET fts_doc = to_tsvector(
    'english',
    COALESCE((SELECT d.qualified_name FROM definitions d WHERE d.id = c.anchor_def_id), '')
    || ' ' || c.content
)
WHERE c.fts_doc IS NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (fts_doc);
