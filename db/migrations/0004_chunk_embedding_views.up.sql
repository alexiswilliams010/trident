-- Multi-view embeddings: each chunk can now have several embedding rows,
-- one per "view" of the same chunk. Views are different *texts* fed to the
-- embedder (e.g. raw source vs source + inlined referenced types). The
-- chunk's `content` column is unchanged — display content stays separate
-- from embedded content. See core/embed_views.py for view definitions.

ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS view_kind TEXT;
UPDATE chunk_embeddings SET view_kind = 'source' WHERE view_kind IS NULL;
ALTER TABLE chunk_embeddings ALTER COLUMN view_kind SET NOT NULL;

-- Stores the exact text that was sent to the embedder, plus a hash. The hash
-- lets the embedder skip re-embedding a (chunk, view, model) triple whose
-- input is unchanged, even if the chunk's `content` shifts in ways that
-- don't affect this view's text.
ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS input_text TEXT;
ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS input_text_hash TEXT;

-- Drop the auto-named UNIQUE on chunk_id so we can keep multiple views per
-- chunk. The constraint was created implicitly via `chunk_id ... UNIQUE` in
-- 0001_init; Postgres names it deterministically but we look it up to be safe.
DO $$
DECLARE n TEXT;
BEGIN
    SELECT conname INTO n FROM pg_constraint
        WHERE conrelid = 'chunk_embeddings'::regclass
          AND contype = 'u'
          AND pg_get_constraintdef(oid) LIKE '%(chunk_id)';
    IF n IS NOT NULL THEN
        EXECUTE format('ALTER TABLE chunk_embeddings DROP CONSTRAINT %I', n);
    END IF;
END $$;

-- (chunk_id, view_kind, model_name) is the new logical key. Including
-- model_name keeps a model swap from colliding with rows for the prior model.
ALTER TABLE chunk_embeddings
    ADD CONSTRAINT chunk_embeddings_chunk_view_model_key
    UNIQUE (chunk_id, view_kind, model_name);
