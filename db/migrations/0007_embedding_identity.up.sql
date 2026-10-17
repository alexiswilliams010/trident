-- Existing vectors are retained, but their unrecorded model/representation
-- revisions are never silently treated as current. Re-embedding populates the
-- requested vector space without deleting historical vectors.
ALTER TABLE chunk_embeddings DROP CONSTRAINT chunk_embeddings_content_hash_key;
ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector;
ALTER TABLE chunk_embeddings
    ADD COLUMN model_version TEXT NOT NULL DEFAULT 'legacy-unversioned',
    ADD COLUMN dimensions INT,
    ADD COLUMN representation_version TEXT NOT NULL DEFAULT 'legacy-unversioned';
UPDATE chunk_embeddings SET dimensions = vector_dims(embedding);
ALTER TABLE chunk_embeddings
    ALTER COLUMN embedding SET NOT NULL,
    ALTER COLUMN dimensions SET NOT NULL,
    ALTER COLUMN model_version DROP DEFAULT,
    ALTER COLUMN representation_version DROP DEFAULT,
    ADD CHECK (dimensions BETWEEN 1 AND 16000 AND dimensions = vector_dims(embedding)),
    ADD CHECK (length(btrim(model_version)) > 0 AND length(btrim(model_name)) > 0
               AND length(btrim(representation_version)) > 0),
    ADD UNIQUE (content_hash, model_name, model_version, dimensions, representation_version);
