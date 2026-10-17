CREATE TABLE connection_site_indexes (
    snapshot_id UUID NOT NULL REFERENCES connection_snapshots(id) ON DELETE CASCADE,
    index_version TEXT NOT NULL,
    representation_version TEXT NOT NULL,
    stats JSONB NOT NULL DEFAULT '{}',
    completed_at TIMESTAMPTZ,
    PRIMARY KEY (snapshot_id, index_version)
);

ALTER TABLE interaction_sites
    ADD COLUMN index_version TEXT,
    ADD COLUMN retrieval_hash TEXT,
    ADD COLUMN terms TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN fts_doc TSVECTOR,
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}',
    ADD FOREIGN KEY (snapshot_id, index_version)
        REFERENCES connection_site_indexes(snapshot_id, index_version) ON DELETE CASCADE,
    ADD CHECK (index_version IS NULL OR
        (retrieval_text IS NOT NULL AND retrieval_hash IS NOT NULL
         AND representation_version IS NOT NULL
         AND retrieval_hash=encode(sha256(convert_to(retrieval_text, 'UTF8')), 'hex')));
CREATE INDEX interaction_sites_index ON interaction_sites(snapshot_id, index_version, kind);
CREATE INDEX interaction_sites_terms ON interaction_sites USING GIN(terms);
CREATE INDEX interaction_sites_fts ON interaction_sites USING GIN(fts_doc);
CREATE INDEX interaction_sites_input ON interaction_sites(retrieval_hash);

CREATE TABLE interaction_embeddings (
    input_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimensions INT NOT NULL CHECK (dimensions BETWEEN 1 AND 16000),
    representation_version TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    CHECK (vector_dims(embedding)=dimensions),
    PRIMARY KEY (input_hash, model_name, model_version, dimensions, representation_version)
);

CREATE TABLE connection_retrieval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    index_version TEXT NOT NULL,
    run_key TEXT NOT NULL UNIQUE,
    corpus_fingerprint TEXT NOT NULL,
    embedding_identity JSONB NOT NULL,
    settings JSONB NOT NULL,
    completed_at TIMESTAMPTZ,
    UNIQUE(snapshot_id, id),
    FOREIGN KEY(snapshot_id, index_version)
        REFERENCES connection_site_indexes(snapshot_id, index_version) ON DELETE CASCADE
);

-- Keep rankings separate from candidate identity: another retrieval run must
-- not overwrite the evidence trail of an earlier decision on the same pair.
CREATE TABLE connection_candidate_sets (
    snapshot_id UUID NOT NULL,
    run_id UUID NOT NULL,
    source_site_id UUID NOT NULL,
    candidate_set_fingerprint TEXT NOT NULL,
    candidate_count INT NOT NULL CHECK (candidate_count >= 0),
    PRIMARY KEY(run_id, source_site_id),
    FOREIGN KEY(snapshot_id, run_id)
        REFERENCES connection_retrieval_runs(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id, source_site_id)
        REFERENCES interaction_sites(snapshot_id, id) ON DELETE CASCADE
);
CREATE TABLE connection_candidate_matches (
    snapshot_id UUID NOT NULL,
    run_id UUID NOT NULL,
    source_site_id UUID NOT NULL,
    candidate_id UUID NOT NULL,
    rank INT NOT NULL CHECK (rank > 0),
    fused_score DOUBLE PRECISION NOT NULL,
    channels JSONB NOT NULL,
    PRIMARY KEY(run_id, candidate_id),
    UNIQUE(run_id, source_site_id, rank),
    FOREIGN KEY(run_id, source_site_id)
        REFERENCES connection_candidate_sets(run_id, source_site_id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id, candidate_id)
        REFERENCES connection_candidates(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id, run_id)
        REFERENCES connection_retrieval_runs(snapshot_id, id) ON DELETE CASCADE
);

CREATE FUNCTION check_connection_match_source() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM connection_candidates
                   WHERE id=NEW.candidate_id AND source_site_id=NEW.source_site_id) THEN
        RAISE EXCEPTION 'Retrieval match source differs from candidate source' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER connection_match_source BEFORE INSERT OR UPDATE ON connection_candidate_matches
FOR EACH ROW EXECUTE FUNCTION check_connection_match_source();
