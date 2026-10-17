ALTER TABLE connection_snapshots
    ADD COLUMN snapshot_key TEXT UNIQUE,
    ADD COLUMN sealed_at TIMESTAMPTZ;

-- Sites/resources and decisions disappear along separate cascade paths when a
-- snapshot is deleted. Check these references after both paths have completed.
ALTER TABLE cross_connections
    ALTER CONSTRAINT cross_connections_snapshot_id_source_site_id_fkey
        DEFERRABLE INITIALLY DEFERRED,
    ALTER CONSTRAINT cross_connections_snapshot_id_target_site_id_fkey
        DEFERRABLE INITIALLY DEFERRED,
    ALTER CONSTRAINT cross_connections_snapshot_id_target_resource_id_fkey
        DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE connection_source_blobs (
    content_hash TEXT PRIMARY KEY,
    content BYTEA NOT NULL,
    CHECK (content_hash = encode(sha256(content), 'hex'))
);

CREATE TABLE connection_snapshot_files (
    snapshot_id UUID NOT NULL,
    repo_id BIGINT NOT NULL,
    path TEXT NOT NULL CHECK (length(path) > 0),
    mode TEXT NOT NULL CHECK (mode IN ('100644', '100755', '120000', '160000')),
    git_oid TEXT NOT NULL CHECK (git_oid ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'),
    content_hash TEXT NOT NULL REFERENCES connection_source_blobs(content_hash),
    PRIMARY KEY (snapshot_id, repo_id, path),
    FOREIGN KEY (snapshot_id, repo_id)
        REFERENCES connection_snapshot_members(snapshot_id, repo_id) ON DELETE CASCADE
);
CREATE INDEX connection_snapshot_files_content ON connection_snapshot_files(content_hash);

CREATE FUNCTION guard_connection_snapshot() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.sealed_at IS NOT NULL THEN
            RAISE EXCEPTION 'Insert a draft snapshot before sealing' USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
    END IF;
    IF OLD.sealed_at IS NOT NULL AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'Sealed snapshot metadata is immutable' USING ERRCODE = '23514';
    END IF;
    IF NEW.sealed_at IS NOT NULL AND OLD.sealed_at IS NULL THEN
        IF EXISTS (SELECT 1 FROM interaction_sites WHERE snapshot_id=NEW.id) OR
           EXISTS (SELECT 1 FROM connection_evidence WHERE snapshot_id=NEW.id) THEN
            RAISE EXCEPTION 'Seal source capture before adding sites or evidence'
                USING ERRCODE = '23514';
        END IF;
        IF NEW.snapshot_key IS NULL OR
           (SELECT count(*) FROM connection_snapshot_members WHERE snapshot_id = NEW.id) < 2 OR
           EXISTS (SELECT 1 FROM connection_snapshot_members WHERE snapshot_id = NEW.id
                   AND revision !~ '^([0-9a-f]{40}|[0-9a-f]{64})$') THEN
            RAISE EXCEPTION 'Sealing requires two distinct repositories at exact Git revisions'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER connection_snapshot_immutable BEFORE INSERT OR UPDATE ON connection_snapshots
FOR EACH ROW EXECUTE FUNCTION guard_connection_snapshot();

CREATE FUNCTION guard_connection_manifest() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE sid UUID;
BEGIN
    -- Lock the parent to serialize manifest edits with sealing. A missing parent
    -- during DELETE means the entire snapshot is being removed by cascade.
    FOR sid IN SELECT DISTINCT x FROM unnest(
        CASE WHEN TG_OP = 'INSERT' THEN ARRAY[NEW.snapshot_id]
             WHEN TG_OP = 'DELETE' THEN ARRAY[OLD.snapshot_id]
             ELSE ARRAY[OLD.snapshot_id, NEW.snapshot_id] END
    ) x ORDER BY x LOOP
        PERFORM 1 FROM connection_snapshots WHERE id = sid FOR UPDATE;
        IF EXISTS (SELECT 1 FROM connection_snapshots WHERE id = sid AND sealed_at IS NOT NULL) THEN
            RAISE EXCEPTION 'Sealed snapshot manifest is immutable' USING ERRCODE = '23514';
        END IF;
    END LOOP;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER connection_members_immutable
BEFORE INSERT OR UPDATE OR DELETE ON connection_snapshot_members
FOR EACH ROW EXECUTE FUNCTION guard_connection_manifest();
CREATE TRIGGER connection_files_immutable
BEFORE INSERT OR UPDATE OR DELETE ON connection_snapshot_files
FOR EACH ROW EXECUTE FUNCTION guard_connection_manifest();

-- Raw bytes are immutable and independently verified by their hash constraint.
CREATE FUNCTION reject_connection_blob_update() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Source blobs are immutable' USING ERRCODE = '23514';
END;
$$;
CREATE TRIGGER connection_blob_immutable BEFORE UPDATE ON connection_source_blobs
FOR EACH ROW EXECUTE FUNCTION reject_connection_blob_update();

CREATE FUNCTION verify_connection_source_span() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE raw BYTEA;
BEGIN
    -- Unsealed records remain staging records for low-level construction/import.
    PERFORM 1 FROM connection_snapshots WHERE id = NEW.snapshot_id FOR SHARE;
    IF EXISTS (SELECT 1 FROM connection_snapshots
               WHERE id = NEW.snapshot_id AND sealed_at IS NOT NULL) THEN
        SELECT b.content INTO raw FROM connection_snapshot_files f
        JOIN connection_source_blobs b USING (content_hash)
        WHERE f.snapshot_id = NEW.snapshot_id AND f.repo_id = NEW.repo_id
          AND f.path = NEW.path AND f.content_hash = NEW.content_hash AND f.mode <> '160000';
        IF raw IS NULL OR NEW.end_byte > octet_length(raw) THEN
            RAISE EXCEPTION 'Source span does not match captured content' USING ERRCODE = '23514';
        END IF;
        IF TG_TABLE_NAME = 'connection_evidence' THEN
            IF convert_to(NEW.excerpt, 'UTF8') <>
               substring(raw FROM NEW.start_byte + 1 FOR NEW.end_byte - NEW.start_byte) THEN
                RAISE EXCEPTION 'Evidence excerpt does not match captured bytes'
                    USING ERRCODE = '23514';
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER interaction_site_source_verified BEFORE INSERT OR UPDATE ON interaction_sites
FOR EACH ROW EXECUTE FUNCTION verify_connection_source_span();
CREATE TRIGGER connection_evidence_source_verified BEFORE INSERT OR UPDATE ON connection_evidence
FOR EACH ROW EXECUTE FUNCTION verify_connection_source_span();

-- Decision reuse is evaluated against a new immutable snapshot. Historical
-- decisions/edges are never deleted just because a later commit changed inputs.
CREATE TABLE connection_decision_inputs (
    decision_id UUID PRIMARY KEY REFERENCES connection_decisions(id) ON DELETE CASCADE,
    input_fingerprint TEXT NOT NULL,
    decision_fingerprint TEXT NOT NULL,
    versions JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE connection_decision_dependencies (
    decision_id UUID NOT NULL REFERENCES connection_decision_inputs(decision_id) ON DELETE CASCADE,
    repo_id BIGINT NOT NULL REFERENCES repos(id),
    path TEXT NOT NULL,
    -- NULL records absence, so a previously missing configuration file appearing
    -- in a later snapshot invalidates the decision too.
    content_hash TEXT,
    mode TEXT,
    PRIMARY KEY (decision_id, repo_id, path)
);
CREATE INDEX connection_dependencies_source ON connection_decision_dependencies(repo_id, path);
