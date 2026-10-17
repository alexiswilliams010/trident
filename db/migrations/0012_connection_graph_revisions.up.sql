CREATE TABLE connection_graph_revisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL REFERENCES connection_snapshots(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    graph JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(snapshot_id, fingerprint),
    UNIQUE(snapshot_id, id)
);
CREATE TABLE connection_active_graphs (
    snapshot_id UUID PRIMARY KEY REFERENCES connection_snapshots(id) ON DELETE CASCADE,
    revision_id UUID NOT NULL,
    FOREIGN KEY(snapshot_id, revision_id)
        REFERENCES connection_graph_revisions(snapshot_id, id) ON DELETE CASCADE
);
CREATE FUNCTION reject_graph_revision_update() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Graph revisions are immutable; publish a new revision' USING ERRCODE='23514';
END;
$$;
CREATE TRIGGER immutable_connection_graph_revision BEFORE UPDATE ON connection_graph_revisions
FOR EACH ROW EXECUTE FUNCTION reject_graph_revision_update();
