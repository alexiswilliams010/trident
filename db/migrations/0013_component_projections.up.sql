CREATE TABLE connection_component_projections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    graph_revision_id UUID NOT NULL,
    fingerprint TEXT NOT NULL,
    projection JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(graph_revision_id, fingerprint),
    FOREIGN KEY(snapshot_id, graph_revision_id)
        REFERENCES connection_graph_revisions(snapshot_id, id) ON DELETE CASCADE
);
CREATE TRIGGER immutable_component_projection BEFORE UPDATE ON connection_component_projections
FOR EACH ROW EXECUTE FUNCTION reject_graph_revision_update();
