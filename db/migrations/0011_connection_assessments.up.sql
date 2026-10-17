CREATE TABLE connection_assessments (
    decision_id UUID PRIMARY KEY,
    snapshot_id UUID NOT NULL,
    packet_id UUID NOT NULL,
    assessment_key TEXT NOT NULL UNIQUE,
    request JSONB NOT NULL,
    response JSONB NOT NULL,
    proposal JSONB NOT NULL,
    FOREIGN KEY(snapshot_id, decision_id)
        REFERENCES connection_decisions(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id, packet_id)
        REFERENCES connection_evidence_packets(snapshot_id, id) ON DELETE CASCADE
);
