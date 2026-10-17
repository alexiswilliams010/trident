CREATE TABLE connection_preparation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    retrieval_run_id UUID NOT NULL,
    run_key TEXT NOT NULL UNIQUE,
    reranker_identity JSONB NOT NULL,
    settings JSONB NOT NULL,
    completed_at TIMESTAMPTZ,
    UNIQUE(snapshot_id, id),
    FOREIGN KEY(snapshot_id, retrieval_run_id)
        REFERENCES connection_retrieval_runs(snapshot_id, id) ON DELETE CASCADE
);

CREATE TABLE connection_evidence_packets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    preparation_run_id UUID NOT NULL,
    source_site_id UUID NOT NULL,
    packet_fingerprint TEXT NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE(preparation_run_id, source_site_id),
    UNIQUE(snapshot_id, id),
    FOREIGN KEY(snapshot_id, preparation_run_id)
        REFERENCES connection_preparation_runs(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id, source_site_id)
        REFERENCES interaction_sites(snapshot_id, id) ON DELETE CASCADE
);

-- Packet excerpts are context; only the later decision stage can label support.
CREATE TABLE connection_packet_evidence (
    snapshot_id UUID NOT NULL,
    packet_id UUID NOT NULL,
    evidence_id UUID NOT NULL,
    PRIMARY KEY(packet_id, evidence_id),
    FOREIGN KEY(snapshot_id, packet_id)
        REFERENCES connection_evidence_packets(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id, evidence_id)
        REFERENCES connection_evidence(snapshot_id, id) ON DELETE CASCADE
);
