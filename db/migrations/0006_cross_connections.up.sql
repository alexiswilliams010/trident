-- Additive contracts only: indexing and single-repo query behavior are unchanged.
-- Snapshot capture/sealing and graph revision publication are not implemented here.
CREATE TABLE connection_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_fingerprint TEXT NOT NULL CHECK (length(context_fingerprint) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE connection_snapshot_members (
    snapshot_id UUID NOT NULL REFERENCES connection_snapshots(id) ON DELETE CASCADE,
    repo_id BIGINT NOT NULL REFERENCES repos(id),
    revision TEXT NOT NULL CHECK (length(btrim(revision)) > 0),
    PRIMARY KEY (snapshot_id, repo_id)
);

-- Source coordinates/content identity survive parser node replacement. Parser IDs
-- are optional navigation hints, not the historical evidence identity.
CREATE TABLE interaction_sites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    repo_id BIGINT NOT NULL,
    path TEXT NOT NULL CHECK (length(path) > 0),
    start_byte INT NOT NULL CHECK (start_byte >= 0),
    end_byte INT NOT NULL CHECK (end_byte >= start_byte),
    content_hash TEXT NOT NULL CHECK (length(content_hash) > 0),
    kind TEXT NOT NULL CHECK (kind IN ('call', 'definition', 'declaration', 'configuration')),
    node_id BIGINT REFERENCES nodes(id) ON DELETE SET NULL,
    definition_id BIGINT REFERENCES definitions(id) ON DELETE SET NULL,
    retrieval_text TEXT,
    representation_version TEXT,
    UNIQUE (snapshot_id, id),
    UNIQUE (snapshot_id, id, repo_id),
    FOREIGN KEY (snapshot_id, repo_id)
        REFERENCES connection_snapshot_members(snapshot_id, repo_id) ON DELETE CASCADE
);
CREATE INDEX interaction_sites_repo ON interaction_sites(snapshot_id, repo_id);
CREATE INDEX interaction_sites_node ON interaction_sites(node_id);
CREATE INDEX interaction_sites_definition ON interaction_sites(definition_id);

CREATE TABLE connection_resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL REFERENCES connection_snapshots(id) ON DELETE CASCADE,
    identity_key TEXT NOT NULL CHECK (length(identity_key) > 0),
    kind TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    UNIQUE (snapshot_id, id),
    UNIQUE (snapshot_id, identity_key)
);

CREATE TABLE connection_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    source_site_id UUID NOT NULL,
    source_repo_id BIGINT NOT NULL,
    target_site_id UUID NOT NULL,
    target_repo_id BIGINT NOT NULL,
    retrieval_rank INT CHECK (retrieval_rank > 0),
    reranker_score DOUBLE PRECISION,
    retrieval_metadata JSONB NOT NULL DEFAULT '{}',
    CHECK (source_repo_id <> target_repo_id),
    UNIQUE (snapshot_id, id),
    UNIQUE (snapshot_id, source_site_id, target_site_id),
    FOREIGN KEY (snapshot_id, source_site_id, source_repo_id)
        REFERENCES interaction_sites(snapshot_id, id, repo_id) ON DELETE CASCADE,
    FOREIGN KEY (snapshot_id, target_site_id, target_repo_id)
        REFERENCES interaction_sites(snapshot_id, id, repo_id) ON DELETE CASCADE
);
CREATE INDEX connection_candidates_target ON connection_candidates(snapshot_id, target_site_id);

CREATE TABLE connection_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    repo_id BIGINT NOT NULL,
    path TEXT NOT NULL CHECK (length(path) > 0),
    start_byte INT NOT NULL CHECK (start_byte >= 0),
    end_byte INT NOT NULL CHECK (end_byte >= start_byte),
    content_hash TEXT NOT NULL CHECK (length(content_hash) > 0),
    excerpt TEXT NOT NULL,
    UNIQUE (snapshot_id, id),
    FOREIGN KEY (snapshot_id, repo_id)
        REFERENCES connection_snapshot_members(snapshot_id, repo_id) ON DELETE CASCADE
);

CREATE TABLE connection_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    candidate_id UUID NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN
        ('supported', 'unsupported', 'insufficient_evidence', 'ambiguous')),
    provider TEXT NOT NULL CHECK (length(provider) > 0),
    model_version TEXT NOT NULL CHECK (length(model_version) > 0),
    policy_version TEXT NOT NULL CHECK (length(policy_version) > 0),
    evidence_fingerprint TEXT NOT NULL CHECK (length(evidence_fingerprint) > 0),
    score DOUBLE PRECISION CHECK (score >= 0 AND score <= 1),
    conditions JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (snapshot_id, id),
    FOREIGN KEY (snapshot_id, candidate_id)
        REFERENCES connection_candidates(snapshot_id, id) ON DELETE CASCADE
);
CREATE INDEX connection_decisions_candidate ON connection_decisions(snapshot_id, candidate_id);

CREATE TABLE connection_decision_evidence (
    snapshot_id UUID NOT NULL,
    decision_id UUID NOT NULL,
    evidence_id UUID NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('support', 'contradiction', 'context')),
    PRIMARY KEY (decision_id, evidence_id, role),
    FOREIGN KEY (snapshot_id, decision_id)
        REFERENCES connection_decisions(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY (snapshot_id, evidence_id)
        REFERENCES connection_evidence(snapshot_id, id) ON DELETE CASCADE
);
CREATE INDEX connection_decision_evidence_source
    ON connection_decision_evidence(snapshot_id, evidence_id);

CREATE TABLE connection_verbs (
    version INT NOT NULL CHECK (version > 0),
    name TEXT NOT NULL,
    definition TEXT NOT NULL,
    allows_site_target BOOLEAN NOT NULL,
    allows_resource_target BOOLEAN NOT NULL,
    PRIMARY KEY (version, name),
    CHECK (allows_site_target OR allows_resource_target)
);
INSERT INTO connection_verbs VALUES
    (1, 'INVOKES', 'Requests execution of an operation.', true, false),
    (1, 'DISPATCHES_TO', 'Routes an operation to another execution target.', true, false),
    (1, 'SPAWNS', 'Starts a process or executable.', true, true),
    (1, 'PRODUCES', 'Publishes or creates an event or message for consumption.', false, true),
    (1, 'CONSUMES', 'Subscribes to or processes events or messages from a resource.', false, true),
    (1, 'READS', 'Retrieves data from a resource.', false, true),
    (1, 'WRITES', 'Modifies data in a resource.', false, true),
    (1, 'PROVISIONS', 'Creates or manages an infrastructure resource.', false, true),
    (1, 'CONFIGURES', 'Supplies settings controlling a component or resource.', true, true);

-- Resource edges point from actor to resource for both PRODUCES and CONSUMES.
-- Two legs under one decision preserve fan-out without inventing a direct call.
CREATE TABLE cross_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL,
    decision_id UUID NOT NULL,
    source_site_id UUID NOT NULL,
    target_site_id UUID,
    target_resource_id UUID,
    verb_version INT NOT NULL,
    verb TEXT NOT NULL,
    CHECK (num_nonnulls(target_site_id, target_resource_id) = 1),
    CHECK (source_site_id <> target_site_id),
    UNIQUE NULLS NOT DISTINCT
        (decision_id, source_site_id, target_site_id, target_resource_id, verb_version, verb),
    FOREIGN KEY (snapshot_id, decision_id)
        REFERENCES connection_decisions(snapshot_id, id) ON DELETE CASCADE,
    FOREIGN KEY (snapshot_id, source_site_id)
        REFERENCES interaction_sites(snapshot_id, id),
    FOREIGN KEY (snapshot_id, target_site_id)
        REFERENCES interaction_sites(snapshot_id, id),
    FOREIGN KEY (snapshot_id, target_resource_id)
        REFERENCES connection_resources(snapshot_id, id),
    FOREIGN KEY (verb_version, verb) REFERENCES connection_verbs(version, name)
);
CREATE INDEX cross_connections_source ON cross_connections(snapshot_id, source_site_id);
CREATE INDEX cross_connections_target ON cross_connections(snapshot_id, target_site_id);
CREATE INDEX cross_connections_resource ON cross_connections(snapshot_id, target_resource_id);

-- Deferred validation permits both resource legs and their evidence to be written
-- in one transaction, while preventing half a resource connection at commit.
CREATE FUNCTION validate_connection_decision(decision UUID) RETURNS VOID
LANGUAGE plpgsql AS $$
DECLARE
    d connection_decisions%ROWTYPE;
    c connection_candidates%ROWTYPE;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM cross_connections WHERE decision_id = decision) THEN
        RETURN;
    END IF;
    SELECT * INTO STRICT d FROM connection_decisions WHERE id = decision;
    SELECT * INTO STRICT c FROM connection_candidates WHERE id = d.candidate_id;
    IF d.outcome <> 'supported' OR NOT EXISTS (
        SELECT 1 FROM connection_decision_evidence
        WHERE decision_id = decision AND role = 'support'
    ) THEN
        RAISE EXCEPTION 'Connections require a supported decision and supporting evidence'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1 FROM cross_connections e
        JOIN connection_verbs v ON (v.version, v.name) = (e.verb_version, e.verb)
        WHERE e.decision_id = decision AND (
            e.source_site_id NOT IN (c.source_site_id, c.target_site_id)
            OR (e.target_site_id IS NOT NULL AND
                (NOT v.allows_site_target OR
                 e.target_site_id NOT IN (c.source_site_id, c.target_site_id)))
            OR (e.target_resource_id IS NOT NULL AND NOT v.allows_resource_target)
        )
    ) THEN
        RAISE EXCEPTION 'Connection endpoints or verb do not match the candidate'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT target_resource_id FROM cross_connections
        WHERE decision_id = decision AND target_resource_id IS NOT NULL
        GROUP BY target_resource_id HAVING count(DISTINCT source_site_id) <> 2
    ) THEN
        RAISE EXCEPTION 'Shared resource requires participation from both repositories'
            USING ERRCODE = '23514';
    END IF;
END;
$$;

CREATE FUNCTION check_connection_publication() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE related UUID;
BEGIN
    IF TG_TABLE_NAME = 'connection_decisions' THEN
        PERFORM validate_connection_decision(NEW.id);
    ELSIF TG_TABLE_NAME = 'connection_candidates' THEN
        FOR related IN SELECT id FROM connection_decisions WHERE candidate_id = NEW.id LOOP
            PERFORM validate_connection_decision(related);
        END LOOP;
    ELSE
        IF TG_OP <> 'INSERT' THEN
            PERFORM validate_connection_decision(OLD.decision_id);
        END IF;
        IF TG_OP <> 'DELETE' THEN
            PERFORM validate_connection_decision(NEW.decision_id);
        END IF;
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER cross_connections_valid
AFTER INSERT OR UPDATE OR DELETE ON cross_connections
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_connection_publication();
CREATE CONSTRAINT TRIGGER connection_evidence_valid
AFTER INSERT OR UPDATE OR DELETE ON connection_decision_evidence
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_connection_publication();
CREATE CONSTRAINT TRIGGER connection_decision_valid
AFTER UPDATE ON connection_decisions
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_connection_publication();
CREATE CONSTRAINT TRIGGER connection_candidate_valid
AFTER UPDATE ON connection_candidates
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_connection_publication();
