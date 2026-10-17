# Cross-repository connections: Phase 1 contracts

Status: schema and Python contracts only. No automatic processing, model calls,
CLI command, component projection, or changes to existing parser queries.

## Eligibility and scope

A connection analysis requires at least two distinct canonical `repos.id` values.
Select one exact revision per repository per snapshot. Two branches of one repo
do not qualify. Snapshot context fingerprints distinguish configuration contexts.
Call `Snapshot.require_cross_repo()` before any future retrieval or model work.

Every candidate pairs sites belonging to different repositories in the same
snapshot. Composite foreign keys prevent a caller from claiming that a site
belongs to another repository or snapshot. Consequently a candidate cannot be
stored under a snapshot containing only one repository. A snapshot record itself
may be staged before its members are populated; its existence is not permission
to run analysis.

Existing parser relationships remain authoritative inputs. This subsystem does
not generate CONTAINS, IMPORTS, IMPLEMENTS, or SIMILAR_TO relationships. Similarity
is retrieval metadata, not a traversable connection.

## Records and ownership

| Record | Role |
| --- | --- |
| connection_snapshots | Analysis identity and configuration fingerprint |
| connection_snapshot_members | Canonical repository and exact revision |
| interaction_sites | Calls and candidate definitions/declarations/configuration sites |
| connection_resources | Snapshot-scoped shared resources |
| connection_candidates | Cross-repository site pairs and retrieval metadata |
| connection_evidence | Source coordinates, content hash, and retained excerpt |
| connection_decisions | Outcome, model/policy provenance, evidence fingerprint, conditions |
| connection_decision_evidence | Evidence association and support/contradiction/context role |
| connection_verbs | Versioned definitions and allowed endpoint kinds |
| cross_connections | Directed edges supported by a decision |

Parser node/definition references on sites are optional navigation hints and can
be cleared when parser objects are removed. They are not historical evidence.
Source paths, byte offsets (end-exclusive), content hashes, retained excerpts,
repository revisions, and configuration context identify the evidence.

Phase 2 will implement snapshot capture/sealing, content verification, dependency
fingerprints, and incremental invalidation. Non-empty revision strings are accepted
by these contracts; resolving them to exact Git objects is the capture stage's job.
The schema alone does not freeze a mutable branch or verify an excerpt against Git.

## Directed vocabulary, version 1

All sources are code/configuration sites. Both sides of event/data interactions
point **toward** the resource: `A --PRODUCES--> topic <--CONSUMES-- B`.

| Verb | Target | Meaning |
| --- | --- | --- |
| INVOKES | Site | Requests execution of an operation |
| DISPATCHES_TO | Site | Routes an operation to another execution target |
| SPAWNS | Site or resource | Starts a process or executable |
| PRODUCES | Resource | Publishes or creates an event/message for consumption |
| CONSUMES | Resource | Subscribes to or processes events/messages from a resource |
| READS | Resource | Retrieves data |
| WRITES | Resource | Modifies data |
| PROVISIONS | Resource | Creates or manages infrastructure |
| CONFIGURES | Site or resource | Supplies controlling settings |

`normalize_verb` normalizes case/whitespace and maps EMITS to PRODUCES. Unknown
verbs fail validation; no free-form labels are silently added. Future vocabulary
changes must add a version rather than reinterpret existing edges. Tests compare
the Python registry to the migration seed, including definitions and endpoint kinds.

## Decisions and publication units

Outcomes are supported, unsupported, insufficient_evidence, and ambiguous. Scores
are optional finite values in [0, 1]; they are neither calibrated guarantees nor
an automatic acceptance threshold. Conditions live on the decision and apply to
all its edges. Different conditions require separate decisions.

Candidate direction records discovery order, not necessarily behavior direction.
A direct edge can run either way between the candidate's two sites. Resource edges
use either candidate site as the source. Each referenced resource must have legs
from **both** candidate sites under the same decision. Resources do not supply a
repository identity and cannot make an otherwise local relationship eligible.

For example, one supported candidate between Billing and Reporting yields:

```
Billing site   --PRODUCES--> invoice.created
Reporting site --CONSUMES--> invoice.created
```

It does not yield a Billing INVOKES Reporting edge. A third consumer uses another
candidate/decision and may reference the same resource. Resource identity keys
must incorporate known environment/namespace distinctions; a matching name alone
is not proof that two resources are identical. Identity resolution comes later.

Persist the decision, its evidence associations, and all connection legs in one
transaction. Deferred constraint triggers validate the final transaction state:

- The decision is supported and has a supporting evidence association.
- Edge endpoints belong to its candidate and snapshot.
- The verb allows the selected endpoint kinds.
- Every resource has participation from both repositories.

The same validation runs when edges/evidence are removed or a decision/candidate
is changed. It prevents leaving one resource leg behind or changing a connected
decision to ambiguous. Repeated identical edges under one decision are rejected.

The Python `validate_connections` helper validates a complete publication unit;
pass evidence and resource IDs loaded from the same snapshot. SQL foreign keys
remain the authoritative check on persisted resource membership. A supported SQL
decision may be staged without edges, but cannot publish any edge without support.

## Remaining phases

2. Reliable snapshots, embedding identities, and invalidation.
3. All-call-site input extraction and cross-repository candidate retrieval.
4. Reranking and bounded evidence assembly.
5. Jev integration, support assessment, and verb selection/normalization.
6. Atomic graph revisions, CLI, and combined graph queries.
7. Component membership and projection with drill-down evidence.
8. Neptune orchestration, caching, and held-out-family evaluation.

Component tables are intentionally deferred to Phase 7. They will project these
connections, not ask a second model to independently infer component edges.

## Validation

Pure contracts: `python -m pytest tests/test_connection_contracts.py`.
Database contracts: set DATABASE_URL to a disposable PostgreSQL 18 database with
pgvector available, then run `python -m pytest tests/test_connection_schema.py`.
The existing pg_pool fixture applies migrations to that database; never point this
test command at a production database. Schema tests verify migration tracking,
registry parity, eligibility, evidence requirements, and atomic resource pairs.
