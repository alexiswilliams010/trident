# Cross-repository connections

Status: contracts, committed-source capture, decision reuse tracking, call-site
extraction, cross-repository retrieval, reranking, and bounded evidence assembly
are implemented as Python APIs. Embedding and reranking use explicitly supplied
callables. Decision model calls, connection CLI, and component projection are
not implemented yet.

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
| connection_snapshot_files | Sealed paths, file modes, Git object IDs, and content hashes |
| connection_source_blobs | Hash-verified immutable bytes, shared across snapshots |
| interaction_sites | Calls and candidate definitions/declarations/configuration sites |
| connection_site_indexes | Versioned extraction settings and completion statistics |
| interaction_embeddings | Deduplicated retrieval inputs in explicitly versioned vector spaces |
| connection_retrieval_runs | Corpus identity, retrieval settings, and completed runs |
| connection_candidate_sets | Per-call candidate fingerprints, including empty results |
| connection_candidate_matches | Run-specific ranks and channel provenance |
| connection_preparation_runs | Reranker identity, preparation settings, and completion |
| connection_evidence_packets | Per-source alternatives and bounded cited context |
| connection_packet_evidence | Snapshot-checked links to stored excerpts |
| connection_resources | Snapshot-scoped shared resources |
| connection_candidates | Cross-repository site pairs and retrieval metadata |
| connection_evidence | Source coordinates, content hash, and retained excerpt |
| connection_decisions | Outcome, model/policy provenance, evidence fingerprint, conditions |
| connection_decision_evidence | Evidence association and support/contradiction/context role |
| connection_decision_inputs | Versioned input and decision fingerprints |
| connection_decision_dependencies | Endpoint/evidence file dependencies, including missing files |
| connection_verbs | Versioned definitions and allowed endpoint kinds |
| cross_connections | Directed edges supported by a decision |

Parser node/definition references on sites are optional navigation hints and can
be cleared when parser objects are removed. They are not historical evidence.
Source paths, byte offsets (end-exclusive), content hashes, retained excerpts,
repository revisions, and configuration context identify the evidence.

## Snapshot capture and source verification

`capture_snapshot(pool, repositories, context=...)` resolves each requested Git
revision to a commit and reads its tree and blobs directly. It captures all tracked
files, including configuration/build files and binaries, independently of the
language parser. Dirty or untracked working-tree files are excluded. Symlinks are
captured as link text without following them; submodules retain their pinned Git
commit as a gitlink and are not recursively ingested. Register a submodule as a
separate repository to analyze its source. Non-UTF-8 Git filenames are rejected.

Repository IDs must match their registered checkout roots. Repeated IDs, repeated
checkouts, and worktrees sharing a Git common directory do not qualify as separate
repositories. Separate clones must still be assigned their canonical repository
identity by the caller; local Git directories cannot establish remote identity.

The capture key includes sorted repository/commit pairs and a canonical hash of
the supplied configuration context. The context itself is not stored verbatim.
Repeated/concurrent captures converge on one sealed snapshot. Manifest and context
edits are forbidden after sealing, while deleting the entire snapshot is supported.
Source blobs remain deduplicated; blob garbage collection is not implemented yet.
Capture currently buffers Git blob output in memory, so very large repositories
will need a streaming implementation before scaling this path.

`read_source` reads captured bytes rather than the current checkout. Sites inserted
into sealed snapshots must match a captured file's hash and byte bounds. Evidence
must also match the exact UTF-8 excerpt at the end-exclusive byte range. Binary
files can be captured/read but cannot serve as arbitrary text excerpts. Parser IDs
remain optional; future call-site extraction must match captured content before
attaching parser objects. Sealing happens before adding sites or evidence.

The existing working-tree indexer separately verifies file bytes on every walk,
including same-size edits. Unchanged hashes still reuse parse results. `--rehash`
remains accepted for compatibility but no longer changes this behavior. This costs
file-read I/O on each index run in exchange for reliable freshness.

## Decision reuse and invalidation

`record_decision_inputs` requires a sealed snapshot. It records dependencies on
both endpoint files and every linked evidence file automatically. Callers must
also supply files inspected during investigation, including paths that were absent
but whose later appearance could affect the answer. File mode changes count as
changes even when bytes are identical.

`ResolutionVersions` records provider/model, policy, vocabulary, representation,
retrieval, reranker, and candidate-set identities. The candidate-set fingerprint
must cover the actual alternatives and relevant context, not merely their names.
Retrieval must be refreshed when its corpus changes: file dependencies alone cannot
detect a new alternative that the original decision never inspected.

`assess_decision_reuse` compares tracked inputs with a new sealed snapshot and
returns explicit invalidation reasons. Unchanged inputs may be reused across
commits; changed endpoints, evidence, missing-file presence, repository membership,
configuration context, or processing versions block reuse. Edits to the original
decision/evidence are detected as well. Re-recording different inputs on the same
decision is rejected.

Historical decisions and connections are preserved. A reusable assessment does
not copy database IDs into a new snapshot or publish edges automatically. Future
decision processing must rebind validated source references and write a new
snapshot-specific decision. Component projections will consume those decisions.

## Embedding identity and migration

Embeddings are unique by `(content_hash, model_name, model_version, dimensions,
representation_version)`. Both indexing and semantic/hybrid queries select that
same identity. The vector column accepts different dimensions, with a constraint
ensuring each row matches its declared dimensions. No ANN index is added here.

Migration 0007 retains previous rows as `legacy-unversioned` for model and input
versions. They are not automatically reused by current indexing or queries.
Run the normal embedding command after upgrading to populate the current identity;
semantic search can return no results until those embeddings exist. The migration
does not call an embedding service or spend API credits.

`OpenAICompatibleEmbedder` and the fake embedder expose their identities. Custom
query callables must pass `embedding_identity=EmbeddingIdentity(...)`; querying
an undeclared vector space fails before the embedding call. Configure dimensions
with `EMBEDDING_DIM` (default 4096), `EMBEDDING_MODEL_VERSION` (defaults to the model
name), and `EMBEDDING_REPRESENTATION_VERSION` (default `chunk-v1`). For moving model
aliases, explicitly pin/update the model version when the provider changes weights;
the application cannot detect a provider's silent alias change.

## Call inventory and candidate retrieval

After capturing a sealed snapshot, use the APIs in this order:

```python
from core.connections.sites import index_snapshot_sites
from core.connections.retrieval import embed_snapshot_sites, retrieve_all_candidates

index = await index_snapshot_sites(pool, snapshot.id)
await embed_snapshot_sites(pool, snapshot.id, index.index_version, embedder.embed)
run = await retrieve_all_candidates(
    pool, snapshot.id, index.index_version, embedder.embedding_identity,
)
```

The supplied embedder must declare an `EmbeddingIdentity` with representation
version `site-v1`, or an undeclared callable must receive an explicit
`embedding_identity=` argument. Configure a dedicated embedder for this input;
the normal chunk embedder's `chunk-v1` identity deliberately does not qualify.
No embedding API is called by extraction, migrations, or retrieval itself.

Extraction reparses captured bytes with the existing language call/definition
rules. It inventories every configured call shape, including locally resolved
calls, without protocol handlers or a confidence cutoff. Supported languages are
Python, Solidity, Go, Rust, JavaScript, and TypeScript. This does not promise to
discover calls the grammars/rules cannot identify, dynamically constructed calls,
or call sites in unsupported languages. Generic overlapping text passages make
configuration, build files, unsupported source, and declarations searchable as
targets. Binary/non-UTF-8 files, empty files, symlinks, and gitlinks are reported
as skipped. Parser relationships are neither regenerated as connection edges nor
required to retrieve candidates.

Inputs are plain source expressions plus bounded enclosing context and module
paths. Source coordinates retain full node spans even when the input is clipped.
Index identity includes extraction settings, language rules, and grammar versions.
Extraction publishes atomically and repeated calls reuse a completed index.
Embedding inputs deduplicate by text hash and complete vector-space identity;
each original call occurrence keeps its own site and candidate set. Failed
embedding batches can resume from already cached batches. Zero, nonfinite, or
wrong-dimension vectors are rejected.

Each call searches only other repositories within its snapshot and index using
cosine similarity, PostgreSQL full-text search, and exact shared tokens such as
identifiers, paths, and event names. Reciprocal-rank fusion combines these channels;
a configurable exact-match reserve protects some candidates from semantic
ranking. Defaults are 50 results per channel, 30 candidates per call, and up to 8
reserved exact matches. There is no semantic similarity threshold. Overlapping
generic passages collapse within the retrieved pool; calls and definitions retain
distinct occurrences. Bounds and token matching are retrieval heuristics, not
proof of behavioral correspondence or guaranteed recall.

`retrieve_candidates(...)` inspects one call without persisting a run.
`retrieve_all_candidates(...)` stores every call's set, even when empty, plus ranks,
channel scores, and fingerprints. Changing retrieval settings creates a separate
run without overwriting earlier rankings. Repeated identical runs reuse their
records. It creates no decisions, verbs, or graph edges. Empty results do not prove
that no cross-repository behavior exists; preparation follows, then decision-model
assessment. Direct SQL mutation of completed indexes/runs is not a supported update path;
change inputs/settings and create a new version instead.

This initial implementation uses exact vector scans and serial per-input retrieval,
with repeated inputs cached within a run. It has no ANN index or large-corpus
performance claim. Tests use controlled vectors to verify channel behavior and
fixture recall, not to measure production embedding quality.

## Reranking and evidence packets

```python
from core.connections.preparation import RerankerIdentity, prepare_candidates

prepared = await prepare_candidates(
    pool, run.id, rerank_fn,
    RerankerIdentity(provider="your-provider", model_version="pinned-model-version"),
    additional_paths=((repo_id, "config/routing.yaml"),),
)
```

`rerank_fn(query, documents)` is an async callable returning one finite numeric
score per document **in input order**. Higher scores rank first; ties preserve
retrieval order. Provider adapters that return sorted results must restore input
order. No provider endpoint or SDK is assumed, and no default heuristic silently
substitutes for a reranker. The caller supplies the adapter and explicitly pins
its provider/model/input versions. Reranking is relative relevance, not calibrated
confidence or acceptance of a relationship.

All retrieved alternatives are scored. Each packet retains candidate IDs, original
ranks/channel provenance, scores, selection status, and input clipping flags.
Defaults select five targets and cap each query/document at 4,000 characters.
Empty candidate sets produce explicit empty packets without invoking the scorer.
Changing identity, options, or requested context creates a new preparation run;
repeating identical inputs reuses completed packets without external calls.

Evidence comes only from captured source blobs. Source and selected targets each
receive a reserved excerpt budget, prioritizing the endpoint over surrounding
context. Default limits are 3,000 UTF-8 bytes per excerpt, 500 surrounding bytes,
24,000 excerpt bytes per packet, and 12 excerpts. Limits concern excerpt content,
not JSON metadata or model tokens. Large sites retain their original coordinates
in `interaction_sites`; packet citations record the actual aligned excerpt range
and whether site/context was clipped. Text is source material for later assessment,
not an instruction to a model and not pre-labelled support or contradiction.

Up to 32 explicit additional paths can supply configuration or other context from
any snapshot member. No protocol-specific path discovery or evidence-expansion
agent is introduced. Dependencies retain path, mode, and content hash; missing,
binary, non-UTF-8, symlink/gitlink, and budget-omitted context is reported explicitly.
The later decision stage must carry packet dependencies, including missing paths,
into decision-input tracking. Bounded context may be insufficient; that must remain
a valid outcome for the decision model.

Packets live in `connection_evidence_packets.payload`, with a fingerprint and
normalized links to `connection_evidence`. Preparation writes no decisions or
graph edges, and leaves prior retrieval rankings unchanged. Publication is atomic
for the entire run: malformed scores, exceptions, and scorer timeouts roll back
all its packets. The default timeout is 60 seconds per source. This initial API
serializes preparation under the snapshot lock and holds its database transaction
during scorer calls; it has no parallel worker or partial-run resume mechanism.

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
Snapshot/invalidation coverage is in `tests/test_snapshot_dependencies.py`; vector
identity and legacy migration coverage is in `tests/test_embedding_identity.py`.
Call extraction, vector reuse, channel fusion, and snapshot-scoped retrieval are
covered by `tests/test_connection_retrieval.py`.

Reranker validation, packet budgets, captured excerpts, rollback, and reuse are
covered by `tests/test_connection_preparation.py`. Tests use controlled scorers;
provider integration and model-quality evaluation remain separate work.
