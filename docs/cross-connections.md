# Cross-repository connections

Status: contracts, committed-source capture, decision reuse tracking, call-site
extraction, cross-repository retrieval, reranking, and bounded evidence assembly
are implemented as Python APIs. Embedding and reranking use explicitly supplied
callables. Jev assessment through Vercel AI Gateway is implemented. Graph
publication, pipeline CLI, and combined call-graph queries are implemented.
Deterministic component membership and evidence-backed projection are implemented.

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
| connection_assessments | Exact typed requests/answers, packet identity, and proposed verbs |
| connection_decision_evidence | Evidence association and support/contradiction/context role |
| connection_decision_inputs | Versioned input and decision fingerprints |
| connection_decision_dependencies | Endpoint/evidence file dependencies, including missing files |
| connection_verbs | Versioned definitions and allowed endpoint kinds |
| cross_connections | Directed edges supported by a decision; retained across revisions |
| connection_graph_revisions | Immutable graph payloads with captured evidence and optional parser links |
| connection_active_graphs | Atomic active-revision pointer for each snapshot |
| connection_component_projections | Immutable membership maps and aggregated graphs tied to graph revisions |

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
The decision stage carries packet dependencies, including missing paths,
into decision-input tracking. Bounded context may be insufficient; that must remain
a valid outcome for the decision model.

Packets live in `connection_evidence_packets.payload`, with a fingerprint and
normalized links to `connection_evidence`. Preparation writes no decisions or
graph edges, and leaves prior retrieval rankings unchanged. Publication is atomic
for the entire run: malformed scores, exceptions, and scorer timeouts roll back
all its packets. The default timeout is 60 seconds per source. This initial API
serializes preparation under the snapshot lock and holds its database transaction
during scorer calls; it has no parallel worker or partial-run resume mechanism.

## Jev assessment through Vercel

The adapter uses Vercel's documented [TypeSafe-compatible endpoint](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe)
with typed Choice questions, rather than chat completions. It defaults to
`https://ai-gateway.vercel.sh/typesafe/v1/systemone` and the pinned model
`typesafe-ai/jev-1.14.0`. Vercel documents that versioned model ID in its
[Jev integration guide](https://vercel.com/i/jev-integrations). Authentication uses
`AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN`. `JEV_MODEL` can select another pinned
model. Credentials are not included in stored requests or configuration reprs.
The HTTP client is separately configurable for compatible HTTPS endpoints.

```python
from core.connections.jev import JevClient, JevConfig
from core.connections.assessment import assess_packet

client = JevClient(JevConfig.from_env())
decision_ids = await assess_packet(pool, packet_id, client)
```

Obtain packet IDs from `connection_evidence_packets` for the preparation run.
Assessment processes selected candidates only; an empty packet makes no model
calls. Each candidate receives compact JSON evidence rendered as plaintext: exact
source and target code, separately labeled surrounding context, and any explicitly
requested additional context. Unrelated candidates and internal provenance IDs are
excluded from model input. Clipping is marked explicitly. The full assessment
policy appears once, followed by simplified questions without repeated policy text.
Source text is explicitly treated as data, never instructions. Jev chooses an outcome, direction/topology, established
or unresolved conditions, applicable verbs, and a role for each provided citation.
Each question includes its own criteria; question identifiers carry no implicit
meaning. No free-form graph JSON, generated citations, or protocol-specific
classification rules are required.

Supported decisions require supporting citations at both actual endpoint sites.
Unresolved conditions, missing endpoint support, missing verbs, or an unidentified
shared resource downgrade to insufficient evidence. Contradictory evidence causes
abstention as ambiguous. Unsupported, ambiguous, and insufficient-evidence outcomes
produce no proposed edges. The code checks citation membership, source identity,
site overlap, packet fingerprints, and candidate membership before inference.

Direct proposals carry an allowed site-target verb in either direction. Shared
resource proposals carry two site-to-resource verbs, including PRODUCES/CONSUMES,
and an evidence anchor selected from supplied excerpts. That anchor is **not** a
canonical resource ID or permission to merge similarly named resources. Publication materializes this as an evidence-scoped resource without globally
merging matching names. This initial policy handles
one interaction shape per candidate; unresolved conditional or multiple incompatible
behaviors must abstain rather than flatten into an inaccurate edge.

Requests, returned model identity, distributions, confidence, usage/metadata, and
proposals are stored for audit. Stored requests also retain the compact JSON and
the mapping from short evidence labels to immutable source citations; those
sidecars are not sent to the model. Evidence labels in proposals are converted back
to stored citation IDs before persistence. The policy version
`cross-repo-assessment-v2-compact-plaintext` prevents reuse of older prompt results.
Provider confidence is not treated as calibrated
certainty or copied into `connection_decisions.score`; the full answer is retained.
Malformed choices, incomplete distributions, unexpected models, failed HTTP calls,
and timeouts fail the operation rather than becoming negative decisions. No silent
fallback model is enabled. The client retries transport errors and transient HTTP
statuses with bounded exponential backoff (two retries by default). A 180-second
outer deadline bounds each decision request including retries.

Assessment publishes a packet's decisions atomically and reuses identical packet,
candidate, provider/model, question, and policy inputs. Like preparation, it holds
a snapshot lock and transaction during model calls. Dependency tracking includes
all packet paths, missing paths, and endpoint files; proposal/request/response
mutation invalidates recorded reuse. Cross-snapshot reuse is conservative because
candidate-set fingerprints currently include snapshot-specific site IDs.

Assessment writes decisions and proposals only. The separate publication
operation materializes `cross_connections` and resource rows. Decision-model quality is not established by mocked transport tests;
live credentials and a labeled cross-repository evaluation set are still needed
for deployment validation. The adapter makes real paid requests only when called
by the application; no live model request is part of the test suite.

## Publication and combined queries

`publish_graph(pool, snapshot_id, decision_ids, branch_ids=())` replaces active
membership with **exactly** the explicitly selected decisions. It rejects
unsupported/untracked decisions, mixed snapshots or site-index versions, multiple
decisions for one candidate, invalid endpoints/verbs, and inputs changed since
assessment. A revision may be explicitly empty to retract all active connections.
Supported decisions not selected for the new revision are not carried forward.

Resource proposals become nodes scoped to the candidate endpoint pair, context,
and source evidence anchor. Both legs publish together. Matching topic names do
not automatically unify resources across different candidate pairs; broader
resource reconciliation needs additional identity evidence. This conservative
scope avoids accidentally merging environments, at the cost of duplicate resource
nodes for some real fan-out interactions.

Edge materialization, revision insertion, and active-pointer replacement commit in
one transaction. A failure leaves the prior active graph intact. Repeating the
same inputs reuses the existing immutable revision; selecting an older set can
reactivate it. Concurrent writers may receive a PostgreSQL serialization error
and should retry the operation. There is no automatic model call during publication.

Revisions copy nodes, edges, decision provenance, conditions, and evidence excerpts
into an immutable payload. Later parser changes, decision changes, or active-set
replacement do not rewrite that history. Updates to revision rows are forbidden;
deleting the parent snapshot still removes its graph history. `cross_connections`
is a historical materialization pool, **not** the active view: read through
`read_graph(pool, snapshot_id=...)` or `read_graph(pool, revision_id=...)`.

Optional parser branches add resolved local CALLS links and their recorded caller
membership (HAS_CALL). These come from existing parser tables, not model inference.
Only one parser branch per snapshot repository is allowed; every indexed local
file must match a captured path and content hash. Dependency-only and unresolved
calls are excluded. Exact path/hash/span/kind matches join parser nodes to
interaction sites. Missing parser coverage stays missing; no guessed links fill it.

Parser resolution is explicitly marked `byte_matched_branch_resolution`: matching
source bytes does not certify all environment/configuration inputs used by an
older resolver run. Confidence and parser edge/branch IDs remain on each captured
link. The combined graph freezes those links at publication time. Inheritance and
data-access exploration remain available through the existing graph CLI; this
combined view currently includes parser calls and cross-repository interactions.

`neighborhood(graph, node_id, hops=1, direction="both", include_parser=False,
max_nodes=200)` traverses the published payload. Parser links require explicit
inclusion. Resource legs retain their directed verbs; a two-hop bidirectional
traversal finds both participants without synthesizing an INVOKES edge. Direction,
hop (maximum 10), and node (maximum 5,000) limits are validated, with node-budget
truncation reported. Evidence for returned decisions is included. The implementation
loads the full revision JSON before filtering; large graphs will need indexed or
streamed storage before this becomes a large-corpus query service.

## CLI workflow

Use `python -m cli.connections` or the installed `trident-connections` entry point.
All output is JSON. As with the other raw CLIs, configure `DATABASE_URL` and apply
migrations first. Repository IDs and checkout roots must already be registered.
Each command prints IDs needed by the next command:

```sh
python -m cli.connections capture --repo 1=HEAD --repo 2=HEAD --context '{"environment":"dev"}'
python -m cli.connections index --snapshot SNAPSHOT_ID
python -m cli.connections embed --snapshot SNAPSHOT_ID --index INDEX_VERSION
python -m cli.connections retrieve --snapshot SNAPSHOT_ID --index INDEX_VERSION
python -m cli.connections prepare --run RETRIEVAL_RUN_ID --context-path 1=config/routing.yaml
python -m cli.connections assess --packet PACKET_ID
python -m cli.connections publish --snapshot SNAPSHOT_ID --decision DECISION_ID
python -m cli.connections show --snapshot SNAPSHOT_ID
python -m cli.connections neighbors --snapshot SNAPSHOT_ID --node site:SITE_ID --hops 2
```

`capture` rejects single-repository input before reading Git. `embed` uses the
existing EMBEDDING_* configuration with a dedicated `site-v1` identity and closes
its HTTP client afterward. `retrieve` uses the same model/version/dimensions but
needs no API key. `prepare` offers Jev as a concrete reranker: per target it asks a
typed relevance choice and converts its distribution into an ordinal score for
selection. This is retrieval relevance, not acceptance confidence. It is serial
and may be slower than a dedicated batched reranker; the Python API still accepts
other rerankers. Preparation prints packet IDs. `assess` prints decision IDs;
select supported decisions for publication after inspecting their assessments.

`embed`, `prepare`, and `assess` make paid model calls; their help text identifies
this. Jev configuration is shared with assessment. No command runs a model during
`show`, `neighbors`, or `publish`. Repeat `--decision` for full desired membership;
use `--empty` for an explicit retraction. Add `--parser-branch ID` at publication,
then `--include-parser` for combined neighborhoods. Use `--revision ID` instead of
`--snapshot ID` to inspect historical revisions. Component commands are described below. Neptune orchestration remains separate work.

## Component membership and projection

A component graph is derived from one immutable published graph revision. It makes
no model calls and infers no new relationships. By default, code nodes belong to
one component per canonical repository. Optional `ComponentRule(repo_id, key,
label, prefix="")` entries partition a repository into named services, packages,
or other components using explicit path boundaries.

Rules match a complete path or a directory prefix followed by `/`; `src/pay` does
not match `src/payments`. The longest matching prefix wins. An empty prefix covers
the whole repository, and unmatched paths use the repository fallback. Multiple
prefixes may share a component key and label. Conflicting assignments at one
prefix or conflicting labels for one key are rejected. Keys are repository-scoped,
so a shared label never collapses nodes from different repositories. There are no
implicit service guesses, glob rules, or protocol-specific manifest handlers.

Membership is recorded for code nodes present in the published graph, including
optional captured parser nodes; it is not a full inventory of every file or symbol
in a repository. Rules matching no published nodes are reported as unused. Component
nodes may be isolated when only parser nodes belong to them. Parser edges are
excluded from the component connection graph, keeping projected relationships
strictly cross-repository. No CONTAINS or similarity edges are generated.

Connections aggregate only when component source, component/resource target, verb,
vocabulary version, and conditions all match. Different environments/conditions,
verbs, or directions stay separate. Resource nodes retain their published identity
and direction; PRODUCES/CONSUMES legs never become an invented direct call. The
existing candidate-pair resource scope is retained, without cross-pair merging.

Each projected edge stores original connection indices and decision IDs plus counts.
These counts describe stored connections and distinct decisions, not independent
corroboration or confidence. No averaging of model confidence occurs. Drill-down
returns contributing connections, all connection legs for their decisions, endpoint
metadata, resource anchors, conditions, and the exact historical evidence excerpts.
This preserves both resource participants even when drilling into just one leg.

```python
from core.connections.components import (
    ComponentRule, create_projection, read_projection, explain_component_edge,
)

projection_id = await create_projection(pool, revision_id, (
    ComponentRule(1, "billing", "Billing service", "services/billing"),
    ComponentRule(2, "reporting", "Reporting worker", "src/reporting"),
))
view = await read_projection(pool, projection_id)
detail = await explain_component_edge(pool, projection_id, edge_id)
```

The projection stores its parent graph revision ID/fingerprint, rules, memberships,
lightweight nodes, and aggregate edges in one immutable database record. Repeating
the same graph and rules reuses it; changing rules or parent revision creates a
new projection. Activating a newer code graph does not rewrite old projections or
change their drill-down evidence. Deleting the parent snapshot/revision cascades to
its projections. Projection currently loads full graph JSON and stores membership
maps in JSON; large-corpus indexing/streaming remains future work.

The CLI accepts a JSON list of rules (omit `--rules` for repository defaults):

```json
[
  {"repo_id": 1, "key": "billing", "label": "Billing service", "prefix": "services/billing"},
  {"repo_id": 2, "key": "reporting", "label": "Reporting worker", "prefix": "src/reporting"}
]
```

```sh
python -m cli.connections project --revision REVISION_ID --rules components.json
python -m cli.connections components --projection PROJECTION_ID
python -m cli.connections component-neighbors --projection PROJECTION_ID --node COMPONENT_ID --hops 2
python -m cli.connections explain-component-edge --projection PROJECTION_ID --edge EDGE_ID
```

`components` omits the code-node membership map by default; add `--memberships` to
inspect it. Neighborhoods support direction and node limits with truncation flags,
using the same bounds as code-graph queries. Resource traversal retains directed
verbs. No command in this projection sequence uses an embedding or decision model.
An empty graph yields an empty projection; it is not evidence that the repositories
have no interactions.

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

8. Neptune orchestration, caching, and held-out-family evaluation.


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

Jev HTTP contract, abstention, proposal direction, persistence, and failure coverage
live in `tests/test_connection_assessment.py`.

Publication atomicity, historical graph isolation, parser capture, resource
direction, and a mocked end-to-end CLI flow are covered by
`tests/test_connection_publication.py`.

Component rules, aggregation boundaries, historical drill-down, resource direction,
and CLI behavior are covered by `tests/test_connection_components.py`.
