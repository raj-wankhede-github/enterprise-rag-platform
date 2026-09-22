# Decisions

Why things are the way they are. Each entry records the alternative that was rejected, because
that is the part that gets forgotten and re-litigated.

## Search engine: OpenSearch, one engine for BM25 and dense kNN

Rejected: Postgres + pgvector + Qdrant (two systems, and the weakest BM25 of the options);
Vespa (most powerful ranking, steepest learning curve, least common operational knowledge).

OpenSearch gives the strongest text analysis of the realistic options, hybrid search in one
engine, and is self-hostable — which BYOC requires.

## Index topology: pooled shared indices, not index-per-tenant

Rejected: **index-per-tenant** — every index is at least one Lucene index per shard with fixed
heap overhead and a cluster-state entry. 2,000 tenants is 2,000 shards, most under 50 MB, which
melts the master node; and tiny HNSW graphs give poor recall-per-byte. Rejected: **one global
index** — a per-tenant reindex becomes a full-index operation.

Chosen: 16 pools, tenant assigned by consistent hash and the binding *recorded* in Postgres
(never recomputed, since rehashing would orphan chunks). `_routing = tenant_id` so a tenant's
query touches one shard — the biggest latency win, and the reason the tenant filter is not
selective *within* the searched shard, which is what otherwise wrecks filtered HNSW recall.
Promotion to a dedicated index above ~2M chunks is a background reindex plus an alias repoint;
the application only ever talks to aliases, so no code changes.

## Fusion: RRF in application code

Rejected: the OpenSearch `hybrid` query with the normalization processor. It normalizes *scores*
rather than fusing *ranks*, and its per-sub-query depth is governed by `pagination_depth`, which
caps recall@50 — the one metric no later stage can recover.

Application-side RRF costs the same single `_msearch` round trip and additionally: fuses four
legs (and legs × sub-queries when a question is decomposed), fuses across the child and parent
indices, makes weighted RRF a Postgres row rather than a mutation to a cluster-level pipeline,
and is a pure function testable with no cluster. `NativeHybridRetriever` implements the same
Protocol and appears as its own ablation row, so the claim stays measured rather than asserted.

## k = 60

Cormack, Clarke and Buettcher (2009). It damps any single leg's top ranks, so a document ranked
about 5th by two legs beats one ranked 1st by a single leg — the behaviour we want when legs
disagree because the query is ambiguous.

## A separate exact-token leg

`exact` is its own leg rather than a field boost inside `bm25`, so the ablation table can show
precisely what identifier matching buys. It uses a whitespace tokenizer with
`word_delimiter_graph` (`preserve_original` plus `catenate_all`) as both index and search
analyzer, so `AB1234X`, `AB-1234/X` and `ab 1234 x` all match the same indexed token set.

## Zero chunk overlap

Overlap exists to recover context lost at chunk boundaries. Parent expansion recovers it
properly, and zero overlap keeps near-duplicate detection and token accounting honest. This is a
deliberate trade against the common default, and it is an ablation row rather than an assumption.

## Reranking: ONNX in a separate container, not torch in the API image

torch CPU installs at 1-2.5 GB; ONNX Runtime is about 150 MB and 1.5-3x faster after int8
dynamic quantization. Running it as a separate process also keeps model inference off the async
event loop. The same container serves the embedder: one runtime, one warmup path, one artifact to
ship into an air-gapped registry.

`jina-reranker-v1-turbo-en` is excluded despite good latency: **CC-BY-NC**, unusable in a
commercial product.

The 100-300 ms budget holds only with four levers held simultaneously — 288-token truncation,
top-24 rather than top-50, int8, and one batched call. `bench/rerank_bench.py` guards it, because
this is exactly the budget that rots silently.

## Queue: Postgres SKIP LOCKED, not Redis or Celery

The decisive property is not throughput. It is that **enqueue commits in the same transaction as
the business row**, so there is no window where a document is saved but its job is lost. A Redis
queue needs an outbox table to match that. Redis is still used for rate limiting and caching,
where its semantics are the right ones.

## Tenancy: shared schema, not schema- or database-per-tenant

Rejected: schema-per-tenant (migrations times N, `search_path` juggling, pool fragmentation) and
database-per-tenant (N migrations, N pools, N backups).

The decisive argument is that **BYOC must be a packaging exercise, not a code branch**: shared
schema runs unchanged with one tenant row in a customer VPC, while schema-per-tenant forces two
tenant-resolution paths. The isolation objection is answered with RLS plus tests rather than with
topology. Database-per-tenant remains available as a "dedicated instance" premium tier — same
code, single-tenant deployment.

## 404, never 403, for foreign resources

A 403 confirms the resource exists, which is itself a cross-tenant leak.

## Document identity is not the content hash

The same bytes legitimately exist as two documents with different owners, ACLs and metadata.
Identity is `(tenant_id, source_system, external_id)`; content identity is tracked separately at
three hash levels. See `app/ingestion/versioning.py`.

Consequence worth stating plainly: **storage and compute are deduplicated; index entries are
not.** Two documents holding the same bytes share a blob, a parse, their context lines and their
embeddings, but their chunks are indexed separately because their ACLs differ and must be
independently filterable. Assembly-time near-duplicate suppression means a reader never sees the
passage twice.

## Users are unique per tenant, not globally

Consultants hold accounts in several tenants. A globally unique email cannot be relaxed later
without a migration that has no correct answer for the rows that already collided, so this is
decided on day one.

## SAML via a self-hosted Keycloak broker

Rejected: `python3-saml` and `pysaml2` — both depend on the `xmlsec1` native library, which has
no usable Windows wheels, and XML signature wrapping is a classic place to build a critical
vulnerability. Rejected harder: hand-rolling assertion parsing.

The product speaks only OIDC; Keycloak speaks SAML to the customer IdP. Self-hostable, so BYOC is
unaffected, and Keycloak is in the dev compose regardless because it is the only sane way to test
OIDC and SAML locally.

Worth knowing when planning: Entra ID and Google Workspace both speak OIDC, so SAML is rarely the
blocker it appears to be. Google Workspace *groups*, however, are not in the ID token and require
an Admin SDK call with domain-wide delegation. Budget for it.

## Tracing: OpenTelemetry as the wire, Langfuse as an optional backend

The product keeps its **own** `answer_traces` table, because a customer-facing trace view cannot
depend on an ops tool's availability or RBAC. Langfuse and Phoenix are exporters, not
dependencies — a BYOC customer who wants only Jaeger drops them and loses nothing structural.
Content capture is **off by default**, which is what lets the answer to "does the vendor store
our documents in a third-party observability tool" be "no".

## The evaluation harness is built before the reranker

The reranker, contextual chunking and layout-aware parsing are each a *claim* about quality. A
claim without the harness is a guess that gets defended socially rather than empirically. The
harness lands at step 6 of the build order, before any of them.
