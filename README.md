# Enterprise RAG Platform

Multi-tenant enterprise knowledge search: hybrid retrieval, RRF fusion, cross-encoder reranking,
and grounded answers where every claim carries a citation — or the system says the answer is not
in the sources.

Built for production enterprise scale and sold as multi-tenant SaaS, with every dependency
self-hostable so a customer-VPC deployment is a packaging exercise rather than a rewrite.

## Status

Early. The foundation is in place and green; the retrieval and ingestion pipelines are being
built on top of it in the order set out in [`docs/build-order.md`](docs/build-order.md).

| Area | State |
|---|---|
| Roles, capabilities, rank model | done, invariant-tested |
| Tenant/ACL filter chokepoint (`search/dsl.py`) | done, permutation-tested |
| RRF fusion | done, unit-tested |
| Document identity, dedup, versioning policy | done, fully specified by tests |
| Data model + ORM tenant guards | done |
| Ingestion pipeline, OpenSearch indices, retrieval legs | next |
| Evaluation harness + ablation table | after that, before any tuning |
| SSO, connectors, admin UI | later |

## Quick start

```bash
cp .env.example .env
docker compose up --build            # API on :8001, docs on :8001/docs
```

Backend development, from `backend/`:

```bash
uv sync
uv run pytest -q                      # no database, no network, no model download
uv run ruff check . && uv run ruff format --check . && uv run mypy app
uv run uvicorn app.main:create_app --factory --reload
```

Optional services, only when you need them:

```bash
docker compose --profile models up -d   # ONNX embedder + cross-encoder reranker
docker compose --profile parse up -d    # Docling parser for scans and tables
docker compose --profile dev up -d      # Keycloak (OIDC/SAML testing), MinIO, Dashboards
```

## How it answers a question

```
Query ─► Query understanding ─► [fast path for identifier lookups?] ─┐
                                                                     ▼
              ┌──────────── Hybrid retrieval (one _msearch) ────────────┐
              │ BM25 analyzed │ BM25 exact │ dense kNN │ parent BM25    │
              └────────┬──────┴──────┬─────┴─────┬─────┴───┬────────────┘
                       └────► RRF fusion (k=60, application-side) ◄──────┘
                                       │  top ~50
                                       ▼
                        Cross-encoder rerank ──► top ~24
                                       ▼
        Assembly: parent expansion, near-duplicate removal, budget packing
                                       ▼
                 Generation with a chunk-ID citation on every claim
                                       ▼
            Citation verification, or an honest "not in the sources"
```

## Design decisions worth knowing

- **OpenSearch is the single engine** for both BM25 and dense kNN. Pooled shared indices with
  `_routing = tenant_id`, promoted to a dedicated index above ~2M chunks.
- **RRF runs in application code**, not in OpenSearch's `hybrid` query: we fuse four legs across
  two indices, need per-leg depth control (`pagination_depth` caps recall@50), and want fusion
  testable without a cluster. `score(d) = Σ w_i / (k + rank_i(d))`, k = 60.
- **The exact-token leg is separate from BM25** so the ablation table can show precisely what
  identifier matching buys. Dense embeddings are unreliable on `AB-1234/X`; BM25 is not.
- **Every ACL clause goes inside the query**, in both `bool.filter` and `knn.filter`. Post-filtering
  loses results for narrow-ACL users and leaks which documents exist.
- **Generation fingerprinting** makes a half-built index invisible rather than blended, so a
  stalled rebuild fails a shadow evaluation instead of quietly degrading relevance.
- **The evaluation harness is built before the reranker and before contextual chunking.** Each of
  those is a claim about quality, and a claim without the harness is a guess defended socially.

Longer rationale: [`docs/decisions.md`](docs/decisions.md).

## Repository layout

```
backend/app/
  core/        settings, error taxonomy, request-scoped principal
  db/          declarative base, tenant guards, session factory
  models/      SQLAlchemy models (tenancy is opt-out, enforced by a test)
  security/    roles, capability matrix, principal
  search/      OpenSearch client, mappings, generations — and dsl.py, the filter chokepoint
  retrieval/   legs, RRF fusion, reranking, parent expansion
  query/       rules-first query understanding and the fast path
  ingestion/   loaders, chunking, contextualization, dedup and versioning
  answer/      context assembly, generation, citation verification
  evals/       golden set, metrics, ablation runner
frontend/src/  React 19 + TypeScript + Vite + Tailwind
deploy/        OpenSearch image, Postgres init, Keycloak dev realm
docs/          decisions, build order
```

## Testing

```bash
uv run pytest -q                  # unit: fast, hermetic, deterministic
uv run pytest -m integration -q   # needs Postgres + OpenSearch
uv run python -m app.cli eval --ablate
```

Three test suites matter more than the rest and must never be marked `xfail`:

- `tests/unit/test_capabilities.py` — the four-role chain stays a chain.
- `tests/security/test_dsl_isolation.py` — every query permutation carries exactly one tenant
  term, and the visibility range is an `lte` on the principal's rank.
- `tests/unit/test_versioning.py` — the executable specification of what a re-upload does.

## License

Proprietary. All rights reserved.
