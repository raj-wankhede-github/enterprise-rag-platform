# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this before changing anything —
it encodes decisions that are expensive to rediscover.

## What this is

A **multi-tenant enterprise knowledge search product**, sold to companies: "ask your documents"
over SharePoint, Drive, Confluence and uploads. FastAPI + Postgres + OpenSearch backend
(`backend/`), React/Vite/Tailwind frontend (`frontend/`).

The core invariant: **an answer is returned only if it is supported by retrieved, validated
evidence, and every claim carries a chunk-ID citation.** Do not add a code path that calls the
LLM without evidence, returns unvalidated text, or relaxes validation to make a case pass — add
an eval case instead.

The second invariant: **every retrieval query is filtered by tenant, rank and ACL *inside* the
query.** Never post-filter. See "Tenancy" below.

## This is designed for production scale

The development machine is **never** a design input. Production is 3+ OpenSearch data nodes at
31 GB heap, managed Postgres, S3 and autoscaled model pools. Local runs are the same topology
with smaller numbers, expressed as environment overrides and compose profiles — not a different
architecture. If you find yourself sizing something around a laptop, you are solving the wrong
problem.

## Commands

From `backend/` (uv-managed **Python 3.14** — verify new dependencies ship 3.14 wheels):

```bash
uv sync                                          # install, dev group included
uv run pytest -q                                 # unit tests: no DB, no network, no model download
uv run pytest -m integration -q                  # needs Postgres + OpenSearch
uv run ruff check . && uv run ruff format --check . && uv run mypy app
uv run alembic upgrade head                      # migrations are explicit, never at startup
uv run alembic check                             # models vs migrations drift
uv run python -m app.cli eval --ablate           # the ablation table
uv run uvicorn app.main:create_app --factory --reload
```

Full stack: `cp .env.example .env`, `docker compose up --build`. API on :8001, docs on
:8001/docs. Optional: `--profile models` (ONNX embedder/reranker), `--profile parse` (Docling),
`--profile dev` (Keycloak, MinIO, Dashboards).

## Why Python 3.14 here but 3.12 in the model containers

The API, worker and CLI run 3.14. The `models` (ONNX) and `parser` (Docling) services are
**separate containers behind HTTP contracts** and pin their own runtime, because that is where
the ML wheel ecosystem lives. That decoupling is the reason they are services rather than
libraries — do not collapse them into the backend image.

## Architecture essentials

- **Four retrieval legs, one `_msearch`**: `bm25` (analyzed), `exact` (identifiers — the leg that
  makes `TKT-99812` work), `dense` (kNN), `parent` (whole sections). Fused by **RRF in
  application code**, not OpenSearch's `hybrid` query: we fuse four legs across two indices,
  need exact per-leg depth (`pagination_depth` caps recall@50), want per-tenant weights as a
  Postgres row, and want fusion unit-testable with no cluster.
- **`app/search/dsl.py` is the only place a filter clause is constructed.** Every leg, both
  halves of a hybrid query. The ACL clauses go into `bool.filter` *and* `knn.filter` — putting
  them only in the outer bool post-filters the kNN results, which silently destroys recall for
  narrow-ACL users and leaks document existence through `total`.
- **Generation fingerprint**: a digest over embedder id, chunker version, contextualizer version,
  mapping hash and analyzer hash, stamped on every chunk and asserted as a term filter in every
  query. Wrong-generation documents are invisible rather than blended, so a stalled backfill
  shows up as missing results in a shadow eval, not as quietly worse answers in production.
- **Postgres is truth; OpenSearch is a derived, disposable projection.** Any index can be dropped
  and rebuilt from Postgres. This is what makes alias swaps safe.
- **Offline-first**: `LLM_PROVIDER=none`, `EMBEDDING_PROVIDER=hashing`, `RERANKER_PROVIDER=identity`
  is the CI default. No paid API, no model download, deterministic — which is what keeps the
  ablation table runnable on every PR rather than nightly-and-ignored.

## Roles

Flat chain, one role per user: **ADMIN (40) > DEV (30) > TEST (20) > PROD (10)**. Fixed by the
product owner; do not add a fifth.

- `app/security/capabilities.py` is the single expression of it. **Routes declare capabilities,
  never roles** — a lint test asserts no route module imports `Role`.
- Two invariants are enforced by `tests/unit/test_capabilities.py`: no orphan capabilities, and
  **strict nesting** (each rank is a proper superset of the one below). A capability that breaks
  nesting is someone smuggling in a fifth role — push back rather than relaxing the test.
- Documents carry `visibility_rank`; a user sees `visibility_rank <= their rank`. PROD-level
  documents are the most widely visible, ADMIN-level the most restricted.
- The **platform operator is a different table** with a different token audience on a different
  host. `users.role` has a CHECK listing exactly the four roles, so "PLATFORM" is unstorable in
  a tenant and no JIT-provisioning path can mint one.
- The flat model cannot express "Alice curates HR but only reads Legal". The mitigation is
  **subtractive only**: `user_collection_scopes` narrows, `allowed_groups` carries IdP ACLs.
  Neither can widen, so "one role per user" stays literally true.

## Tenancy — four layers, none optional

1. `ContextVar` principal (`app/core/context.py`). Workers bind it per job; there is no default.
2. `SET LOCAL app.tenant_id` per transaction + Postgres RLS, with the app connecting as a
   non-owner role so `FORCE ROW LEVEL SECURITY` binds.
3. ORM guards (`app/db/guards.py`): SELECTs auto-filtered, INSERTs stamped, `tenant_id` immutable.
4. `TenantRepository` for hand-written SQL and raw OpenSearch DSL, which the ORM cannot reach.

**Foreign resources return 404, never 403** — a 403 confirms the row exists.
`tests/unit/test_model_registry.py` fails the build if a new table is neither `TenantScoped` nor
explicitly listed in `GLOBAL_TABLES` with a reason. Tenancy is opt-out, not opt-in.

## Document identity, dedup and versioning

Identity is `(tenant_id, source_system, external_id)` — deliberately **not** the content hash,
because the same bytes legitimately exist as two documents with different ACLs. The policy lives
in `app/ingestion/versioning.py` as pure functions, fully specified by
`tests/unit/test_versioning.py`. The four cases:

| Case | Behaviour |
|---|---|
| Identical bytes, same document | No version, no work, `deduplicated: true`. Ingestion is idempotent, which is what makes at-least-once job delivery safe. |
| Identical bytes, different document | Blob stored once and refcounted; parse, context lines and embeddings reused. Chunks **are** indexed twice — different ACLs must be independently filterable. Assembly-time dedup stops a reader seeing the passage twice. |
| Same document, changed bytes | New version. Per-chunk hash diff: unchanged chunks keep their vectors and get a metadata-only update; only added chunks are embedded. |
| Content exists, uploader may not see it | `409` with a message that names nothing. A duplicate here would be an ACL bypass, since the copy would be visible at the uploader's lower rank. |

**Behaviour is identical for ADMIN and DEV.** Versioning is a property of the data, not of the
uploader's role; the only role-sensitive parts are that 409 and the visibility ceiling
(`resolve_visibility_rank`). Embeddings are never computed twice for identical text —
`embedding_cache` is keyed on `(model_id, content_sha256)` — and `chunk_vectors` persists fp16
vectors so a generation rebuild re-embeds nothing.

## Conventions

- Ruff (line length 120, formatter owns wrapping) and mypy strict are configured in
  `backend/pyproject.toml`.
- Tunable thresholds belong in `core/config.py` **with a comment explaining the default**, and in
  `.env.example`.
- Raise `app/core/errors.py` classes: safe `message` to the client, log-only `detail`. Handlers
  in `api/errors.py`. A `detail` must never reach a response body.
- Migrations are explicit. Never at app startup.
- Integration tests refuse any database not named `*_test`.
- Keep source ASCII: tool-written `\uXXXX` escapes become literal characters.

## Reference

`C:\Users\rajwa\Projects\AI-Agent-Customer-Support-RAG` is a sibling project by the same author
with working implementations of chunking, RRF, grounding validation and an eval runner. It is a
**reference to read, not to copy** — this product was deliberately started clean.
