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

**Abstention is a first-class success, not an error.** `app/answer/evidence.py` gates on the
retrieved candidates *before* the generator runs, so a model handed thin evidence never gets the
chance to be fluent about it. `app/answer/abstain.py` holds the messages as plain strings, never
an LLM call — a model asked to phrase a refusal will sometimes answer the question instead. The
messages must state plainly that there is no answer, never grovel, never hedge toward guessing,
say what was searched, offer a next step, and never name a document the user cannot see.
`tests/unit/test_abstention.py` fails the build on any of those.

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
- **The parent leg needs `parent_id` in the parent mapping.** That field is how a section
  projects down to its children; dropping it makes the leg silently link nothing. There is a
  test for it because it has already happened once.
- **`fuse_rrf` tracks what *this* pass counted**, not what is already on `candidate.legs`. The
  parent leg pre-populates that entry while projecting, and an earlier version read it as
  "already counted" and fused every parent-only candidate to zero.
- **A failed leg degrades, never raises.** Losing the dense leg costs recall; raising costs the
  answer. Failures land in `RetrievalDiagnostics.legs[].error`.
- **Priors are applied after fusion**, bounded and multiplicative, so each leg stays
  independently measurable and the prior is its own ablation row.
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

## Embeddings

- **The embedder id and dimension are read FROM the models service, never configured.** Both
  enter the generation fingerprint, so they must describe what actually produced the vectors; a
  replica running an older image would otherwise stamp a plausible label onto a different
  checkpoint's output and the fingerprint would be lying. `build_container_async` resolves it at
  startup, which is why that factory is async.
- **The embedding client does NOT degrade.** Everything else here falls back -- a reranker
  timeout drops to fusion order, an LLM outage to the template. Falling back here writes a second
  vector space into one index, where similarity is meaningless and nothing downstream can tell.
  An outage raises, the job retries, the index stays consistent. A slow ingest is recoverable; a
  poisoned index is a rebuild. **A mid-run change of embedder id is fatal** for the same reason.
- **Pooling and query prefixes come from `embedding_card.json`, baked beside the weights.** BGE
  uses CLS pooling, E5 uses mean; BGE prefixes queries only, E5 prefixes both. Getting either
  wrong does not error -- it silently costs several points of recall.
- **Mean pooling must exclude padding**, or the same text embedded in two differently-shaped
  batches gets two different vectors and the index is inconsistent with itself.
  `tests/unit/test_models_pooling.py` is the guard.
- The server normalizes, not the client: the mapping uses `innerproduct`, which equals cosine
  only on unit vectors.

## Ingestion at volume

- **The queue is Postgres, `FOR UPDATE SKIP LOCKED`, because enqueue commits in the same
  transaction as the business row.** No window where a document exists with no job.
- **The claim re-checks `status = 'QUEUED'` in the UPDATE's own WHERE, not only in the CTE.**
  Under READ COMMITTED, EvalPlanQual re-evaluates the quals of the *locking* query; a predicate
  inside a CTE is not re-checked. Without that line, 8 workers completed 420 of 400 jobs.
- **Fairness is in the claim query**, counting running jobs and this batch's position per tenant.
  One tenant's 200k backfill starving everyone else is the commonest multi-tenant ingestion
  failure.
- A lease, not a lock: a killed worker's document returns to the queue with its attempt count
  intact, so a poison message still reaches `DEAD` rather than cycling forever.

## Auth, keys and limits

- **Identity is matched on the IdP's immutable subject, never email.** Entra's `oid` before
  `sub`, because `sub` is pairwise per application and changes if the app registration is
  recreated. Email is consulted in exactly one place: linking a pre-provisioned account on its
  *first* SSO login, and only when the provider marks it verified.
- **JIT can never create an ADMIN.** A new user gets the config's `default_role`, which a
  database CHECK forbids from being ADMIN. Group mappings apply to existing users only.
- **Highest matching role wins, not first match** -- first-match makes the outcome depend on
  dictionary ordering. An existing user whose groups stop matching keeps their role, or every
  administrator is silently demoted the day a claim name changes.
- **Every password-login failure is identical** in message, status and response time. The real
  reason goes to `login_attempts`.
- **The client refreshes ONCE however many requests 401 at the same moment**
  (`frontend/src/api/client.ts`). Six parallel refreshes means five replays of a rotated token,
  which the backend correctly reads as theft and answers by revoking the family -- signing the
  user out everywhere because they loaded a page.
- **API keys can never hold** `sso:configure`, `user:manage`, `audit:view`, `apikey:manage_any`
  or `export:documents`. Subtracted when the principal is built, not when the key is created.
- **Rate limits and concurrency caps are different controls.** Fifty concurrent `ask` requests
  are all inside a 300/minute budget and all hold an LLM call open. `enforce()` checks every
  scope before recording any, or a tenant-level throttle also drains each user's own allowance.

## The operator plane

Separate table, separate token audience, separate ASGI app on a separate host. `users.role` has
a CHECK listing exactly the four tenant roles, so a platform role is unstorable in a tenant.

**Reading tenant content requires a grant the customer approved** -- a row with a mandatory
expiry, a stated reason and a named approver unless it is break-glass. Scopes are ordered
(`metadata` < `content` < `impersonate`) rather than independent flags, and keeping the metadata
case genuinely cheap is what stops every ticket requesting content access on principle.
`app/platform/grants.py::authorize` is the only decision point, and an action missing from
`_REQUIREMENTS` is refused rather than defaulted.

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
- The frontend's capability table is **generated** (`uv run python -m app.cli capabilities`) and
  a test regenerates it rather than comparing two checked-in copies -- two copies drift together.
- Document text is rendered as text, never markup. ESLint bans `dangerouslySetInnerHTML`, and
  `localStorage`/`sessionStorage` in production code, because the session lives in HttpOnly
  cookies.
- Observability attributes deny by default in both directions: content attributes need the
  tenant's opt-in, and an *unclassified* attribute is dropped outright.

## Reference

`C:\Users\rajwa\Projects\AI-Agent-Customer-Support-RAG` is a sibling project by the same author
with working implementations of chunking, RRF, grounding validation and an eval runner. It is a
**reference to read, not to copy** — this product was deliberately started clean.
