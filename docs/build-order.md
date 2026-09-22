# Build order

Each step ends in something measurable. Steps 0-6 are the critical path to a number; nothing
after step 6 ships without the ablation table showing it earned its latency.

| # | Step | Exit criterion | State |
|---|---|---|---|
| 0 | Skeleton: uv, ruff/mypy/pytest, compose, CI | stack healthy, gate green | **done** |
| 1 | Tenancy spine: `TenantScoped`, ContextVar, ORM guards, RLS | cross-tenant suite passes with guards on *and* with RLS alone | guards + registry test done; RLS migration pending |
| 2 | Roles and capabilities | exhaustiveness + strict-nesting tests | **done** |
| 3 | `search/`: mappings, generations, router, `dsl.py` | every leg x profile x filter permutation carries exactly one tenant term | **done** (router pending) |
| 4 | Ingestion v1 + dedup and versioning | identical re-upload creates no version; edited re-upload re-embeds under 5% of chunks; case 4 returns 409 | **done** (loaders, chunker, contextualiser, embedder, reuse) |
| 5 | Retrieval: four legs and one `_msearch` | candidates carry per-leg ranks | **done** |
| 6 | **Eval harness, golden set, ablation runner, CI gate** | the table prints in under four minutes | **done** -- 20s, gating CI |
| 7 | Answer path: assembly, extractive generator, deterministic verification | citation-support and abstention metrics appear in the table | **done** -- made_up 0.300 -> 0.200, `cite_ok` in the table |
| 8 | `models` container, cross-encoder reranker, `rerank_bench.py` | a real nDCG delta, p95 within budget | **done** -- nDCG@10 0.832 -> 0.895 for +12 ms |
| 9 | Query understanding: rules, fast path, then one LLM call | fast path p95 under 60 ms | **done** -- same quality, lower p95 |
| 10 | Contextual retrieval: template, then LLM with prompt caching | `+contextual` row, measured cost per 1k chunks | **done** |
| 11 | `parser` container (Docling), OCR, tables | scanned-PDF and table strata pass | **done** (container not built locally -- see note) |
| 12 | Generation rebuild and atomic alias swap, shadow-evaluated | zero errors and zero empty results under continuous traffic | **done** (integration suite written, not yet run -- see note) |
| 13 | SSO: discovery, OIDC, JIT provisioning, admin wizard | round trip against Keycloak and a real Entra tenant | **done** (no live IdP round trip yet) |
| 14 | Frontend: routes, AuthContext, login, search and ask, admin | capability-gated nav, every error state demoed | pending |
| 15 | API keys, Redis rate limiting, concurrency caps | limits hold across two replicas | pending |
| 16 | Platform operator plane: separate app, host and audience; support grants | an operator cannot read tenant content without a grant | pending |
| 17 | Observability, audit UI, SharePoint connector | a restricted file is invisible to a non-member, end to end | pending |
| 18 | SAML via the broker, SCIM, BYOC packaging | isolation suite green against the BYOC compose file | pending |

Steps 1 and 2 are non-negotiably first: every later table and route inherits their shape, and
retrofitting `tenant_id` and capability gating is the most expensive rewrite available.

## Why the eval harness comes before the quality features

Steps 8, 10 and 11 — reranking, contextual chunking, layout-aware parsing — are each a claim that
something improves answers. Without step 6 there is no way to test the claim, so the decision
gets made by whoever argues best. With it, each one is a row in a table with a latency column
next to it, and a feature that does not earn its milliseconds gets removed.


## What the harness measured, and what it refuses to claim

First run, 14 documents / 71 chunks / 61 questions, offline triple:

```
config                   recall@10  recall@50  nDCG@10  MRR@10  abstain  made_up  leak
bm25_only                0.500      0.500      0.467    0.467   0.900    0.100    0.000
dense_only               0.853      0.961      0.680    0.635   0.700    0.300    0.000
hybrid_rrf               0.912      0.980      0.791    0.765   0.700    0.300    0.000
hybrid_rrf + contextual  0.941      0.990      0.832    0.818   0.700    0.300    0.000
```

Three things worth stating plainly, because the table is easy to over-read:

1. **recall@50 is not evidence at this corpus size.** 50 of 71 chunks is 70% of the index, so a
   semantically blind hashing embedder scores 0.96. `recall@10` is the honest column, and the
   report prints a warning until the corpus is large enough for the deeper cutoff to
   discriminate. Growing the corpus is the single highest-value improvement to this harness.
2. **`dense_only` beating `bm25_only` says nothing about embedding quality.** The hashing
   embedder has no semantics; it wins here because it retrieves broadly on a tiny corpus. The
   ablation measures the *pipeline*, not the model.
3. **`made_up` is 0.300 and that is the honest number.** `bench/abstention_calibration.py` shows
   no coverage threshold separates the remainder: driving it to zero means refusing 39% of
   answerable questions. The residue is questions whose topic terms are all present and only the
   asked-for value is absent. Citation verification at step 7 is what closes it.


## Step 7: what was predicted, and what was measured

When the harness landed I predicted citation verification would close the 0.300 made-up rate.
That was **half right**, and the half that was wrong is worth recording.

`made_up` did fall, 0.300 -> 0.200. But it fell because fixing the harness exposed a dead
identifier rule (see below), **not** because of verification. The `+ verified` row scores
identically to the row without it.

The reason is structural. CI's generator is extractive: it quotes verbatim and cannot fabricate,
so verification has nothing to catch. Its value is proven in
`tests/unit/test_answer_path.py::test_pipeline_rejects_a_fabricated_value_and_abstains`, where a
deliberately fabricating generator states a per-diem for a grade the table does not list and is
caught with no model involved. Against a real LLM that matters; against the extractive floor it
is a no-op.

The two questions still answered are a **different failure**, and `citation_support` is 1.0 for
both, correctly:

- *"what is the per diem for a grade D destination"* -> quotes a real sentence about destination
  grades that never states grade D's rate.
- *"what is the retention period for board minutes"* -> quotes real sentences about retention
  periods that never mention board minutes.

The evidence is quoted faithfully and simply does not answer the question. That is an entailment
judgement, not a citation check, and no deterministic rule sees it. The cross-encoder arriving
with the models container at step 8 is the thing that can, which is why the `unsupported_answer_rate`
floor stays at 0.25 until that is measured rather than being asserted now.


## Step 8: reranking earns its latency; entailment does not close the gap

Measured with the lexical reranker, which is what CI can run offline:

```
config                               recall@10  nDCG@10  MRR@10  made_up  cite_ok  p95ms
hybrid_rrf + contextual              0.941      0.832    0.818   0.200    1.000    26
hybrid + contextual + rerank         0.980      0.895    0.889   0.200    1.000    37
hybrid + contextual + verified       0.941      0.832    0.818   0.200    1.000    154
hybrid + rerank + verified + entail  0.980      0.895    0.889   0.200    1.000    137
```

**Reranking earns its place.** +6.3 points of nDCG@10 and +7.1 of MRR@10 for about 12 ms, from a
reranker with no idea what a sentence means. That is the floor; the cross-encoder should beat it,
and `bench/rerank_bench.py` is what will say whether it does so inside the latency budget.

**Entailment did not move the made-up rate**, and the reason is now measured rather than
suspected. CI's generator is extractive, so a claim *is* a quote from its own evidence: lexical
entailment scores it 1.0 by construction, and `citation_support` is legitimately 1.0. There is a
test asserting exactly this limitation so the proxy is never mistaken for a model.

A correction to the step 7 write-up, which was right by accident: the `verified` row scored
identically to the plain row because **the answer path was never wired into the eval runner** --
a patch was silently lost. It is wired now (p95 26 -> 154 ms proves it), the numbers are
unchanged, and the conclusion stands for the right reason.

The two surviving cases need a judgement of *"does this evidence answer the question"*, not
*"does this evidence support the claim"*. Those are different questions and only the second was
being asked. A real cross-encoder scoring the question against the cited evidence is the thing
that can answer the first.


## Step 9: the fast path costs nothing in quality

```
config                        recall@10  nDCG@10  MRR@10  p95ms
hybrid + contextual + rerank  0.980      0.895    0.889   46
planner (fast path on)        0.980      0.895    0.889   31
```

Identical quality, lower p95. That is the claim the fast path makes and the only one it is
allowed to make: it is a latency optimisation, so any quality it cost would be a regression
rather than a trade. The thresholds file now encodes that, so a fast path that starts losing
recall fails the build.

**The fast path fires on 5 of 61 questions (8%)**, not the 15-30% quoted for real traffic. That
is not a contradiction and not a disappointment -- the golden set is deliberately question-shaped
to exercise retrieval, so it under-represents the bare-identifier lookups ("TKT-99812",
"SUP-4471") that dominate a real deployment's cheap tail. The 8% is what this corpus measures;
the traffic figure is an expectation that only a real deployment can confirm.

Two bugs the tests caught, both of which would have been silent:

- `SEC-4.2.1` was not recognised as an identifier. The code pattern required two leading digits,
  so section references fell through to dense retrieval on an opaque code -- the single worst
  case for embeddings, and precisely what the exact leg exists to prevent.
- `"previous"` was missing from the historical-intent words (only `"previously"` was there), so
  "the previous policy" would have been answered from current documents only.

The eval runner now calls the real planner instead of the local regex that approximated it.
Measuring an approximation of the shipping path is how an evaluation drifts away from the system
it claims to describe.


## Step 11: the parser, and a note on what was not verified

The Docling client, the container and the routing are written and unit-tested. **The container
was not built or run on this machine** -- the Docker VHDX had grown to 57 GB and filled the disk,
and the Docling image (torch, vision models, OCR language packs) is the largest in the stack by
a wide margin. So the mapping from Docling's output to `ExtractedDocument` is tested against
recorded response shapes rather than against a live parser, and that distinction is worth
keeping in mind: it verifies the contract, not the parser's fidelity.

The routing is the part that matters most, and it is fully tested. A scanned PDF sent to the
cheap loader produces a document of empty chunks that indexes cleanly and retrieves nothing --
an ingest that succeeds and a document that never comes back from a search. `chars_per_page` is
what catches it.

One deliberate asymmetry with the reranker: **a parser outage raises rather than degrading.**
There is no cheaper way to read a scan, so failing the job is correct -- it retries, and the
document appears as a failed ingest. Degrading would produce exactly the silent empty document
the routing exists to prevent.


## Step 12: the rebuild, and what has not been run

The orchestration, the two gates and the swap are complete and unit-tested (66 new tests). The
integration suite `tests/integration/test_generation_swap.py` -- which is where the acceptance
criterion actually lives, because "zero empty results under continuous traffic" is a property of
how OpenSearch applies alias actions rather than of our control flow -- **has been written but
not executed**, because the machine has no disk left for the cluster. It should be the first
thing run once space is free:

```bash
docker compose up -d opensearch postgres
cd backend && uv run pytest -m integration -q
```

Two bugs the unit tests caught that are worth recording, because both would have failed a real
rebuild only after the backfill had already run:

* The backfill wrote `simhash` where the mapping declares `simhash64`, and wrote it as an integer
  where the mapping declares a keyword. Under `dynamic: strict` the first is a rejected bulk
  item; the second is silently coerced, which is worse.
* The parent body wrote `ordinal`, but the parent mapping renames it to `parent_ordinal` and adds
  `child_count`. The parent index is not the chunk index minus a vector.

Both were found by asserting each body's field set against the mapping itself rather than against
the ingest path, which is now the test that guards this: `tests/unit/test_backfill_source.py`.

A third thing worth recording is a design error caught while writing it. The first cursor design
paged on the OpenSearch `_id`, which is a SHA-256 -- meaning every page would have computed a
hash per candidate row and ordered randomly against every index the table has. Paging is now
keyset on `(tenant_id, document_version_id, ordinal)`, and the cursor is opaque to the
orchestrator so a source owns its own key space.


## Step 13: SSO

Discovery, the OIDC code flow with PKCE, JIT provisioning with role mapping, session and refresh
handling, CSRF, and the admin wizard's claim preview. 159 new tests.

What is **not** done: the live round trip against Keycloak and a real Entra tenant. That needs
containers this machine has no disk for. Everything either side of the network call is tested --
the authorization request, state signing, the token exchange against a mock transport, claim
extraction, role resolution and the provisioning decision -- so what remains untested is
specifically the provider's own behaviour.

Decisions worth not rediscovering:

* **Identity is matched on the IdP's immutable subject, never on email.** Entra's `oid` before
  `sub`, because `sub` is pairwise per application and changes if the app registration is
  recreated. Email is consulted in exactly one place: linking an administrator-created account on
  its *first* SSO login, and only when the provider marks the address verified.
* **JIT can never create an ADMIN.** A new user gets the config's `default_role`, which a
  database CHECK forbids from being ADMIN. Group mappings apply to existing users only. The first
  ADMIN of a tenant is created by a human.
* **Highest matching role wins**, not first match — otherwise the outcome depends on dictionary
  ordering. An existing user whose groups stop matching keeps their role rather than falling back
  to the default, which would silently demote every administrator the day a claim name changes.
* **Every password-login failure is identical** in message, status and response time. The real
  reason goes to `login_attempts`, where an administrator can see it and an attacker cannot.
* **State is signed and self-contained, not a cookie.** A cookie-dependent callback fails
  intermittently in Safari, in in-app webviews, and anywhere a SameSite rule drops it on the way
  back from the provider — failures that are close to unreproducible.
* **Google Workspace groups are not in the ID token.** They need a Directory API call with
  domain-wide delegation, and the wizard's preview says so explicitly rather than leaving an
  administrator to discover it after activation.

One bug found by a route test rather than by review: raising an `AuthenticationError` after
clearing cookies discarded the `Set-Cookie` headers, because the exception handler builds its own
response. A failed refresh therefore left the dead token in the browser and every subsequent
request retried with it — a loop the user could not escape without clearing site data. `AppError`
now carries `clear_cookies` and the handler applies it.


## Step 14: the frontend

React 19, TypeScript 5.9, Vite 7, Tailwind 4, Vitest. 56 tests; lint, typecheck and production
build all clean. Routes are code-split; the entry bundle is 86 KB gzipped.

Four decisions worth keeping:

* **The capability table is generated from the backend enum** (`uv run python -m app.cli
  capabilities`) and a test regenerates it rather than comparing two checked-in copies — two
  copies drift together and prove nothing. CI installs uv in the frontend job for that one test,
  because the version that skips silently would be a no-op exactly where it matters.
* **One refresh at a time.** A page firing six requests on mount would otherwise start six
  refreshes; five present a token the first has already rotated, the backend correctly reads that
  as replay, and the session family is revoked. The user is signed out of every device by loading
  a page — and the backend is behaving exactly as designed, so nothing on that side reports a
  problem. The single-flight in `api/client.ts` is what makes refresh-token reuse detection
  compatible with a real application.
* **`/api/auth/me`, not `/api/auth/refresh`, for the session probe.** Probing via refresh would
  burn a rotation per page load, and two tabs opened at once would each present a token the other
  had rotated.
* **Abstention is rendered as an answer, not an error.** No alert role, no red. Styling the
  honest refusal as a malfunction teaches users to distrust it, and the pressure that follows is
  to make the system answer anyway — which is the failure the product exists to avoid.

ESLint enforces two of these structurally: `dangerouslySetInnerHTML` is banned outright (a remote
image in a retrieved passage is the classic exfiltration channel) and `localStorage` /
`sessionStorage` are banned in production code, since the session lives in HttpOnly cookies.
